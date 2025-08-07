#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Kimi-Audio ASR的TTS音频筛选脚本

根据语音识别结果与groundtruth文本的CER差异来筛选TTS合成音频
CER > 5% 的音频将被标记为需要过滤
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import torch
from jiwer import cer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import shutil


# 导入WeTextProcessing的TN模块
try:
    from tn.chinese.normalizer import Normalizer as ChineseNormalizer
    from tn.english.normalizer import Normalizer as EnglishNormalizer
except ImportError:
    print("警告: 无法导入WeTextProcessing，将使用简单的文本标准化")
    ChineseNormalizer = None
    EnglishNormalizer = None

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KimiAudioProcessor:
    """Kimi-Audio处理器 - 简化版本，专门用于TTS筛选"""
    
    def __init__(self, model_path: str, device: str = "cuda:0", kimi_audio_dir: str = None, gpu_id: int = 0, language: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.model = None
        self.kimi_audio_dir = kimi_audio_dir or "/root/code/github_repos/Kimi-Audio"
        self.original_cwd = os.getcwd()
        self.language = language
    
    def load_model(self):
        """加载Kimi-Audio模型"""
        if self.model is not None:
            return
        
        try:
            # 在子进程中设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                logger.info(f"GPU {self.gpu_id}: 设置CUDA设备")
            
            # 切换到Kimi-Audio目录
            if os.path.exists(self.kimi_audio_dir):
                os.chdir(self.kimi_audio_dir)
                logger.info(f"GPU {self.gpu_id}: 切换到Kimi-Audio目录: {self.kimi_audio_dir}")
                
                # 添加Kimi-Audio目录到Python路径
                if self.kimi_audio_dir not in sys.path:
                    sys.path.insert(0, self.kimi_audio_dir)
            
            from kimia_infer.api.kimia import KimiAudio
            
            # 加载模型
            self.model = KimiAudio(
                model_path=self.model_path,
                load_detokenizer=False,
            )
            
            logger.info(f"GPU {self.gpu_id}: ✓ 成功加载Kimi-Audio模型")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 加载Kimi-Audio模型失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(self.original_cwd)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """使用Kimi-Audio进行语音识别"""
        current_dir = os.getcwd()
        
        try:
            # 确保模型已经加载
            if self.model is None:
                self.load_model()
            
            # 切换到Kimi-Audio目录进行推理
            if os.path.exists(self.kimi_audio_dir):
                os.chdir(self.kimi_audio_dir)
            
            # 准备消息 - 根据语言设置选择提示语
            if self.language == "en":
                prompt = "Please transcribe the audio content to text."
                logger.debug(f"  Kimi-Audio使用英文提示语")
            elif self.language == "zh":
                prompt = "请将音频内容转换为文字。"
                logger.debug(f"  Kimi-Audio使用中文提示语")
            else:  # auto or default
                prompt = "Please transcribe the audio content to text. / 请将音频内容转换为文字。"
                logger.debug(f"  Kimi-Audio使用双语提示语（自动检测）")
            
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": audio_path}
            ]
            
            # 生成参数
            sampling_params = {
                "audio_temperature": 0.8,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
            }
            
            # 生成转录
            _, text_output = self.model.generate(
                messages, 
                **sampling_params, 
                output_type="text"
            )
            
            # 清理文本
            if text_output:
                text_output = text_output.strip()
                # 根据语言设置选择要移除的前缀
                if self.language == "en":
                    prefixes_to_remove = [
                        "The audio says:",
                        "The transcription is:",
                        "Transcription:",
                        "Audio content:",
                        "The audio content is:",
                        "Here is the transcription:",
                    ]
                elif self.language == "zh":
                    prefixes_to_remove = [
                        "音频内容是：",
                        "转录结果：",
                        "转录内容：",
                        "音频内容为：",
                        "这是转录结果：",
                    ]
                else:  # auto
                    prefixes_to_remove = [
                        "The audio says:",
                        "The transcription is:",
                        "Transcription:",
                        "Audio content:",
                        "音频内容是：",
                        "转录结果：",
                        "转录内容：",
                    ]
                    
                for prefix in prefixes_to_remove:
                    if text_output.startswith(prefix):
                        text_output = text_output[len(prefix):].strip()
            
            return text_output or ""
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 语音识别失败 {audio_path}: {e}")
            return ""
        finally:
            try:
                os.chdir(current_dir)
            except:
                try:
                    os.chdir(self.original_cwd)
                except:
                    pass

class TextNormalizer:
    """文本标准化器 - 使用WeTextProcessing"""
    
    def __init__(self):
        self.chinese_normalizer = None
        self.english_normalizer = None
        
        try:
            if ChineseNormalizer:
                self.chinese_normalizer = ChineseNormalizer()
                logger.info("✓ 成功加载中文文本标准化器")
        except Exception as e:
            logger.warning(f"加载中文标准化器失败: {e}")
        
        try:
            if EnglishNormalizer:
                self.english_normalizer = EnglishNormalizer()
                logger.info("✓ 成功加载英文文本标准化器")
        except Exception as e:
            logger.warning(f"加载英文标准化器失败: {e}")
    
    def normalize_text(self, text: str, language: str = 'auto') -> str:
        """标准化文本"""
        if not text:
            return ""
        
        # 自动检测语言
        if language == 'auto':
            # 简单的语言检测：如果包含中文字符，则认为是中文
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                language = 'zh'
            else:
                language = 'en'
        
        logger.debug(f"  文本标准化 - 语言: {language}{'(用户指定)' if language != 'auto' else '(自动检测)'}, 原始长度: {len(text)}")
        
        try:
            if language == 'zh' and self.chinese_normalizer:
                # 中文标准化
                normalized = self.chinese_normalizer.normalize(text)
                if normalized != text:
                    logger.debug(f"    中文TN: '{text}' -> '{normalized}'")
                return normalized
            elif language == 'en' and self.english_normalizer:
                # 英文标准化
                normalized = self.english_normalizer.normalize(text)
                if normalized != text:
                    logger.debug(f"    英文TN: '{text}' -> '{normalized}'")
                return normalized
            else:
                # 简单的后备标准化
                logger.debug(f"    使用简单标准化 (无WeTextProcessing)")
                return self._simple_normalize(text, language)
        except Exception as e:
            logger.warning(f"文本标准化失败: {e}，使用简单标准化")
            return self._simple_normalize(text, language)
    
    def _simple_normalize(self, text: str, language: str) -> str:
        """简单的文本标准化"""
        import re
        
        if language == 'zh':
            # 中文：去除标点符号和空格
            text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        else:
            # 英文：转小写，去除标点符号
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = ' '.join(text.split())  # 标准化空格
        
        return text.strip()

class TTSFilterProcessor:
    """TTS音频筛选处理器"""
    
    def __init__(self, config: Dict, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # 语言设置
        self.language = config.get('language', 'auto')
        
        # 初始化Kimi处理器
        self.kimi_processor = KimiAudioProcessor(
            model_path=config['kimi_model_path'],
            device=self.device,
            kimi_audio_dir=config.get('kimi_audio_dir'),
            gpu_id=gpu_id,
            language=self.language
        )
        
        # 初始化文本标准化器
        self.text_normalizer = TextNormalizer()
        
        # CER阈值
        self.cer_threshold = config.get('cer_threshold', 0.05)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'filtered_files': 0,
            'passed_files': 0,
            'cer_values': []
        }
    
    def process_single_audio(self, audio_path: str, groundtruth_text: str, 
                           voiceprint_id: str, prompt_id: str) -> Dict:
        """处理单个音频文件"""
        result = {
            'audio_path': audio_path,
            'voiceprint_id': voiceprint_id,
            'prompt_id': prompt_id,
            'groundtruth_text': groundtruth_text,
            'transcription': '',
            'normalized_groundtruth': '',
            'normalized_transcription': '',
            'cer': 1.0,
            'passed': False,
            'success': False,
            'error_message': ''
        }
        
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                result['error_message'] = f"音频文件不存在: {audio_path}"
                return result
            
            # 输出处理开始信息
            logger.info(f"GPU {self.gpu_id}: 处理音频: {os.path.basename(audio_path)}")
            logger.info(f"  Prompt ID: {prompt_id}, Voiceprint ID: {voiceprint_id}")
            
            # 语音识别
            transcription = self.kimi_processor.transcribe_audio(audio_path)
            result['transcription'] = transcription
            
            if not transcription:
                result['error_message'] = "ASR识别失败，返回空文本"
                logger.warning(f"  ASR识别失败")
                return result
            
            # 输出原始文本对比
            logger.info(f"  原始Groundtruth: {groundtruth_text}")
            logger.info(f"  原始ASR识别结果: {transcription}")
            
            # 文本标准化
            normalized_groundtruth = self.text_normalizer.normalize_text(groundtruth_text, language=self.language)
            normalized_transcription = self.text_normalizer.normalize_text(transcription, language=self.language)
            
            result['normalized_groundtruth'] = normalized_groundtruth
            result['normalized_transcription'] = normalized_transcription
            
            # 输出标准化后的文本对比
            logger.info(f"  TN后Groundtruth: {normalized_groundtruth}")
            logger.info(f"  TN后ASR识别结果: {normalized_transcription}")
            
            # 计算CER
            if normalized_groundtruth and normalized_transcription:
                cer_score = cer(normalized_groundtruth, normalized_transcription)
                result['cer'] = cer_score
                result['passed'] = cer_score <= self.cer_threshold
                result['success'] = True
                
                # 更新统计
                self.stats['cer_values'].append(cer_score)
                
                # 输出CER结果
                status = "✓ 通过" if result['passed'] else "✗ 筛除"
                logger.info(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%) - {status}")
                logger.info("-" * 80)
            else:
                result['error_message'] = "标准化后的文本为空"
                logger.warning(f"  标准化后的文本为空")
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"GPU {self.gpu_id}: 处理音频失败 {audio_path}: {e}")
        
        return result
    
    def process_prompt_directory(self, prompt_dir: str, prompt_id: str, 
                               voiceprint_texts: List[Tuple[str, str]]) -> List[Dict]:
        """处理一个prompt目录下的所有音频"""
        results = []
        
        logger.info(f"\nGPU {self.gpu_id}: 开始处理prompt目录")
        logger.info(f"  Prompt ID: {prompt_id}")
        logger.info(f"  目录路径: {prompt_dir}")
        logger.info(f"  音频文件数: {len(voiceprint_texts)}")
        logger.info("=" * 80)
        
        for idx, (voiceprint_id, groundtruth_text) in enumerate(voiceprint_texts, 1):
            # 输出进度
            logger.info(f"\n[{idx}/{len(voiceprint_texts)}] 处理进度")
            
            # 构建音频文件路径
            audio_filename = f"{voiceprint_id}.wav"
            audio_path = os.path.join(prompt_dir, audio_filename)
            
            # 处理音频
            result = self.process_single_audio(
                audio_path, groundtruth_text, voiceprint_id, prompt_id
            )
            results.append(result)
            
            # 更新统计
            self.stats['total_files'] += 1
            if result['success']:
                self.stats['processed_files'] += 1
                if result['passed']:
                    self.stats['passed_files'] += 1
                else:
                    self.stats['filtered_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        # 输出当前prompt的统计
        logger.info(f"\nPrompt {prompt_id} 处理完成")
        logger.info(f"  总处理: {len(results)}, 通过: {sum(1 for r in results if r.get('passed'))}, " 
                   f"筛除: {sum(1 for r in results if r.get('success') and not r.get('passed'))}")
        logger.info("=" * 80)
        
        return results
    
    def process_subset(self, subset_data: List[Tuple[str, str, List[Tuple[str, str]]]], 
                      subset_id: int) -> List[Dict]:
        """处理数据子集"""
        logger.info(f"GPU {self.gpu_id}: 开始处理子集 {subset_id}，共 {len(subset_data)} 个prompt")
        
        all_results = []
        
        for prompt_dir, prompt_id, voiceprint_texts in tqdm(subset_data, 
                                                           desc=f"GPU {self.gpu_id}: 子集{subset_id}"):
            results = self.process_prompt_directory(prompt_dir, prompt_id, voiceprint_texts)
            all_results.extend(results)
        
        return all_results

def load_json_data(json_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """加载JSON数据并解析"""
    logger.info(f"加载JSON文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解析数据格式
    parsed_data = {}
    for prompt_id, entries in data.items():
        voiceprint_texts = []
        for entry in entries:
            # 解析格式: "voiceprint_ID\t文本内容"
            parts = entry.split('\t', 1)
            if len(parts) == 2:
                voiceprint_id = parts[0]
                text = parts[1]
                voiceprint_texts.append((voiceprint_id, text))
        
        parsed_data[prompt_id] = voiceprint_texts
    
    logger.info(f"加载了 {len(parsed_data)} 个prompt的数据")
    return parsed_data

def prepare_data_for_processing(base_dir: str, json_data: Dict) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """准备处理数据"""
    zero_shot_dir = os.path.join(base_dir, "zero_shot")
    
    if not os.path.exists(zero_shot_dir):
        logger.error(f"zero_shot目录不存在: {zero_shot_dir}")
        return []
    
    data_list = []
    
    # 遍历所有prompt
    for prompt_id, voiceprint_texts in json_data.items():
        prompt_dir = os.path.join(zero_shot_dir, prompt_id)
        
        if os.path.exists(prompt_dir) and os.path.isdir(prompt_dir):
            data_list.append((prompt_dir, prompt_id, voiceprint_texts))
        else:
            logger.warning(f"Prompt目录不存在: {prompt_dir}")
    
    return data_list

def split_data(data_list: List, num_splits: int) -> List[List]:
    """将数据分割成多个子集"""
    total_items = len(data_list)
    items_per_split = total_items // num_splits
    remainder = total_items % num_splits
    
    splits = []
    start_idx = 0
    
    for i in range(num_splits):
        current_split_size = items_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size
        splits.append(data_list[start_idx:end_idx])
        start_idx = end_idx
    
    return splits

def process_gpu_subset(args_tuple):
    """处理单个GPU子集的函数"""
    gpu_id, subset_data, config, subset_id = args_tuple
    
    try:
        # 设置进程的GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 确保在子进程中重新初始化CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        # 创建处理器
        processor = TTSFilterProcessor(config, gpu_id)
        
        # 处理子集
        results = processor.process_subset(subset_data, subset_id)
        
        return results, processor.stats
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 处理子集失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def merge_and_save_results(results_list: List[List[Dict]], stats_list: List[Dict], 
                          output_path: str, base_dir: str, json_path: str):
    """合并结果并保存"""
    # 合并所有结果
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    # 合并统计信息
    merged_stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'filtered_files': 0,
        'passed_files': 0,
        'cer_values': []
    }
    
    for stats in stats_list:
        for key in ['total_files', 'processed_files', 'failed_files', 
                   'filtered_files', 'passed_files']:
            merged_stats[key] += stats.get(key, 0)
        merged_stats['cer_values'].extend(stats.get('cer_values', []))
    
    # 计算CER统计
    if merged_stats['cer_values']:
        merged_stats['cer_stats'] = {
            'mean': float(np.mean(merged_stats['cer_values'])),
            'median': float(np.median(merged_stats['cer_values'])),
            'std': float(np.std(merged_stats['cer_values'])),
            'min': float(np.min(merged_stats['cer_values'])),
            'max': float(np.max(merged_stats['cer_values']))
        }
    
    # 保存结果
    result_data = {
        'base_dir': base_dir,
        'json_path': json_path,
        'timestamp': datetime.now().isoformat(),
        'statistics': merged_stats,
        'filter_results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_path}")
    
    # 打印统计摘要
    print_summary(merged_stats)
    
    # 生成筛选文件列表
    filtered_files = [r for r in all_results if r['success'] and not r['passed']]
    if filtered_files:
        filtered_list_path = output_path.replace('.json', '_filtered_list.txt')
        with open(filtered_list_path, 'w', encoding='utf-8') as f:
            for r in filtered_files:
                f.write(f"{r['audio_path']}\n")
        logger.info(f"筛选文件列表保存到: {filtered_list_path}")
    
    return all_results

def print_summary(stats: Dict):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("TTS音频筛选结果统计")
    print("=" * 80)
    print(f"总音频文件数:     {stats['total_files']}")
    print(f"成功处理:         {stats['processed_files']}")
    print(f"处理失败:         {stats['failed_files']}")
    print(f"通过筛选:         {stats['passed_files']} ({stats['passed_files']/max(1,stats['processed_files'])*100:.1f}%)")
    print(f"被筛选掉:         {stats['filtered_files']} ({stats['filtered_files']/max(1,stats['processed_files'])*100:.1f}%)")
    
    if 'cer_stats' in stats:
        print(f"\nCER统计:")
        print(f"  平均值:         {stats['cer_stats']['mean']:.4f}")
        print(f"  中位数:         {stats['cer_stats']['median']:.4f}")
        print(f"  标准差:         {stats['cer_stats']['std']:.4f}")
        print(f"  最小值:         {stats['cer_stats']['min']:.4f}")
        print(f"  最大值:         {stats['cer_stats']['max']:.4f}")
    
    print("=" * 80)

def main():
    """主函数"""
    # 设置multiprocessing启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="基于Kimi-Audio ASR的TTS音频筛选")
    parser.add_argument("base_dir", type=str, help="音频文件基础目录")
    parser.add_argument("json_file", type=str, help="包含groundtruth的JSON文件")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--cer_threshold", type=float, default=0.05, 
                       help="CER阈值，超过此值的音频将被筛选掉 (默认: 0.05)")
    parser.add_argument("--num_gpus", type=int, help="使用的GPU数量")
    parser.add_argument("--kimi_model_path", type=str, 
                       default="/root/data/pretrained_models/Kimi-Audio-7B-Instruct",
                       help="Kimi-Audio模型路径")
    parser.add_argument("--kimi_audio_dir", type=str,
                       default="/root/code/github_repos/Kimi-Audio",
                       help="Kimi-Audio代码目录")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，只处理前10个prompt")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细的处理日志，包括TN过程")
    parser.add_argument("--language", type=str, choices=['auto', 'zh', 'en'], 
                       default='auto',
                       help="文本语言：auto(自动检测), zh(中文), en(英文) (默认: auto)")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("启用详细日志模式")
    
    # 检测可用GPU数
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        logger.error("至少需要1张GPU")
        return
    
    num_gpus = args.num_gpus if args.num_gpus else available_gpus
    num_gpus = min(num_gpus, available_gpus)
    
    logger.info(f"使用 {num_gpus} 张GPU进行处理")
    
    # 准备配置
    config = {
        'kimi_model_path': args.kimi_model_path,
        'kimi_audio_dir': args.kimi_audio_dir,
        'cer_threshold': args.cer_threshold,
        'verbose': args.verbose,
        'language': args.language
    }
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        base_name = Path(args.json_file).stem
        output_path = f"tts_filter_results_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print("基于Kimi-Audio ASR的TTS音频筛选")
    print("=" * 80)
    print(f"基础目录: {args.base_dir}")
    print(f"JSON文件: {args.json_file}")
    print(f"CER阈值: {args.cer_threshold}")
    print(f"使用GPU数: {num_gpus}")
    print(f"语言设置: {args.language} {'(自动检测)' if args.language == 'auto' else '(中文)' if args.language == 'zh' else '(英文)'}")
    print(f"输出文件: {output_path}")
    print("=" * 80)
    
    try:
        # 加载JSON数据
        json_data = load_json_data(args.json_file)
        
        # 准备处理数据
        data_list = prepare_data_for_processing(args.base_dir, json_data)
        
        if not data_list:
            logger.error("没有找到可处理的数据")
            return
        
        # 测试模式：只处理前10个prompt
        if args.test_mode:
            original_count = len(data_list)
            data_list = data_list[:10]
            logger.info(f"测试模式：从 {original_count} 个prompt中只处理前 {len(data_list)} 个")
        
        logger.info(f"找到 {len(data_list)} 个prompt目录需要处理")
        
        # 分割数据给多个GPU
        data_splits = split_data(data_list, num_gpus)
        
        # 准备多进程参数
        process_args = []
        for i in range(num_gpus):
            if i < len(data_splits) and data_splits[i]:
                process_args.append((i, data_splits[i], config, i))
        
        # 启动多进程处理
        logger.info("启动多GPU并行处理...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for args_tuple in process_args:
                future = executor.submit(process_gpu_subset, args_tuple)
                futures.append(future)
            
            # 等待所有进程完成
            results_list = []
            stats_list = []
            for future in futures:
                try:
                    results, stats = future.result()
                    results_list.append(results)
                    stats_list.append(stats)
                except Exception as e:
                    logger.error(f"进程执行失败: {e}")
                    results_list.append([])
                    stats_list.append({})
        
        processing_time = time.time() - start_time
        logger.info(f"处理完成，耗时: {processing_time:.2f} 秒")
        
        # 合并结果并保存
        merge_and_save_results(results_list, stats_list, output_path, 
                              args.base_dir, args.json_file)
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()