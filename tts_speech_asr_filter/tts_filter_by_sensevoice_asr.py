#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SenseVoice Small ASR和NeMo文本标准化的TTS音频筛选脚本

根据语音识别结果与groundtruth文本的CER差异来筛选TTS合成音频
CER > 阈值的音频将被标记为需要过滤

文本标准化：
- 英文：使用 NeMo 文本标准化（nemo_text_processing）
- 中文：使用简单标准化（去除标点、空格）
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import soundfile as sf
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from tqdm import tqdm
import torch
from jiwer import cer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import shutil

# 设置日志（需要在导入检查之前，以便可以使用 logger）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 注意：已移除 WeTextProcessing 依赖，现在使用 NeMo 文本标准化

# 导入NeMo的文本标准化模块（用于英文TN）
NEMO_AVAILABLE = False
NeMoNormalizer = None

try:
    # 尝试导入 nemo_text_processing（独立包）
    from nemo_text_processing.text_normalization.normalize import Normalizer as NeMoNormalizer
    NEMO_AVAILABLE = True
    logger.info("✓ 成功导入 NeMo 文本标准化模块 (nemo_text_processing)")
except ImportError:
    try:
        # 尝试从 nemo 主包导入
        from nemo.collections.nlp.models.text_normalization import TextNormalizationModel
        NeMoNormalizer = TextNormalizationModel
        NEMO_AVAILABLE = True
        logger.info("✓ 成功导入 NeMo 文本标准化模块 (nemo)")
    except ImportError:
        try:
            # 尝试另一种导入方式
            from nemo.collections.nlp.data.text_normalization import Normalizer as NeMoNormalizer
            NEMO_AVAILABLE = True
            logger.info("✓ 成功导入 NeMo 文本标准化模块 (nemo.data)")
        except ImportError:
            logger.info("提示: NeMo文本标准化未安装，英文将使用简单标准化")
            logger.info("  安装命令: pip install nemo_text_processing")
            NeMoNormalizer = None
            NEMO_AVAILABLE = False

# 导入FunASR的AutoModel
try:
    from funasr import AutoModel as FunasrAutoModel
except ImportError:
    logger.error("无法导入funasr，请安装: pip install funasr")
    FunasrAutoModel = None

class SenseVoiceProcessor:
    """SenseVoice Small语音识别处理器"""
    
    def __init__(self, model_dir: str = "iic/SenseVoiceSmall", device: str = "cuda:0", 
                 gpu_id: int = 0, language: str = "auto"):
        self.model_dir = model_dir
        self.device = device
        self.gpu_id = gpu_id
        self.language = language
        self.model = None
        
        logger.info(f"GPU {gpu_id}: 初始化SenseVoice处理器")
        logger.info(f"  模型: {model_dir}")
        logger.info(f"  设备: {device}")
        logger.info(f"  语言: {language}")
    
    def load_model(self):
        """加载SenseVoice模型"""
        if self.model is not None:
            return
        
        if FunasrAutoModel is None:
            raise RuntimeError("funasr库未安装，无法加载SenseVoice模型")
        
        try:
            logger.info(f"GPU {self.gpu_id}: 加载SenseVoice模型 {self.model_dir}...")
            
            # 加载模型到指定设备
            self.model = FunasrAutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                device=self.device
            )
            
            logger.info(f"GPU {self.gpu_id}: ✓ 成功加载SenseVoice模型")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 加载SenseVoice模型失败: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """使用SenseVoice进行语音识别"""
        try:
            # 确保模型已经加载
            if self.model is None:
                self.load_model()
            
            # 根据语言设置决定参数
            language_param = None
            if self.language == "zh":
                language_param = "zh"
            elif self.language == "en":
                language_param = "en"
            # auto 模式使用 None，让模型自动检测
            
            logger.debug(f"GPU {self.gpu_id}: 使用SenseVoice进行语音识别 (语言: {self.language})")
            
            # 调用模型进行识别
            result = self.model.generate(
                input=audio_path,
                cache={},
                language=language_param if language_param else "auto",
                use_itn=True,  # 不使用逆文本标准化，因为我们后面会用NeMo TN处理
                batch_size=1,
                disable_pbar=True
            )
            
            # 处理返回结果
            # SenseVoice返回格式可能是列表或字典
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], dict):
                        text = result[0].get("text", "").strip()
                    else:
                        text = str(result[0]).strip()
                else:
                    text = ""
            elif isinstance(result, dict):
                text = result.get("text", "").strip()
            else:
                text = str(result).strip()
            
            # 处理特殊标记
            if text and "|nospeech|" in text:
                logger.warning(f"GPU {self.gpu_id}: SenseVoice检测到无语音内容")
                return ""
            
            if not text:
                logger.warning(f"GPU {self.gpu_id}: SenseVoice未能识别出文本")
                return ""
            
            # 清理 SenseVoice 输出的标签（格式：<|tag|>）
            # 例如：<|en|><|EMO_UNKNOWN|><|BGM|><|woitn|>text
            original_text = text
            # 匹配 <|...|> 格式的标签并移除
            text = re.sub(r'<\|[^|]+\|>', '', text).strip()
            
            # 如果清理后文本为空，但原始文本不为空，记录警告
            if not text and original_text:
                logger.warning(f"GPU {self.gpu_id}: 清理标签后文本为空，使用原始文本")
                text = original_text.strip()
            
            logger.debug(f"GPU {self.gpu_id}: SenseVoice识别结果（原始）: {original_text}")
            logger.debug(f"GPU {self.gpu_id}: SenseVoice识别结果（清理后）: {text}")
            return text
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 语音识别失败 {audio_path}: {e}")
            return ""

class TextNormalizer:
    """文本标准化器 - 英文使用NeMo TN，中文使用简单标准化"""
    
    def __init__(self):
        self.chinese_normalizer = None
        self.english_normalizer = None
        self.nemo_normalizer = None
        
        # 尝试加载 NeMo 英文文本标准化器
        if NEMO_AVAILABLE and NeMoNormalizer:
            try:
                # 根据不同的 NeMo 版本，初始化方式可能不同
                try:
                    # nemo_text_processing 包的初始化方式
                    self.nemo_normalizer = NeMoNormalizer(input_case='cased', lang='en')
                    logger.info("✓ 成功加载 NeMo 英文文本标准化器 (nemo_text_processing)")
                except (TypeError, AttributeError):
                    try:
                        # 尝试其他初始化方式
                        self.nemo_normalizer = NeMoNormalizer(lang='en')
                        logger.info("✓ 成功加载 NeMo 英文文本标准化器")
                    except Exception:
                        # 如果是从 nemo 包导入的模型类，可能需要 from_pretrained
                        try:
                            self.nemo_normalizer = NeMoNormalizer.from_pretrained(model_name="nemo_text_normalization_en")
                            logger.info("✓ 成功加载 NeMo 英文文本标准化模型")
                        except Exception:
                            raise
            except Exception as e:
                logger.warning(f"加载 NeMo 标准化器失败: {e}，将使用简单标准化")
                self.nemo_normalizer = None
        else:
            logger.info("NeMo 文本标准化不可用，英文将使用简单标准化")
        
        # 注意：中文使用简单标准化，不再使用 WeTextProcessing
    
    def normalize_text(self, text: str, language: str = 'auto') -> str:
        """标准化文本（TN - Text Normalization）
        
        英文：使用 NeMo 文本标准化（如果可用），否则使用简单标准化
        中文：使用简单标准化（去除标点、空格）
        """
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
            if language == 'en' and self.nemo_normalizer:
                # 使用 NeMo 英文文本标准化
                try:
                    normalized_result = None
                    
                    # 尝试不同的 NeMo API 调用方式
                    if hasattr(self.nemo_normalizer, 'normalize'):
                        try:
                            # 方式1: normalize(text, verbose=False)
                            normalized_result = self.nemo_normalizer.normalize(text, verbose=False)
                        except TypeError:
                            try:
                                # 方式2: normalize(text)
                                normalized_result = self.nemo_normalizer.normalize(text)
                            except Exception:
                                # 方式3: 可能是模型类，需要特殊调用
                                if hasattr(self.nemo_normalizer, 'forward'):
                                    normalized_result = self.nemo_normalizer.forward(text)
                    
                    # 处理返回结果
                    if normalized_result is not None:
                        if isinstance(normalized_result, list):
                            normalized = ' '.join(str(x) for x in normalized_result) if normalized_result else text
                        elif isinstance(normalized_result, str):
                            normalized = normalized_result
                        else:
                            normalized = str(normalized_result)
                        
                        normalized = normalized.strip()
                        
                        if normalized != text:
                            logger.debug(f"    NeMo TN: '{text}' -> '{normalized}'")
                        
                        # 进一步简单标准化（去除标点、转小写）以确保格式统一
                        normalized = self._simple_normalize(normalized, language)
                        
                        return normalized
                    else:
                        raise ValueError("NeMo normalize 返回 None")
                        
                except Exception as e:
                    logger.warning(f"NeMo 标准化失败: {e}，回退到简单标准化")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return self._simple_normalize(text, language)
            else:
                # 中文或其他情况：使用简单标准化
                normalized = self._simple_normalize(text, language)
                if normalized != text:
                    logger.debug(f"    简单TN: '{text}' -> '{normalized}'")
                return normalized
        except Exception as e:
            logger.warning(f"文本标准化失败: {e}，使用简单标准化")
            return self._simple_normalize(text, language)
    
    def _simple_normalize(self, text: str, language: str) -> str:
        """简单的文本标准化
        
        对于英文：转小写，去除标点符号，标准化空格（保留单个空格）
        对于中文：去除标点符号和空格
        """
        import re
        
        if language == 'zh':
            # 中文：去除标点符号和空格
            text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        else:
            # 英文：转小写，去除标点符号，标准化空格（将多个空格合并为一个）
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
            text = re.sub(r'\s+', ' ', text)  # 将多个空格合并为一个空格
        
        return text.strip()

class TTSFilterProcessor:
    """TTS音频筛选处理器"""
    
    def __init__(self, config: Dict, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        # 获取实际的GPU ID（用于日志显示）
        self.actual_gpu_id = config.get('actual_gpu_id', gpu_id)
        # 在多进程环境中，device始终是cuda:0或cuda:gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # 语言设置
        self.language = config.get('language', 'auto')
        logger.info(f"GPU {self.actual_gpu_id}: 语言设置为 '{self.language}' (设备: {self.device})")
        
        # 初始化SenseVoice处理器
        self.sensevoice_processor = SenseVoiceProcessor(
            model_dir=config.get('sensevoice_model_dir', 'iic/SenseVoiceSmall'),
            device=self.device,
            gpu_id=self.actual_gpu_id,
            language=self.language
        )
        
        # 初始化NeMo文本标准化器
        self.text_normalizer = TextNormalizer()
        
        # CER阈值
        self.cer_threshold = config.get('cer_threshold', 0.05)
        
        # 已处理的音频路径集合（用于增量处理）
        self.processed_audio_paths = config.get('processed_audio_paths', set())
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'filtered_files': 0,
            'passed_files': 0,
            'skipped_files': 0,
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
            'error_message': '',
            'language': self.language
        }
        
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                result['error_message'] = f"音频文件不存在: {audio_path}"
                return result
            
            # 输出处理开始信息
            logger.info(f"GPU {self.actual_gpu_id}: 处理音频: {os.path.basename(audio_path)}")
            logger.info(f"  Prompt ID: {prompt_id}, Voiceprint ID: {voiceprint_id}")
            logger.info(f"  语言模式: {self.language}")
            
            # 语音识别
            logger.info(f"  使用SenseVoice进行语音识别 (语言: {self.language})...")
            transcription = self.sensevoice_processor.transcribe_audio(audio_path)
            result['transcription'] = transcription
            
            if not transcription:
                result['error_message'] = "ASR识别失败，返回空文本"
                logger.warning(f"  ASR识别失败")
                return result
            
            # 输出原始文本对比
            logger.info(f"  原始Groundtruth: {groundtruth_text}")
            logger.info(f"  原始ASR识别结果: {transcription}")
            
            # 文本标准化 - 使用 NeMo（英文）或简单标准化（中文）
            logger.info(f"  使用文本标准化 (语言: {self.language})...")
            normalized_groundtruth = self.text_normalizer.normalize_text(groundtruth_text, language=self.language)
            normalized_transcription = self.text_normalizer.normalize_text(transcription, language=self.language)
            
            result['normalized_groundtruth'] = normalized_groundtruth
            result['normalized_transcription'] = normalized_transcription
            
            # 输出标准化后的文本对比
            logger.info(f"  标准化后Groundtruth: {normalized_groundtruth}")
            logger.info(f"  标准化后ASR识别结果: {normalized_transcription}")
            
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
            logger.error(f"GPU {self.actual_gpu_id}: 处理音频失败 {audio_path}: {e}")
        
        return result
    
    def process_sample_tasks(self, sample_tasks: List[Tuple[str, str, str, str]], subset_id: int) -> List[Dict]:
        """按样本级任务处理（每个元素代表一个音频样本）
        
        sample_tasks: List[(audio_path, voiceprint_id, groundtruth_text, prompt_id)]
        """
        logger.info(f"GPU {self.actual_gpu_id}: 开始处理样本子集 {subset_id}，共 {len(sample_tasks)} 个样本 (语言: {self.language})")
        
        all_results: List[Dict] = []
        skipped_in_subset = 0
        
        for idx, (audio_path, voiceprint_id, groundtruth_text, prompt_id) in enumerate(
            tqdm(sample_tasks, desc=f"GPU {self.actual_gpu_id}: 样本子集{subset_id}"), start=1
        ):
            if self.processed_audio_paths and audio_path in self.processed_audio_paths:
                logger.info(f"\n[{idx}/{len(sample_tasks)}] 跳过已处理的音频: {os.path.basename(audio_path)}")
                self.stats['skipped_files'] += 1
                skipped_in_subset += 1
                continue
            
            logger.info(f"\n[{idx}/{len(sample_tasks)}] 处理进度 (语言: {self.language})")
            result = self.process_single_audio(audio_path, groundtruth_text, voiceprint_id, prompt_id)
            all_results.append(result)
            
            self.stats['total_files'] += 1
            if result['success']:
                self.stats['processed_files'] += 1
                if result['passed']:
                    self.stats['passed_files'] += 1
                else:
                    self.stats['filtered_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        logger.info(f"\n样本子集 {subset_id} 处理完成 (语言: {self.language})")
        logger.info(f"  新处理: {len(all_results)}, 跳过: {skipped_in_subset}, "
                    f"通过: {sum(1 for r in all_results if r.get('passed'))}, "
                    f"筛除: {sum(1 for r in all_results if r.get('success') and not r.get('passed'))}")
        logger.info("=" * 80)
        
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

def load_existing_results(output_path: str) -> Tuple[Dict[str, Dict], Dict, Set[str]]:
    """加载已存在的结果文件
    
    返回:
        - existing_results: 已有的所有结果，以音频路径为键
        - existing_stats: 已有的统计信息
        - processed_audio_paths: 已处理的音频路径集合
    """
    if not os.path.exists(output_path):
        return {}, {}, set()
    
    try:
        logger.info(f"加载已有结果文件: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取已有结果
        existing_results = {}
        processed_audio_paths = set()
        
        if 'filter_results' in data:
            for result in data['filter_results']:
                audio_path = result.get('audio_path')
                if audio_path:
                    existing_results[audio_path] = result
                    processed_audio_paths.add(audio_path)
        
        # 提取已有统计
        existing_stats = data.get('statistics', {})
        
        logger.info(f"已加载 {len(processed_audio_paths)} 个已处理的音频结果")
        
        return existing_results, existing_stats, processed_audio_paths
        
    except Exception as e:
        logger.error(f"加载已有结果文件失败: {e}")
        return {}, {}, set()

def build_sample_task_list(base_dir: str, json_data: Dict[str, List[Tuple[str, str]]], 
                          additional_base_dirs: List[str] = None) -> List[Tuple[str, str, str, str]]:
    """将JSON展开为样本级任务列表
    
    参数:
        base_dir: 主要的基础目录
        json_data: JSON数据
        additional_base_dirs: 额外的base_dir列表，用于在多个目录中搜索音频文件
    
    返回的每个任务为 (audio_path, voiceprint_id, groundtruth_text, prompt_id)
    """
    # 收集所有要搜索的base_dir
    base_dirs_to_search = [base_dir]
    if additional_base_dirs:
        base_dirs_to_search.extend(additional_base_dirs)
    
    sample_tasks: List[Tuple[str, str, str, str]] = []
    
    for prompt_id, voiceprint_texts in json_data.items():
        for voiceprint_id, text in voiceprint_texts:
            audio_filename = f"{voiceprint_id}.wav"
            audio_path = None
            
            # 在所有base_dir中搜索音频文件
            for bd in base_dirs_to_search:
                zero_shot_dir = os.path.join(bd, "zero_shot")
                if not os.path.exists(zero_shot_dir):
                    continue
                
                prompt_dir = os.path.join(zero_shot_dir, prompt_id)
                if os.path.exists(prompt_dir) and os.path.isdir(prompt_dir):
                    candidate_path = os.path.join(prompt_dir, audio_filename)
                    if os.path.exists(candidate_path):
                        audio_path = candidate_path
                        break
            
            if audio_path is None:
                logger.warning(f"未找到音频文件: prompt_id={prompt_id}, voiceprint_id={voiceprint_id}")
                continue
            
            sample_tasks.append((audio_path, voiceprint_id, text, prompt_id))
    
    return sample_tasks

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
    gpu_id, subset_data, config, subset_id, processed_audio_paths = args_tuple
    
    try:
        # 设置进程的GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 确保在子进程中重新初始化CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        # 创建处理器 - 在子进程中始终使用cuda:0
        # 因为CUDA_VISIBLE_DEVICES已经限制了可见的GPU
        modified_config = config.copy()
        modified_config['actual_gpu_id'] = gpu_id  # 保存原始GPU ID用于日志
        modified_config['processed_audio_paths'] = processed_audio_paths  # 传递已处理的音频路径
        processor = TTSFilterProcessor(modified_config, gpu_id=0)  # 使用cuda:0
        
        # 处理样本子集
        results = processor.process_sample_tasks(subset_data, subset_id)
        
        return results, processor.stats
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 处理子集失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def merge_and_save_results(results_list: List[List[Dict]], stats_list: List[Dict], 
                          output_path: str, base_dir: str, json_path: str, 
                          existing_results: Dict = None, existing_stats: Dict = None):
    """合并结果并保存（支持增量更新）"""
    # 合并所有新结果
    new_results = []
    for results in results_list:
        new_results.extend(results)
    
    # 合并新的统计信息
    merged_stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'filtered_files': 0,
        'passed_files': 0,
        'skipped_files': 0,
        'cer_values': []
    }
    
    for stats in stats_list:
        for key in ['total_files', 'processed_files', 'failed_files', 
                   'filtered_files', 'passed_files', 'skipped_files']:
            merged_stats[key] += stats.get(key, 0)
        merged_stats['cer_values'].extend(stats.get('cer_values', []))
    
    # 如果有已存在的结果，进行合并
    if existing_results:
        # 合并结果
        all_results_dict = existing_results.copy()
        for result in new_results:
            audio_path = result.get('audio_path')
            if audio_path:
                all_results_dict[audio_path] = result
        
        # 转换回列表
        all_results = list(all_results_dict.values())
        
        # 合并统计信息
        if existing_stats:
            # 重新计算总数
            merged_stats['total_files'] += existing_stats.get('total_files', 0)
            merged_stats['processed_files'] += existing_stats.get('processed_files', 0)
            merged_stats['failed_files'] += existing_stats.get('failed_files', 0)
            merged_stats['filtered_files'] += existing_stats.get('filtered_files', 0)
            merged_stats['passed_files'] += existing_stats.get('passed_files', 0)
            
            # 合并CER值
            existing_cer_values = existing_stats.get('cer_values', [])
            merged_stats['cer_values'] = existing_cer_values + merged_stats['cer_values']
        
        logger.info(f"增量处理完成: 新处理 {len(new_results)} 个文件，累计 {len(all_results)} 个文件")
    else:
        all_results = new_results
        logger.info(f"首次处理: 共处理 {len(all_results)} 个文件")
    
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
    print("TTS音频筛选结果统计 (SenseVoice + NeMo TN)")
    print("=" * 80)
    print(f"总音频文件数:     {stats.get('total_files', 0)}")
    print(f"成功处理:         {stats.get('processed_files', 0)}")
    print(f"处理失败:         {stats.get('failed_files', 0)}")
    if stats.get('skipped_files', 0) > 0:
        print(f"跳过已处理:       {stats['skipped_files']}")
    print(f"通过筛选:         {stats.get('passed_files', 0)} ({stats.get('passed_files', 0)/max(1,stats.get('processed_files', 1))*100:.1f}%)")
    print(f"被筛选掉:         {stats.get('filtered_files', 0)} ({stats.get('filtered_files', 0)/max(1,stats.get('processed_files', 1))*100:.1f}%)")
    
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
    
    parser = argparse.ArgumentParser(description="基于SenseVoice Small ASR和NeMo文本标准化的TTS音频筛选")
    parser.add_argument("base_dir", type=str, help="音频文件基础目录")
    parser.add_argument("json_file", type=str, help="包含groundtruth的JSON文件")
    parser.add_argument("--additional_base_dirs", type=str, nargs="+", 
                       help="额外的base_dir列表，用于在多个目录中搜索音频文件（合并模式使用）")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--cer_threshold", type=float, default=0.05, 
                       help="CER阈值，超过此值的音频将被筛选掉 (默认: 0.05)")
    parser.add_argument("--num_gpus", type=int, help="使用的GPU数量")
    parser.add_argument("--sensevoice_model_dir", type=str, default="iic/SenseVoiceSmall",
                       help="SenseVoice模型路径或模型ID (默认: iic/SenseVoiceSmall)")
    parser.add_argument("--language", type=str, choices=['auto', 'zh', 'en'], 
                       default='auto',
                       help="文本语言：auto(自动检测), zh(中文), en(英文) (默认: auto)")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，只处理前10个prompt")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细的处理日志")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="跳过已存在的结果文件 (默认: True)")
    parser.add_argument("--no_skip_existing", action="store_true",
                       help="不跳过已存在的结果，强制重新处理")
    parser.add_argument("--force", action="store_true",
                       help="强制重新处理（等同于--no_skip_existing）")
    parser.add_argument("--debug_mode", action="store_true",
                       help="调试模式：限制样本数量并优先使用8卡（若可用）")
    parser.add_argument("--debug_samples", type=int, default=1000,
                       help="调试模式下处理的样本上限 (默认: 1000)")
    
    args = parser.parse_args()
    
    # 处理skip_existing标志
    skip_existing = args.skip_existing
    if args.no_skip_existing or args.force:
        skip_existing = False
    
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
    # 调试模式：优先使用8卡（若可用）
    if args.debug_mode:
        desired = min(8, available_gpus)
        if num_gpus != desired:
            logger.info(f"调试模式：将使用 {desired} 张GPU进行处理（原请求: {num_gpus}）")
        num_gpus = desired
    
    logger.info(f"使用 {num_gpus} 张GPU进行处理")
    
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            global_config = file_config.get('global_config', {})
    else:
        global_config = {}
    
    # 准备配置
    config = {
        'sensevoice_model_dir': args.sensevoice_model_dir,
        'cer_threshold': args.cer_threshold,
        'verbose': args.verbose,
        'language': args.language
    }
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        base_name = Path(args.json_file).stem
        output_path = f"tts_filter_results_sensevoice_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 加载已存在的结果（用于增量处理）
    existing_results = {}
    existing_stats = {}
    processed_audio_paths = set()
    
    if os.path.exists(output_path):
        if skip_existing:
            # 增量模式：加载已有结果，只处理新的音频
            logger.info("=" * 80)
            logger.info("检测到结果文件已存在，进入增量处理模式")
            logger.info(f"文件路径: {output_path}")
            
            existing_results, existing_stats, processed_audio_paths = load_existing_results(output_path)
            
            if processed_audio_paths:
                logger.info(f"已有 {len(processed_audio_paths)} 个音频的处理结果")
                logger.info("将跳过这些已处理的音频，只处理新的音频文件")
            
            logger.info("=" * 80)
        else:
            # 强制重新处理模式
            logger.info("=" * 80)
            logger.info("检测到结果文件已存在，但使用强制重新处理模式")
            logger.info(f"将覆盖文件: {output_path}")
            logger.info("=" * 80)
    
    print("基于SenseVoice Small ASR和NeMo文本标准化的TTS音频筛选")
    print("=" * 80)
    print(f"基础目录: {args.base_dir}")
    print(f"JSON文件: {args.json_file}")
    print(f"CER阈值: {args.cer_threshold}")
    print(f"使用GPU数: {num_gpus}")
    print(f"SenseVoice模型: {args.sensevoice_model_dir}")
    print(f"语言设置: {args.language} {'(自动检测)' if args.language == 'auto' else '(中文)' if args.language == 'zh' else '(英文)'}")
    print(f"文本标准化: NeMo (英文) / 简单标准化 (中文)")
    print(f"输出文件: {output_path}")
    print(f"处理模式: {'增量处理（跳过已处理）' if (skip_existing and processed_audio_paths) else '全新处理' if not processed_audio_paths else '强制重新处理'}")
    if processed_audio_paths:
        print(f"已处理音频数: {len(processed_audio_paths)}")
    print("=" * 80)
    
    try:
        # 加载JSON数据
        json_data = load_json_data(args.json_file)
        
        # 按样本级准备处理数据
        additional_base_dirs = getattr(args, 'additional_base_dirs', None)
        all_sample_tasks = build_sample_task_list(args.base_dir, json_data, additional_base_dirs)
        
        if not all_sample_tasks:
            logger.error("没有找到可处理的样本任务")
            return
        
        # 测试/调试模式限制样本数量
        if args.debug_mode:
            original_count = len(all_sample_tasks)
            limit = max(1, int(args.debug_samples))
            all_sample_tasks = all_sample_tasks[:limit]
            logger.info(f"调试模式：限制处理样本数为 {limit}（原始 {original_count}）")
        elif args.test_mode:
            original_count = len(all_sample_tasks)
            seen_prompts: Set[str] = set()
            limited_tasks: List[Tuple[str, str, str, str]] = []
            for task in all_sample_tasks:
                pid = task[3]
                if len(seen_prompts) < 10 or pid in seen_prompts:
                    limited_tasks.append(task)
                    seen_prompts.add(pid)
            all_sample_tasks = limited_tasks
            logger.info(f"测试模式：从 {original_count} 个样本中，只处理前10个prompt的样本，共 {len(all_sample_tasks)} 个")
        
        logger.info(f"找到 {len(all_sample_tasks)} 个样本需要处理")
        
        # 增量模式：在主进程中过滤掉已处理样本，以实现样本级均衡分发
        if processed_audio_paths:
            before = len(all_sample_tasks)
            all_sample_tasks = [t for t in all_sample_tasks if t[0] not in processed_audio_paths]
            after = len(all_sample_tasks)
            if before != after:
                logger.info(f"增量模式：过滤掉已处理样本 {before - after} 个，剩余 {after} 个待处理样本")
        
        if not all_sample_tasks:
            logger.info("没有新的样本需要处理，直接合并并保存现有统计")
            merge_and_save_results([], [], output_path, args.base_dir, args.json_file,
                                   existing_results, existing_stats)
            return
        
        # 分割样本任务给多个GPU
        data_splits = split_data(all_sample_tasks, num_gpus)
        
        # 准备多进程参数
        process_args = []
        for i in range(num_gpus):
            if i < len(data_splits) and data_splits[i]:
                process_args.append((i, data_splits[i], config, i, processed_audio_paths))
        
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
        
        # 合并结果并保存（支持增量更新）
        merge_and_save_results(results_list, stats_list, output_path, 
                              args.base_dir, args.json_file,
                              existing_results, existing_stats)
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

