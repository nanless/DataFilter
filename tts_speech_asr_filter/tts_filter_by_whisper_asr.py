#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Whisper ASR和LLM文本标准化的TTS音频筛选脚本

根据语音识别结果与groundtruth文本的CER差异来筛选TTS合成音频
CER > 阈值的音频将被标记为需要过滤
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import soundfile as sf
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
import whisper
import requests

# 设置代理绕过
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperProcessor:
    """Whisper语音识别处理器"""
    
    def __init__(self, model_size: str = "large-v3", device: str = "cuda:0", gpu_id: int = 0, 
                 language: str = "auto", model_dir: str = None):
        self.model_size = model_size
        self.device = device
        self.gpu_id = gpu_id
        self.language = language
        self.model = None
        self.model_dir = model_dir or "/root/data/pretrained_models/whisper_modes"
        
        logger.info(f"GPU {gpu_id}: 初始化Whisper处理器")
        logger.info(f"  模型: whisper-{model_size}")
        logger.info(f"  模型目录: {self.model_dir}")
        logger.info(f"  设备: {device}")
        logger.info(f"  语言: {language}")
    
    def load_model(self):
        """加载Whisper模型"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"GPU {self.gpu_id}: 加载Whisper模型 {self.model_size}...")
            logger.info(f"  从目录: {self.model_dir}")
            
            # 加载模型到指定设备，使用本地模型目录
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device,
                download_root=self.model_dir
            )
            
            logger.info(f"GPU {self.gpu_id}: ✓ 成功加载Whisper模型")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 加载Whisper模型失败: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """使用Whisper进行语音识别"""
        try:
            # 确保模型已经加载
            if self.model is None:
                self.load_model()
            
            # 根据语言设置决定参数
            if self.language == "auto":
                # 自动检测语言
                logger.debug(f"GPU {self.gpu_id}: 使用Whisper自动语言检测")
                result = self.model.transcribe(audio_path, task="transcribe")
                detected_language = result.get("language", "unknown")
                logger.info(f"GPU {self.gpu_id}: Whisper检测到语言: {detected_language}")
            elif self.language == "zh":
                # 指定中文
                logger.info(f"GPU {self.gpu_id}: 使用Whisper中文模式 (language='zh')")
                result = self.model.transcribe(audio_path, language="zh", task="transcribe")
            elif self.language == "en":
                # 指定英文
                logger.info(f"GPU {self.gpu_id}: 使用Whisper英文模式 (language='en')")
                result = self.model.transcribe(audio_path, language="en", task="transcribe")
            else:
                # 其他语言
                logger.info(f"GPU {self.gpu_id}: 使用Whisper {self.language}语言模式")
                result = self.model.transcribe(audio_path, language=self.language, task="transcribe")
            
            # 获取转录文本
            text = result.get("text", "").strip()
            
            if not text:
                logger.warning(f"GPU {self.gpu_id}: Whisper未能识别出文本")
                return ""
            
            logger.debug(f"GPU {self.gpu_id}: Whisper识别结果: {text}")
            return text
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 语音识别失败 {audio_path}: {e}")
            return ""

class HTTPTextNormalizer:
    """通过HTTP API调用LLM进行文本标准化"""
    
    def __init__(self, service_url: str = "http://localhost:8000", timeout: int = 60, 
                 max_retries: int = 3, gpu_id: int = 0):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.gpu_id = gpu_id
        self.session = self._create_session()
        
        logger.info(f"GPU {gpu_id}: 初始化HTTP文本标准化器")
        logger.info(f"  服务地址: {self.service_url}")
        
        # 检查服务连接
        if not self.check_service():
            raise RuntimeError(f"无法连接到LLM服务: {self.service_url}")
    
    def _create_session(self):
        """创建HTTP会话，禁用代理"""
        session = requests.Session()
        session.proxies = {
            'http': None,
            'https': None,
            'no_proxy': 'localhost,127.0.0.1,::1'
        }
        return session
    
    def check_service(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            response = self.session.get(
                f"{self.service_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"GPU {self.gpu_id}: ✓ LLM服务连接成功")
                return True
            else:
                logger.error(f"GPU {self.gpu_id}: LLM服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 连接LLM服务失败: {e}")
            return False
    
    def normalize_text_pair(self, text1: str, text2: str, language: str = "auto") -> Tuple[str, str]:
        """通过LLM服务标准化文本对"""
        for attempt in range(self.max_retries):
            try:
                # 构建请求数据
                request_data = {
                    "text1": text1,
                    "text2": text2
                }
                
                # 根据语言设置添加提示
                if language != "auto":
                    logger.info(f"GPU {self.gpu_id}: 向LLM请求{language}语言的文本标准化")
                else:
                    logger.info(f"GPU {self.gpu_id}: 向LLM请求自动检测语言的文本标准化")
                
                # 发送请求
                response = self.session.post(
                    f"{self.service_url}/normalize",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        normalized_text1 = result.get("normalized_text1", text1)
                        normalized_text2 = result.get("normalized_text2", text2)
                        
                        logger.debug(f"GPU {self.gpu_id}: LLM标准化成功")
                        logger.debug(f"  原始1: {text1}")
                        logger.debug(f"  标准化1: {normalized_text1}")
                        logger.debug(f"  原始2: {text2}")
                        logger.debug(f"  标准化2: {normalized_text2}")
                        
                        return normalized_text1, normalized_text2
                    else:
                        error_msg = result.get("error_message", "未知错误")
                        logger.error(f"GPU {self.gpu_id}: LLM标准化失败: {error_msg}")
                else:
                    logger.error(f"GPU {self.gpu_id}: LLM服务返回错误: {response.status_code}")
                    logger.error(f"响应内容: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"GPU {self.gpu_id}: LLM请求超时 (尝试 {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: LLM请求异常: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        # 如果所有重试都失败，返回原始文本
        logger.error(f"GPU {self.gpu_id}: LLM标准化失败，返回原始文本")
        return text1, text2

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
        
        # 初始化Whisper处理器
        self.whisper_processor = WhisperProcessor(
            model_size=config.get('whisper_model_size', 'large-v3'),
            device=self.device,
            gpu_id=self.actual_gpu_id,  # 用于日志显示
            language=self.language,
            model_dir=config.get('whisper_model_dir', '/root/data/pretrained_models/whisper_modes')
        )
        
        # 初始化LLM文本标准化器
        # 使用实际的GPU ID来计算端口号
        llm_service_url = config.get('llm_service_url', f'http://localhost:{8000 + self.actual_gpu_id}')
        self.text_normalizer = HTTPTextNormalizer(
            service_url=llm_service_url,
            timeout=config.get('llm_timeout', 60),
            max_retries=config.get('llm_max_retries', 3),
            gpu_id=self.actual_gpu_id  # 用于日志显示
        )
        
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
            'skipped_files': 0,  # 新增：跳过的文件数
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
            logger.info(f"  使用Whisper进行语音识别 (语言: {self.language})...")
            transcription = self.whisper_processor.transcribe_audio(audio_path)
            result['transcription'] = transcription
            
            if not transcription:
                result['error_message'] = "ASR识别失败，返回空文本"
                logger.warning(f"  ASR识别失败")
                return result
            
            # 输出原始文本对比
            logger.info(f"  原始Groundtruth: {groundtruth_text}")
            logger.info(f"  原始ASR识别结果: {transcription}")
            
            # 文本标准化 - 通过LLM服务
            logger.info(f"  使用LLM进行文本标准化 (语言: {self.language})...")
            normalized_transcription, normalized_groundtruth = self.text_normalizer.normalize_text_pair(
                transcription, groundtruth_text, language=self.language
            )
            
            result['normalized_groundtruth'] = normalized_groundtruth
            result['normalized_transcription'] = normalized_transcription
            
            # 输出标准化后的文本对比
            logger.info(f"  LLM标准化后Groundtruth: {normalized_groundtruth}")
            logger.info(f"  LLM标准化后ASR识别结果: {normalized_transcription}")
            
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
    
    def process_prompt_directory(self, prompt_dir: str, prompt_id: str, 
                               voiceprint_texts: List[Tuple[str, str]]) -> List[Dict]:
        """处理一个prompt目录下的所有音频"""
        results = []
        
        logger.info(f"\nGPU {self.actual_gpu_id}: 开始处理prompt目录")
        logger.info(f"  Prompt ID: {prompt_id}")
        logger.info(f"  目录路径: {prompt_dir}")
        logger.info(f"  音频文件数: {len(voiceprint_texts)}")
        logger.info(f"  处理语言: {self.language}")
        if self.processed_audio_paths:
            logger.info(f"  已处理音频数: {len(self.processed_audio_paths)}")
        logger.info("=" * 80)
        
        skipped_in_prompt = 0
        for idx, (voiceprint_id, groundtruth_text) in enumerate(voiceprint_texts, 1):
            # 构建音频文件路径
            audio_filename = f"{voiceprint_id}.wav"
            audio_path = os.path.join(prompt_dir, audio_filename)
            
            # 检查是否已经处理过
            if audio_path in self.processed_audio_paths:
                logger.info(f"\n[{idx}/{len(voiceprint_texts)}] 跳过已处理的音频: {audio_filename}")
                self.stats['skipped_files'] += 1
                skipped_in_prompt += 1
                continue
            
            # 输出进度
            logger.info(f"\n[{idx}/{len(voiceprint_texts)}] 处理进度 (语言: {self.language})")
            
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
        logger.info(f"\nPrompt {prompt_id} 处理完成 (语言: {self.language})")
        logger.info(f"  新处理: {len(results)}, 跳过: {skipped_in_prompt}, " 
                   f"通过: {sum(1 for r in results if r.get('passed'))}, " 
                   f"筛除: {sum(1 for r in results if r.get('success') and not r.get('passed'))}")
        logger.info("=" * 80)
        
        return results
    
    def process_subset(self, subset_data: List[Tuple[str, str, List[Tuple[str, str]]]], 
                      subset_id: int) -> List[Dict]:
        """处理数据子集"""
        logger.info(f"GPU {self.actual_gpu_id}: 开始处理子集 {subset_id}，共 {len(subset_data)} 个prompt (语言: {self.language})")
        
        all_results = []
        
        for prompt_dir, prompt_id, voiceprint_texts in tqdm(subset_data, 
                                                           desc=f"GPU {self.actual_gpu_id}: 子集{subset_id}"):
            results = self.process_prompt_directory(prompt_dir, prompt_id, voiceprint_texts)
            all_results.extend(results)
        
        return all_results

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

def build_sample_task_list(base_dir: str, json_data: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, str, str, str]]:
    """将JSON展开为样本级任务列表
    
    返回的每个任务为 (audio_path, voiceprint_id, groundtruth_text, prompt_id)
    """
    zero_shot_dir = os.path.join(base_dir, "zero_shot")
    
    if not os.path.exists(zero_shot_dir):
        logger.error(f"zero_shot目录不存在: {zero_shot_dir}")
        return []
    
    sample_tasks: List[Tuple[str, str, str, str]] = []
    
    for prompt_id, voiceprint_texts in json_data.items():
        prompt_dir = os.path.join(zero_shot_dir, prompt_id)
        if not (os.path.exists(prompt_dir) and os.path.isdir(prompt_dir)):
            logger.warning(f"Prompt目录不存在: {prompt_dir}")
            continue
        
        for voiceprint_id, text in voiceprint_texts:
            audio_filename = f"{voiceprint_id}.wav"
            audio_path = os.path.join(prompt_dir, audio_filename)
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
    print("TTS音频筛选结果统计")
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

def check_llm_services(num_gpus: int) -> bool:
    """检查LLM服务是否可用"""
    session = requests.Session()
    session.proxies = {
        'http': None,
        'https': None,
        'no_proxy': 'localhost,127.0.0.1,::1'
    }
    
    available_services = 0
    logger.info("检查LLM服务状态...")
    
    for i in range(num_gpus):
        url = f"http://localhost:{8000 + i}"
        try:
            response = session.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ LLM服务正常: {url}")
                available_services += 1
            else:
                logger.warning(f"LLM服务异常: {url}, 状态码: {response.status_code}")
        except Exception as e:
            logger.warning(f"LLM服务不可用: {url}, 错误: {e}")
    
    if available_services == 0:
        logger.error("没有可用的LLM服务，请先启动")
        logger.error("运行: cd /path/to/intergrate_enhanced_speech_bygtcer && ./auto_start_llm_services.sh")
        return False
    
    logger.info(f"发现 {available_services}/{num_gpus} 个可用的LLM服务")
    return True

def main():
    """主函数"""
    # 设置multiprocessing启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="基于Whisper ASR和LLM标准化的TTS音频筛选")
    parser.add_argument("base_dir", type=str, help="音频文件基础目录")
    parser.add_argument("json_file", type=str, help="包含groundtruth的JSON文件")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--cer_threshold", type=float, default=0.1, 
                       help="CER阈值，超过此值的音频将被筛选掉 (默认: 0.1)")
    parser.add_argument("--num_gpus", type=int, help="使用的GPU数量")
    parser.add_argument("--whisper_model_size", type=str, default="large-v3",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper模型大小 (默认: large-v3)")
    parser.add_argument("--language", type=str, choices=['auto', 'zh', 'en'], 
                       default='en',
                       help="文本语言：auto(自动检测), zh(中文), en(英文) (默认: en)")
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
    
    # 检查LLM服务
    if not check_llm_services(num_gpus):
        return
    
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
        'whisper_model_size': args.whisper_model_size,
        'whisper_model_dir': global_config.get('whisper_model_dir', '/root/data/pretrained_models/whisper_modes'),
        'cer_threshold': args.cer_threshold,
        'verbose': args.verbose,
        'language': args.language,
        'llm_timeout': 60,
        'llm_max_retries': 3
    }
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        base_name = Path(args.json_file).stem
        output_path = f"tts_filter_results_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
    
    print("基于Whisper ASR和LLM标准化的TTS音频筛选")
    print("=" * 80)
    print(f"基础目录: {args.base_dir}")
    print(f"JSON文件: {args.json_file}")
    print(f"CER阈值: {args.cer_threshold}")
    print(f"使用GPU数: {num_gpus}")
    print(f"Whisper模型: {args.whisper_model_size}")
    print(f"语言设置: {args.language} {'(自动检测)' if args.language == 'auto' else '(中文)' if args.language == 'zh' else '(英文)'}")
    print(f"输出文件: {output_path}")
    print(f"处理模式: {'增量处理（跳过已处理）' if (skip_existing and processed_audio_paths) else '全新处理' if not processed_audio_paths else '强制重新处理'}")
    if processed_audio_paths:
        print(f"已处理音频数: {len(processed_audio_paths)}")
    print("=" * 80)
    
    try:
        # 加载JSON数据
        json_data = load_json_data(args.json_file)
        
        # 按样本级准备处理数据
        all_sample_tasks = build_sample_task_list(args.base_dir, json_data)
        
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
                # 为每个GPU配置独立的LLM服务URL
                gpu_config = config.copy()
                gpu_config['llm_service_url'] = f'http://localhost:{8000 + i}'
                process_args.append((i, data_splits[i], gpu_config, i, processed_audio_paths))
        
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