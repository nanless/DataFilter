#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将filtered_speech目录下的TTS克隆音频及JSON描述文件拷贝到新的目录结构中

功能：
1. 从多个数据集的utt2spk文件加载prompt_id到speaker_id的映射
2. 扫描filtered_speech目录下的音频和JSON文件 (支持按cer_range组织的结构)
3. 根据prompt_id找到对应的数据集和说话人
4. 将文件拷贝到目标目录: cer_range/dataset_name/speaker_name/prompt_id/cloned_file
"""

import os
import sys
import json
import shutil
import argparse
import logging
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量用于多进程共享
_global_prompt_to_speaker = None

def _init_worker(prompt_to_speaker):
    """初始化子进程，加载映射到全局变量"""
    global _global_prompt_to_speaker
    _global_prompt_to_speaker = prompt_to_speaker

# 数据集配置 (与原脚本保持一致)
DATASET_CONFIG = [
    {
        'name': 'childmandarin',
        'target_dir': 'childmandarin',
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/utt2spk',
    },
    {
        'name': 'chineseenglishchildren',
        'target_dir': 'chineseenglishchildren',
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/utt2spk',
    },
    {
        'name': 'king-asr',
        'target_dir': None,
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/utt2spk',
    },
    {
        'name': 'speechocean762',
        'target_dir': 'speechocean762',
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/utt2spk',
    }
]

class Utt2SpkMapper:
    def __init__(self, dataset_config: List[Dict]):
        self.dataset_config = dataset_config
        self.prompt_to_speaker = {}
        
    def load_all_mappings(self):
        logger.info("开始加载utt2spk映射...")
        for config in self.dataset_config:
            dataset_name = config['name']
            utt2spk_path = config['utt2spk_path']
            if not os.path.exists(utt2spk_path):
                logger.warning(f"utt2spk文件不存在: {utt2spk_path}")
                continue
            logger.info(f"加载 {dataset_name}: {utt2spk_path}")
            self._load_single_mapping(utt2spk_path, dataset_name)
        logger.info(f"总共加载 {len(self.prompt_to_speaker)} 个prompt_id映射")
        return len(self.prompt_to_speaker)
    
    def _load_single_mapping(self, utt2spk_path: str, dataset_name: str):
        with open(utt2spk_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    prompt_id, speaker_id = parts
                    self.prompt_to_speaker[prompt_id] = (dataset_name, speaker_id)

def scan_files(source_dirs: List[str]) -> List[Tuple[str, str, str, str]]:
    """
    扫描源目录，收集所有音频和JSON文件
    返回: [(cer_range, prompt_id, filename, file_path), ...]
    """
    files_to_copy = []
    for source_dir in source_dirs:
        logger.info(f"扫描目录: {source_dir}")
        # 结构是 source_dir/cer_range/prompt_id/file.ext
        # 使用glob扫描所有 .wav 和 .json 文件
        for ext in ['*.wav', '*.json']:
            pattern = os.path.join(source_dir, '*', '*', ext)
            for file_path in glob.iglob(pattern):
                parts = file_path.split(os.sep)
                if len(parts) < 4:
                    continue
                filename = parts[-1]
                prompt_id = parts[-2]
                cer_range = parts[-3]
                files_to_copy.append((cer_range, prompt_id, filename, file_path))
    
    logger.info(f"总共找到 {len(files_to_copy)} 个待处理文件")
    return files_to_copy

def _copy_single_file(item: Tuple[str, str, str, str], 
                      output_base_dir: str,
                      dry_run: bool = False,
                      use_hardlink: bool = False) -> Tuple[bool, Optional[str], Optional[Dict]]:
    global _global_prompt_to_speaker
    
    try:
        cer_range, prompt_id, filename, source_path = item
        
        speaker_info = _global_prompt_to_speaker.get(prompt_id)
        if not speaker_info:
            return False, f"未找到映射: prompt_id={prompt_id}", None
        
        dataset_name, speaker_id = speaker_info
        
        # 确定 dataset_name 对应的目录 (逻辑与原脚本一致)
        if dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                target_dataset = 'kingasr612'
            elif 'King-ASR-725' in speaker_id:
                target_dataset = 'king-asr-725'
            else:
                return False, f"无法判断King-ASR子集: {speaker_id}", None
        else:
            dataset_mapping = {d['name']: d['target_dir'] for d in DATASET_CONFIG if d['target_dir']}
            target_dataset = dataset_mapping.get(dataset_name)
            if not target_dataset:
                return False, f"未知数据集: {dataset_name}", None

        # 确定 speaker_name (逻辑与原脚本一致)
        if dataset_name == 'childmandarin':
            target_speaker = f"childmandarin_{speaker_id}"
        elif dataset_name == 'chineseenglishchildren':
            target_speaker = f"chineseenglishchildren_{speaker_id}"
        elif dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                speaker_num = speaker_id.split('SPEAKER')[-1] if 'SPEAKER' in speaker_id else speaker_id
                target_speaker = f"kingasr612_{speaker_num}"
            elif 'King-ASR-725' in speaker_id:
                speaker_num = speaker_id.split('SPEAKER')[-1] if 'SPEAKER' in speaker_id else speaker_id
                target_speaker = f"king-asr-725_SPEAKER{speaker_num}"
            else:
                target_speaker = f"king-asr_{speaker_id}"
        elif dataset_name == 'speechocean762':
            if 'test_' in speaker_id:
                target_speaker = f"speechocean762_{speaker_id.split('test_')[-1]}"
            elif 'train_' in speaker_id:
                target_speaker = f"speechocean762_{speaker_id.split('train_')[-1]}"
            else:
                target_speaker = speaker_id if speaker_id.startswith('speechocean762_') else f"speechocean762_{speaker_id}"
        else:
            target_speaker = f"{target_dataset}_{speaker_id}"

        # 构建目标路径: output_base_dir/cer_range/target_dataset/target_speaker/prompt_id/filename
        target_dir = os.path.join(output_base_dir, cer_range, target_dataset, target_speaker, prompt_id)
        target_path = os.path.join(target_dir, filename)

        if dry_run:
            return True, None, {'source': source_path, 'target': target_path}

        os.makedirs(target_dir, exist_ok=True)
        
        if use_hardlink:
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
            
        return True, None, {'source': source_path, 'target': target_path}
    
    except Exception as e:
        return False, str(e), None

def main():
    parser = argparse.ArgumentParser(description="TTS克隆音频及JSON描述文件复制脚本 v2")
    parser.add_argument("--source_dirs", type=str, nargs='+', required=True, help="源目录列表")
    parser.add_argument("--output_base_dir", type=str, required=True, help="目标基础目录")
    parser.add_argument("--num_workers", type=int, default=32, help="工作进程数")
    parser.add_argument("--dry_run", action="store_true", help="模拟运行")
    parser.add_argument("--use_hardlink", action="store_true", help="使用硬链接")
    args = parser.parse_args()

    # 加载映射
    mapper = Utt2SpkMapper(DATASET_CONFIG)
    if mapper.load_all_mappings() == 0:
        return 1

    # 扫描文件
    files_to_copy = scan_files(args.source_dirs)
    if not files_to_copy:
        logger.error("未找到任何文件")
        return 1

    logger.info(f"开始{'模拟' if args.dry_run else ''}复制...")
    
    success_count = 0
    failed_count = 0
    
    with Pool(processes=args.num_workers, initializer=_init_worker, initargs=(mapper.prompt_to_speaker,)) as pool:
        copy_func = partial(_copy_single_file, output_base_dir=args.output_base_dir, dry_run=args.dry_run, use_hardlink=args.use_hardlink)
        results = pool.imap_unordered(copy_func, files_to_copy, chunksize=100)
        
        for idx, (success, error_msg, info) in enumerate(results, 1):
            if success:
                success_count += 1
                if args.dry_run and success_count % 500 == 0:
                    logger.info(f"Dry run example: {info['source']} -> {info['target']}")
            else:
                failed_count += 1
                if failed_count <= 10:
                    logger.warning(f"复制失败: {error_msg}")
            
            if idx % 1000 == 0:
                logger.info(f"进度: {idx}/{len(files_to_copy)} ({(idx/len(files_to_copy))*100:.1f}%)")

    logger.info(f"完成! 成功: {success_count}, 失败: {failed_count}")
    return 0 if failed_count == 0 else 2

if __name__ == "__main__":
    sys.exit(main())

