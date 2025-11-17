#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重组筛选后的TTS音频数据
将filtered_speech目录下的音频按照说话人组织到目标目录

功能：
1. 读取多个utt2spk文件，建立prompt_id到speaker_id的映射
2. 根据prompt_id前缀判断数据集类型（BAAI、King-ASR、Ocean等）
3. 将音频复制到目标目录，按照 {全局标签}_{数据集标签}_{speaker_id}/{全局标签}_{prompt_id}.wav 组织
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _copy_file_worker(args: Tuple[str, str, str, str, str, Dict[str, str]]) -> Tuple[bool, str, str]:
    """
    多进程文件复制工作函数（独立函数，可以被pickle）
    
    Args:
        args: (prompt_id, voiceprint_id, source_path, target_dir, global_prefix, utt2spk)
    
    Returns:
        (success, info/error_message, prompt_id)
    """
    prompt_id, voiceprint_id, source_path, target_dir, global_prefix, utt2spk = args
    
    try:
        if not os.path.exists(source_path):
            return False, f"源文件不存在: {source_path}", prompt_id
        
        # 查找speaker_id
        speaker_id = utt2spk.get(prompt_id)
        if speaker_id is None:
            return False, f"未找到speaker映射: {prompt_id}", prompt_id
        
        # 判断数据集标签
        if prompt_id.startswith('King-ASR'):
            dataset_label = 'King-ASR'
        elif prompt_id.startswith('speechocean762'):
            dataset_label = 'Ocean'
        elif prompt_id.startswith('Chinese_English'):
            dataset_label = 'CESSC'
        else:
            dataset_label = 'BAAI'
        
        # 构建目标路径
        speaker_dir_name = f"{global_prefix}_{dataset_label}_{speaker_id}"
        target_speaker_dir = os.path.join(target_dir, speaker_dir_name)
        # 文件名包含prompt_id和voiceprint_id以避免重复
        target_filename = f"{global_prefix}_{prompt_id}_{voiceprint_id}.wav"
        target_file_path = os.path.join(target_speaker_dir, target_filename)
        
        # 创建目录并复制文件
        os.makedirs(target_speaker_dir, exist_ok=True)
        shutil.copy2(source_path, target_file_path)
        
        return True, dataset_label, prompt_id
        
    except Exception as e:
        return False, f"复制失败: {e}", prompt_id


class AudioReorganizer:
    """音频数据重组器"""
    
    def __init__(self, source_dir: str, target_dir: str, 
                 global_prefix: str = "cosyvoice2-kidclone-filtered-20251116"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.global_prefix = global_prefix
        
        # prompt_id到speaker_id的映射
        self.utt2spk = {}
        
        # 数据集前缀到标签的映射
        self.dataset_prefix_map = {
            'King-ASR': 'King-ASR',
            'speechocean762': 'Ocean',
            'Chinese_English_Scripted_Speech_Corpus_Children': 'CESSC'
        }
        
        # 统计信息
        self.stats = {
            'total_prompts': 0,
            'total_files': 0,
            'copied_files': 0,
            'failed_files': 0,
            'missing_mapping': 0,
            'datasets': defaultdict(int)
        }
    
    def load_utt2spk_files(self, utt2spk_paths: List[str]):
        """加载utt2spk文件，建立prompt_id到speaker_id的映射"""
        for path in utt2spk_paths:
            if not os.path.exists(path):
                logger.warning(f"utt2spk文件不存在: {path}")
                continue
            
            logger.info(f"加载utt2spk: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        spk_id = parts[1]
                        self.utt2spk[utt_id] = spk_id
        
        logger.info(f"加载完成: {len(self.utt2spk)} 条映射")
    
    def get_dataset_label(self, prompt_id: str) -> str:
        """根据prompt_id前缀判断数据集标签"""
        # King-ASR数据集
        if prompt_id.startswith('King-ASR'):
            return 'King-ASR'
        
        # speechocean762数据集
        if prompt_id.startswith('speechocean762'):
            return 'Ocean'
        
        # Chinese_English数据集
        if prompt_id.startswith('Chinese_English'):
            return 'CESSC'
        
        # 默认为BAAI数据集
        return 'BAAI'
    
    def get_speaker_id(self, prompt_id: str) -> Tuple[str, str]:
        """
        获取speaker_id和数据集标签
        
        Returns:
            (speaker_id, dataset_label)
        """
        # 查找utt2spk映射
        speaker_id = self.utt2spk.get(prompt_id)
        
        if speaker_id is None:
            logger.warning(f"未找到speaker映射: {prompt_id}")
            self.stats['missing_mapping'] += 1
            return None, None
        
        dataset_label = self.get_dataset_label(prompt_id)
        
        return speaker_id, dataset_label
    
    def get_target_paths(self, prompt_id: str, voiceprint_id: str) -> Tuple[str, str]:
        """
        生成目标目录和文件路径
        
        Returns:
            (target_speaker_dir, target_file_path)
        """
        speaker_id, dataset_label = self.get_speaker_id(prompt_id)
        
        if speaker_id is None:
            return None, None
        
        # 构建说话人目录名: {全局标签}_{数据集标签}_{speaker_id}
        speaker_dir_name = f"{self.global_prefix}_{dataset_label}_{speaker_id}"
        target_speaker_dir = os.path.join(self.target_dir, speaker_dir_name)
        
        # 构建文件名: {全局标签}_{prompt_id}.wav
        target_filename = f"{self.global_prefix}_{prompt_id}.wav"
        target_file_path = os.path.join(target_speaker_dir, target_filename)
        
        return target_speaker_dir, target_file_path
    
    def scan_source_directory(self) -> List[Tuple[str, str, str]]:
        """
        扫描源目录，获取所有需要复制的文件
        
        Returns:
            List of (prompt_id, voiceprint_id, source_path)
        """
        files_to_copy = []
        
        audio_dir = os.path.join(self.source_dir, 'audio')
        if not os.path.exists(audio_dir):
            logger.error(f"音频目录不存在: {audio_dir}")
            return files_to_copy
        
        # 遍历prompt_id目录
        for prompt_id in os.listdir(audio_dir):
            prompt_dir = os.path.join(audio_dir, prompt_id)
            if not os.path.isdir(prompt_dir):
                continue
            
            self.stats['total_prompts'] += 1
            
            # 遍历voiceprint音频文件
            for filename in os.listdir(prompt_dir):
                if not filename.endswith('.wav'):
                    continue
                
                source_path = os.path.join(prompt_dir, filename)
                # 提取voiceprint_id（去除.wav后缀）
                voiceprint_id = filename[:-4]
                
                files_to_copy.append((prompt_id, voiceprint_id, source_path))
                self.stats['total_files'] += 1
        
        logger.info(f"扫描完成: {self.stats['total_prompts']} 个prompts, {self.stats['total_files']} 个文件")
        return files_to_copy
    
    def reorganize(self, num_workers: int = 8):
        """重组音频文件（使用多进程加速）"""
        logger.info("开始扫描源目录...")
        files_to_copy = self.scan_source_directory()
        
        if not files_to_copy:
            logger.warning("没有找到需要复制的文件")
            return
        
        logger.info(f"准备复制 {len(files_to_copy)} 个文件，使用 {num_workers} 个工作进程")
        
        # 创建目标根目录
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 准备工作参数（每个文件一个参数元组）
        work_args = [
            (prompt_id, voiceprint_id, source_path, self.target_dir, self.global_prefix, self.utt2spk)
            for prompt_id, voiceprint_id, source_path in files_to_copy
        ]
        
        # 使用多进程复制文件
        copied_count = 0
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_copy_file_worker, args) for args in work_args]
            
            for future in as_completed(futures):
                success, info, prompt_id = future.result()
                
                if success:
                    copied_count += 1
                    self.stats['copied_files'] += 1
                    self.stats['datasets'][info] += 1
                    
                    # 每1000个文件报告一次进度
                    if copied_count % 1000 == 0 or copied_count == len(files_to_copy):
                        logger.info(f"复制进度: {copied_count}/{len(files_to_copy)} "
                                  f"({copied_count/len(files_to_copy)*100:.1f}%)")
                else:
                    failed_count += 1
                    self.stats['failed_files'] += 1
                    if failed_count <= 10:  # 只显示前10个错误
                        logger.warning(info)
        
        logger.info("复制完成！")
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("音频数据重组统计")
        print("=" * 80)
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        print(f"全局标签: {self.global_prefix}")
        print()
        print(f"扫描到的prompts数: {self.stats['total_prompts']}")
        print(f"扫描到的文件数:   {self.stats['total_files']}")
        print(f"成功复制:         {self.stats['copied_files']}")
        print(f"复制失败:         {self.stats['failed_files']}")
        print(f"缺失映射:         {self.stats['missing_mapping']}")
        print()
        print("各数据集统计:")
        for dataset, count in sorted(self.stats['datasets'].items()):
            print(f"  {dataset}: {count}")
        print("=" * 80 + "\n")
    
    def save_summary(self):
        """保存统计摘要到文件"""
        summary_path = os.path.join(self.target_dir, "reorganize_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("音频数据重组统计\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"源目录: {self.source_dir}\n")
            f.write(f"目标目录: {self.target_dir}\n")
            f.write(f"全局标签: {self.global_prefix}\n\n")
            f.write(f"扫描到的prompts数: {self.stats['total_prompts']}\n")
            f.write(f"扫描到的文件数:   {self.stats['total_files']}\n")
            f.write(f"成功复制:         {self.stats['copied_files']}\n")
            f.write(f"复制失败:         {self.stats['failed_files']}\n")
            f.write(f"缺失映射:         {self.stats['missing_mapping']}\n\n")
            f.write("各数据集统计:\n")
            for dataset, count in sorted(self.stats['datasets'].items()):
                f.write(f"  {dataset}: {count}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"统计摘要已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="重组筛选后的TTS音频数据")
    parser.add_argument("--source_dir", type=str, 
                       default="/root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech",
                       help="源目录（filtered_speech）")
    parser.add_argument("--target_dir", type=str,
                       default="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments_20250808/merged_datasets_20250610_vad_segments/audio/cosyvoice2-kidclone-filtered-20251116",
                       help="目标目录")
    parser.add_argument("--global_prefix", type=str,
                       default="cosyvoice2-kidclone-filtered-20251116",
                       help="全局标签前缀")
    parser.add_argument("--utt2spk_baai", type=str,
                       default="/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/utt2spk",
                       help="BAAI数据集的utt2spk文件")
    parser.add_argument("--utt2spk_cessc", type=str,
                       default="/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/utt2spk",
                       help="Chinese_English数据集的utt2spk文件")
    parser.add_argument("--utt2spk_kingasr", type=str,
                       default="/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/utt2spk",
                       help="King-ASR数据集的utt2spk文件")
    parser.add_argument("--utt2spk_ocean", type=str,
                       default="/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/utt2spk",
                       help="speechocean762数据集的utt2spk文件")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="并行工作进程数（默认: 16）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查源目录
    if not os.path.exists(args.source_dir):
        logger.error(f"源目录不存在: {args.source_dir}")
        return 1
    
    print("\n" + "=" * 80)
    print("音频数据重组工具")
    print("=" * 80)
    print(f"源目录: {args.source_dir}")
    print(f"目标目录: {args.target_dir}")
    print(f"全局标签: {args.global_prefix}")
    print(f"工作进程数: {args.num_workers}")
    print("=" * 80 + "\n")
    
    try:
        # 创建重组器
        reorganizer = AudioReorganizer(args.source_dir, args.target_dir, args.global_prefix)
        
        # 加载utt2spk文件
        logger.info("加载utt2spk映射文件...")
        utt2spk_files = [
            args.utt2spk_baai,
            args.utt2spk_cessc,
            args.utt2spk_kingasr,
            args.utt2spk_ocean
        ]
        reorganizer.load_utt2spk_files(utt2spk_files)
        
        # 执行重组
        logger.info("开始重组音频文件...")
        reorganizer.reorganize(num_workers=args.num_workers)
        
        # 打印和保存统计信息
        reorganizer.print_summary()
        reorganizer.save_summary()
        
        logger.info("完成！")
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

