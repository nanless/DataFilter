#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按声纹相似度分档抽样脚本

功能：
1. 从筛选结果中按声纹相似度分档（0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0）
2. 每个档次随机抽取N个样本
3. 复制原始音频（prompt/source）和TTS复刻音频
4. 生成每个样本的详细信息（CER、相似度、文本等）

用途：人工听辨不同相似度档次的音频质量差异
"""

import os
import sys
import json
import shutil
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilaritySampler:
    """声纹相似度分档抽样器"""
    
    def __init__(self, result_json_path: str, 
                 bins: List[Tuple[float, float]] = None,
                 samples_per_bin: int = 20,
                 similarity_type: str = 'both'):
        self.result_json_path = result_json_path
        self.samples_per_bin = samples_per_bin
        self.similarity_type = similarity_type  # 'vad', 'original', 'both'
        
        # 默认相似度分档（考虑负值，细分0.5以下）
        if bins is None:
            self.bins = [
                (-1.0, 0.0),   # 负值档
                (0.0, 0.3),    # 极低
                (0.3, 0.5),    # 很低
                (0.5, 0.6),    # 低
                (0.6, 0.7),    # 中等偏低
                (0.7, 0.8),    # 中等
                (0.8, 0.9),    # 高
                (0.9, 1.0)     # 极高
            ]
        else:
            self.bins = bins
        
        self.results = []
        # 如果是both模式，需要分别存储
        if similarity_type == 'both':
            self.binned_samples_vad = defaultdict(list)
            self.binned_samples_original = defaultdict(list)
        else:
            self.binned_samples = defaultdict(list)
    
    def load_results(self):
        """加载筛选结果"""
        logger.info(f"加载筛选结果: {self.result_json_path}")
        
        with open(self.result_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results = data.get('filter_results', [])
        logger.info(f"加载了 {len(self.results)} 条结果")
    
    def bin_by_similarity(self):
        """按相似度分档"""
        logger.info(f"按相似度分档（类型: {self.similarity_type}）...")
        
        for item in self.results:
            # 兼容两种数据格式：
            # 1. 嵌套格式（合并结果）：item['voiceprint']['similarity_vad']
            # 2. 扁平格式（声纹筛选结果）：item['similarity_vad']
            
            # 尝试从嵌套格式获取
            vp_info = item.get('voiceprint', {})
            if vp_info:
                similarity_vad = vp_info.get('similarity_vad')
                similarity_original = vp_info.get('similarity_original')
            else:
                # 扁平格式
                similarity_vad = item.get('similarity_vad')
                similarity_original = item.get('similarity_original')
            
            # 根据similarity_type处理
            if self.similarity_type == 'both':
                # VAD相似度分档
                if similarity_vad is not None:
                    self._bin_single_similarity(item, similarity_vad, self.binned_samples_vad, 'VAD')
                
                # Original相似度分档
                if similarity_original is not None:
                    self._bin_single_similarity(item, similarity_original, self.binned_samples_original, 'Original')
            
            elif self.similarity_type == 'vad':
                if similarity_vad is not None:
                    self._bin_single_similarity(item, similarity_vad, self.binned_samples, 'VAD')
            
            elif self.similarity_type == 'original':
                if similarity_original is not None:
                    self._bin_single_similarity(item, similarity_original, self.binned_samples, 'Original')
        
        # 打印统计
        if self.similarity_type == 'both':
            logger.info("VAD相似度分布:")
            for bin_key in sorted(self.binned_samples_vad.keys()):
                count = len(self.binned_samples_vad[bin_key])
                logger.info(f"  {bin_key}: {count} 个样本")
            
            logger.info("\nOriginal相似度分布:")
            for bin_key in sorted(self.binned_samples_original.keys()):
                count = len(self.binned_samples_original[bin_key])
                logger.info(f"  {bin_key}: {count} 个样本")
        else:
            logger.info("相似度分布:")
            for bin_key in sorted(self.binned_samples.keys()):
                count = len(self.binned_samples[bin_key])
                logger.info(f"  {bin_key}: {count} 个样本")
    
    def _bin_single_similarity(self, item: Dict, similarity: float, 
                               bin_dict: Dict, sim_type: str):
        """将单个样本按相似度分档"""
        for bin_min, bin_max in self.bins:
            # 处理边界情况
            if bin_min <= similarity < bin_max or (similarity == bin_max and bin_max == 1.0):
                bin_key = f"{bin_min:.2f}-{bin_max:.2f}"
                # 在item中标记相似度类型，方便后续识别
                item_copy = item.copy()
                item_copy['_similarity_type'] = sim_type
                item_copy['_similarity_value'] = similarity
                bin_dict[bin_key].append(item_copy)
                break
    
    def sample_from_bins(self) -> Dict[str, Dict[str, List[Dict]]]:
        """从每个档次随机抽样"""
        logger.info(f"从每个档次随机抽取 {self.samples_per_bin} 个样本...")
        
        sampled = {}
        
        if self.similarity_type == 'both':
            # 分别处理VAD和Original
            sampled['vad'] = self._sample_single_type(self.binned_samples_vad, 'VAD')
            sampled['original'] = self._sample_single_type(self.binned_samples_original, 'Original')
        else:
            sampled['single'] = self._sample_single_type(self.binned_samples, 
                                                         'VAD' if self.similarity_type == 'vad' else 'Original')
        
        return sampled
    
    def _sample_single_type(self, binned_samples: Dict, sim_type: str) -> Dict[str, List[Dict]]:
        """从单个类型的分档中抽样"""
        sampled = {}
        
        logger.info(f"\n处理 {sim_type} 相似度:")
        for bin_key, items in binned_samples.items():
            if len(items) == 0:
                logger.warning(f"  档次 {bin_key} 没有样本")
                sampled[bin_key] = []
                continue
            
            # 如果样本数少于需要的数量，全部使用
            if len(items) <= self.samples_per_bin:
                sampled[bin_key] = items
                logger.info(f"  {bin_key}: 仅有 {len(items)} 个样本（全部使用）")
            else:
                sampled[bin_key] = random.sample(items, self.samples_per_bin)
                logger.info(f"  {bin_key}: 从 {len(items)} 个中抽取 {self.samples_per_bin} 个")
        
        return sampled


def generate_sample_info(item: Dict, sample_id: int) -> str:
    """生成样本信息文本"""
    # 获取相似度类型和值
    sim_type = item.get('_similarity_type', 'Unknown')
    sim_value = item.get('_similarity_value', 0.0)
    
    # 兼容两种数据格式：嵌套格式（合并结果）和扁平格式（声纹筛选结果）
    vp_info = item.get('voiceprint', {})
    asr_info = item.get('asr', {})
    
    # 从扁平格式或嵌套格式获取相似度
    if vp_info:
        sim_original = vp_info.get('similarity_original', 0.0)
        sim_vad = vp_info.get('similarity_vad')
        vad_info = vp_info.get('vad_info', {})
    else:
        sim_original = item.get('similarity_original', 0.0)
        sim_vad = item.get('similarity_vad')
        vad_info = item.get('vad', {})
    
    info_lines = [
        "=" * 80,
        f"样本信息 - Sample {sample_id} ({sim_type}相似度)",
        "=" * 80,
        "",
        "基本信息:",
        f"  Prompt ID: {item.get('prompt_id', 'N/A')}",
        f"  Voiceprint ID: {item.get('voiceprint_id', 'N/A')}",
    ]
    
    # 如果是合并结果，显示筛选信息
    if item.get('passed') is not None:
        info_lines.extend([
            f"  是否通过筛选: {'✓ 是' if item.get('passed') else '✗ 否'}",
            f"  筛选原因: {item.get('reason', 'N/A')}",
        ])
    
    info_lines.extend([
        f"  抽样依据: {sim_type}相似度 = {sim_value:.4f}",
        "",
    ])
    
    # ASR信息（如果有）
    if asr_info:
        info_lines.extend([
            "ASR识别信息:",
            f"  CER (字符错误率): {asr_info.get('cer', 0.0):.4f} ({asr_info.get('cer', 0.0)*100:.2f}%)",
            f"  CER阈值: {asr_info.get('cer_threshold', 0.0):.4f}",
            f"  CER是否达标: {'✓ 是' if asr_info.get('cer_ok') else '✗ 否'}",
            f"  原始文本 (Groundtruth): {asr_info.get('groundtruth_text', 'N/A')}",
            f"  识别文本 (Transcription): {asr_info.get('transcription', 'N/A')}",
        ])
        
        if asr_info.get('error_message'):
            info_lines.append(f"  错误信息: {asr_info.get('error_message')}")
        
        info_lines.append("")
    
    # 声纹相似度信息
    info_lines.extend([
        "声纹相似度信息:",
        f"  相似度 (原始音频): {sim_original:.4f}",
    ])
    
    if sim_vad is not None:
        info_lines.append(f"  相似度 (VAD处理后): {sim_vad:.4f}")
    
    # 如果是合并结果，显示阈值和达标信息
    if vp_info:
        info_lines.extend([
            f"  相似度阈值: {vp_info.get('similarity_threshold', 0.0):.4f}",
            f"  相似度是否达标: {'✓ 是' if vp_info.get('sim_ok') else '✗ 否'}",
        ])
    
    # VAD信息
    if vad_info.get('used'):
        info_lines.extend([
            "",
            "VAD处理信息:",
            f"  使用VAD: ✓ 是",
            f"  源音频有效占比: {vad_info.get('src_active_ratio', 0.0):.2%}",
            f"  TTS音频有效占比: {vad_info.get('tts_active_ratio', 0.0):.2%}",
        ])
    
    # 错误信息
    error_msg = vp_info.get('error_message') if vp_info else item.get('error_message')
    if error_msg:
        info_lines.append(f"  错误信息: {error_msg}")
    
    info_lines.extend([
        "",
        "音频文件路径:",
        f"  源音频 (Source): {item.get('source_path', 'N/A')}",
        f"  TTS音频 (TTS): {item.get('tts_path', 'N/A')}",
        "",
        "=" * 80,
    ])
    
    return "\n".join(info_lines)


def copy_sample_files(item: Dict, output_dir: str, sample_id: int) -> Tuple[bool, str]:
    """复制样本音频文件"""
    try:
        source_path = item.get('source_path', '')
        tts_path = item.get('tts_path', '')
        
        # 创建样本目录
        sample_dir = os.path.join(output_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 复制源音频
        if source_path and os.path.exists(source_path):
            source_ext = os.path.splitext(source_path)[1]
            source_target = os.path.join(sample_dir, f"source{source_ext}")
            shutil.copy2(source_path, source_target)
        else:
            return False, f"源音频不存在: {source_path}"
        
        # 复制TTS音频
        if tts_path and os.path.exists(tts_path):
            tts_ext = os.path.splitext(tts_path)[1]
            tts_target = os.path.join(sample_dir, f"tts{tts_ext}")
            shutil.copy2(tts_path, tts_target)
        else:
            return False, f"TTS音频不存在: {tts_path}"
        
        # 生成信息文件
        info_text = generate_sample_info(item, sample_id)
        info_path = os.path.join(sample_dir, "info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_text)
        
        # 生成简短的README
        sim_type = item.get('_similarity_type', 'Unknown')
        sim_value = item.get('_similarity_value', 0.0)
        
        readme_text = f"""样本 {sample_id} ({sim_type}相似度)

相似度({sim_type}): {sim_value:.4f}
CER: {item.get('asr', {}).get('cer', 0.0):.4f} ({item.get('asr', {}).get('cer', 0.0)*100:.2f}%)

文本: {item.get('asr', {}).get('groundtruth_text', 'N/A')}

文件说明:
- source.wav: 原始音频（参考音频/prompt）
- tts.wav: TTS复刻音频
- info.txt: 详细信息
"""
        readme_path = os.path.join(sample_dir, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_text)
        
        return True, sample_dir
        
    except Exception as e:
        return False, f"复制失败: {e}"


def save_samples(sampled: Dict, output_dir: str, similarity_type: str):
    """保存抽样结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_lines = [
        "=" * 80,
        "声纹相似度分档抽样 - 统计摘要",
        "=" * 80,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"抽样模式: {similarity_type}",
        "",
        "抽样统计:",
    ]
    
    total_samples = 0
    total_success = 0
    total_failed = 0
    
    # 处理不同的抽样模式
    if similarity_type == 'both':
        # 处理VAD和Original两种类型
        for sim_type, type_samples in sampled.items():
            summary_lines.append(f"\n{sim_type.upper()}相似度:")
            type_dir = os.path.join(output_dir, sim_type)
            
            for bin_key in sorted(type_samples.keys()):
                items = type_samples[bin_key]
                if not items:
                    continue
                
                bin_dir = os.path.join(type_dir, f"similarity_{bin_key}")
                os.makedirs(bin_dir, exist_ok=True)
                
                logger.info(f"\n处理{sim_type.upper()}档次 {bin_key} ({len(items)} 个样本)...")
                
                success_count = 0
                failed_count = 0
                
                for idx, item in enumerate(items, 1):
                    success, result = copy_sample_files(item, bin_dir, idx)
                    
                    if success:
                        success_count += 1
                        logger.debug(f"  ✓ 样本 {idx}: {result}")
                    else:
                        failed_count += 1
                        logger.warning(f"  ✗ 样本 {idx}: {result}")
                
                logger.info(f"  完成: 成功 {success_count}, 失败 {failed_count}")
                
                summary_lines.append(f"  {bin_key}: {success_count} 个样本")
                
                total_samples += len(items)
                total_success += success_count
                total_failed += failed_count
                
                # 生成每个档次的README
                bin_readme = os.path.join(bin_dir, "README.txt")
                with open(bin_readme, 'w', encoding='utf-8') as f:
                    f.write(f"相似度类型: {sim_type.upper()}\n")
                    f.write(f"相似度档次: {bin_key}\n")
                    f.write(f"样本数量: {len(items)}\n")
                    f.write(f"\n每个样本目录包含:\n")
                    f.write(f"- source.wav: 原始音频\n")
                    f.write(f"- tts.wav: TTS复刻音频\n")
                    f.write(f"- info.txt: 详细信息\n")
                    f.write(f"- README.txt: 简要说明\n")
    else:
        # 单一类型（vad或original）
        type_samples = sampled['single']
        
        for bin_key in sorted(type_samples.keys()):
            items = type_samples[bin_key]
            if not items:
                continue
            
            bin_dir = os.path.join(output_dir, f"similarity_{bin_key}")
            os.makedirs(bin_dir, exist_ok=True)
            
            logger.info(f"\n处理档次 {bin_key} ({len(items)} 个样本)...")
            
            success_count = 0
            failed_count = 0
            
            for idx, item in enumerate(items, 1):
                success, result = copy_sample_files(item, bin_dir, idx)
                
                if success:
                    success_count += 1
                    logger.debug(f"  ✓ 样本 {idx}: {result}")
                else:
                    failed_count += 1
                    logger.warning(f"  ✗ 样本 {idx}: {result}")
            
            logger.info(f"  完成: 成功 {success_count}, 失败 {failed_count}")
            
            summary_lines.append(f"  {bin_key}: {success_count} 个样本")
            
            total_samples += len(items)
            total_success += success_count
            total_failed += failed_count
            
            # 生成每个档次的README
            bin_readme = os.path.join(bin_dir, "README.txt")
            with open(bin_readme, 'w', encoding='utf-8') as f:
                f.write(f"相似度档次: {bin_key}\n")
                f.write(f"样本数量: {len(items)}\n")
                f.write(f"\n每个样本目录包含:\n")
                f.write(f"- source.wav: 原始音频\n")
                f.write(f"- tts.wav: TTS复刻音频\n")
                f.write(f"- info.txt: 详细信息\n")
                f.write(f"- README.txt: 简要说明\n")
    
    summary_lines.extend([
        "",
        "总计:",
        f"  总样本数: {total_samples}",
        f"  成功: {total_success}",
        f"  失败: {total_failed}",
        "",
        "=" * 80,
    ])
    
    # 保存摘要
    summary_path = os.path.join(output_dir, "samples_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    
    logger.info(f"\n抽样摘要保存到: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="按声纹相似度分档抽样，用于人工听辨质量差异")
    parser.add_argument("--result_json", type=str, required=True,
                       help="声纹筛选结果JSON文件路径（或合并筛选结果）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（存放抽样样本）")
    parser.add_argument("--samples_per_bin", type=int, default=20,
                       help="每个档次抽取的样本数（默认: 20）")
    parser.add_argument("--similarity_type", type=str, default='both',
                       choices=['vad', 'original', 'both'],
                       help="相似度类型：vad（VAD处理后）, original（原始音频）, both（都抽样）（默认: both）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，确保可重复（默认: 42）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入文件
    if not os.path.exists(args.result_json):
        logger.error(f"结果文件不存在: {args.result_json}")
        return 1
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("\n" + "=" * 80)
    print("声纹相似度分档抽样")
    print("=" * 80)
    print(f"输入文件: {args.result_json}")
    print(f"输出目录: {args.output_dir}")
    print(f"每档样本数: {args.samples_per_bin}")
    print(f"相似度类型: {args.similarity_type}")
    print(f"随机种子: {args.seed}")
    print("=" * 80 + "\n")
    
    try:
        # 创建采样器
        sampler = SimilaritySampler(
            args.result_json,
            samples_per_bin=args.samples_per_bin,
            similarity_type=args.similarity_type
        )
        
        # 加载结果
        sampler.load_results()
        
        # 分档
        sampler.bin_by_similarity()
        
        # 抽样
        sampled = sampler.sample_from_bins()
        
        # 保存样本
        logger.info("\n开始复制样本文件...")
        save_samples(sampled, args.output_dir, args.similarity_type)
        
        print("\n" + "=" * 80)
        print("✓ 完成！")
        print("=" * 80)
        print(f"样本目录: {args.output_dir}")
        print(f"查看摘要: cat {os.path.join(args.output_dir, 'samples_summary.txt')}")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

