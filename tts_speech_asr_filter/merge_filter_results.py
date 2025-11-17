#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并ASR筛选和声纹筛选结果，根据双重阈值筛选TTS音频

功能：
1. 读取ASR筛选结果（CER）和声纹筛选结果（相似度）
2. 根据双重阈值筛选：CER <= cer_threshold AND similarity >= sim_threshold
3. 复制通过筛选的音频文件到目标目录
4. 生成详细的筛选报告和统计信息
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FilterResultMerger:
    """合并ASR和声纹筛选结果"""
    
    def __init__(self, asr_result_path: str, voiceprint_result_path: str,
                 cer_threshold: float = 0.05, similarity_threshold: float = 0.65):
        self.asr_result_path = asr_result_path
        self.voiceprint_result_path = voiceprint_result_path
        self.cer_threshold = cer_threshold
        self.similarity_threshold = similarity_threshold
        
        # 加载结果
        self.asr_results = {}  # key: (prompt_id, voiceprint_id) -> result
        self.vp_results = {}   # key: (prompt_id, voiceprint_id) -> result
        
    def load_results(self):
        """加载ASR和声纹筛选结果（仅提取必要字段以加快速度）"""
        logger.info(f"加载ASR筛选结果: {self.asr_result_path}")
        with open(self.asr_result_path, 'r', encoding='utf-8') as f:
            asr_data = json.load(f)
        
        # 解析ASR结果 - 只保留必要字段
        for item in asr_data.get('filter_results', []):
            voiceprint_id = item.get('voiceprint_id', '')
            prompt_id = item.get('prompt_id', '')
            
            # 如果prompt_id为空，尝试从路径提取
            if not prompt_id:
                audio_path = item.get('audio_path', '')
                if audio_path and 'zero_shot' in audio_path:
                    parts = audio_path.split('/')
                    idx = parts.index('zero_shot')
                    if idx + 1 < len(parts):
                        prompt_id = parts[idx + 1]
            
            if prompt_id and voiceprint_id:
                key = (prompt_id, voiceprint_id)
                # 只保存CER，减少内存占用
                self.asr_results[key] = {
                    'cer': item.get('cer', 1.0),
                    'groundtruth_text': item.get('groundtruth_text', ''),
                    'transcription': item.get('transcription', '')
                }
        
        logger.info(f"加载ASR结果: {len(self.asr_results)} 条")
        
        logger.info(f"加载声纹筛选结果: {self.voiceprint_result_path}")
        with open(self.voiceprint_result_path, 'r', encoding='utf-8') as f:
            vp_data = json.load(f)
        
        # 解析声纹结果
        for item in vp_data.get('filter_results', []):
            prompt_id = item.get('prompt_id', '')
            voiceprint_id = item.get('voiceprint_id', '')
            
            if prompt_id and voiceprint_id:
                key = (prompt_id, voiceprint_id)
                self.vp_results[key] = item
        
        logger.info(f"加载声纹结果: {len(self.vp_results)} 条")
    
    def merge_and_filter(self) -> List[Dict]:
        """合并并筛选结果 - 简化逻辑：只用CER和max(similarity_vad, similarity_original)"""
        merged_results = []
        
        # 找出在两个结果中都存在的音频
        asr_keys = set(self.asr_results.keys())
        vp_keys = set(self.vp_results.keys())
        common_keys = asr_keys & vp_keys
        
        logger.info(f"ASR独有: {len(asr_keys - vp_keys)} 条")
        logger.info(f"声纹独有: {len(vp_keys - asr_keys)} 条")
        logger.info(f"共同音频: {len(common_keys)} 条")
        
        for key in common_keys:
            prompt_id, voiceprint_id = key
            asr_item = self.asr_results[key]
            vp_item = self.vp_results[key]
            
            # 提取关键信息
            cer = asr_item.get('cer', 1.0)
            similarity_vad = vp_item.get('similarity_vad', 0.0)
            similarity_original = vp_item.get('similarity_original', 0.0)
            
            # 使用VAD和Original中的较大值
            similarity = max(similarity_vad, similarity_original)
            
            # 获取音频路径
            tts_path = vp_item.get('tts_path', '')
            source_path = vp_item.get('source_path', '')
            
            # 简化筛选：只检查CER和相似度
            cer_ok = cer <= self.cer_threshold
            sim_ok = similarity >= self.similarity_threshold
            passed = cer_ok and sim_ok
            
            # 合并结果
            merged_item = {
                'prompt_id': prompt_id,
                'voiceprint_id': voiceprint_id,
                'tts_path': tts_path,
                'source_path': source_path,
                'passed': passed,
                'asr': {
                    'cer': cer,
                    'cer_threshold': self.cer_threshold,
                    'cer_ok': cer_ok,
                    'groundtruth_text': asr_item.get('groundtruth_text', ''),
                    'transcription': asr_item.get('transcription', '')
                },
                'voiceprint': {     
                    'similarity_vad': similarity_vad,
                    'similarity_original': similarity_original,
                    'similarity': similarity,  # 使用的最大值
                    'similarity_threshold': self.similarity_threshold,
                    'sim_ok': sim_ok,
                    'vad_info': vp_item.get('vad', {})
                },
                'reason': self._get_filter_reason(cer_ok, sim_ok)
            }
            
            merged_results.append(merged_item)
        
        return merged_results
    
    def _get_filter_reason(self, cer_ok: bool, sim_ok: bool) -> str:
        """获取筛选原因"""
        if not cer_ok:
            return f"CER超标 (threshold={self.cer_threshold})"
        if not sim_ok:
            return f"相似度不足 (threshold={self.similarity_threshold})"
        return "通过"


def _copy_single_audio(item: Dict, output_dir: str, organize_by_prompt: bool) -> Tuple[bool, Optional[str]]:
    """复制单个音频文件（用于多进程）"""
    try:
        tts_path = item['tts_path']
        prompt_id = item['prompt_id']
        voiceprint_id = item['voiceprint_id']
        
        if not os.path.exists(tts_path):
            return False, f"音频文件不存在: {tts_path}"
        
        # 组织输出目录结构
        if organize_by_prompt:
            # 按prompt组织: output_dir/<prompt_id>/<voiceprint_id>.wav
            target_dir = os.path.join(output_dir, prompt_id)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{voiceprint_id}.wav")
        else:
            # 扁平结构: output_dir/<prompt_id>_<voiceprint_id>.wav
            target_path = os.path.join(output_dir, f"{prompt_id}_{voiceprint_id}.wav")
        
        shutil.copy2(tts_path, target_path)
        return True, None
        
    except Exception as e:
        return False, f"复制失败 {item.get('tts_path', 'unknown')}: {e}"


def copy_filtered_audio(results: List[Dict], output_dir: str, 
                       organize_by_prompt: bool = True,
                       num_workers: int = 8) -> Tuple[int, int]:
    """复制通过筛选的音频文件（多进程加速）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选出需要复制的音频
    to_copy = [item for item in results if item['passed']]
    
    if not to_copy:
        logger.warning("没有音频需要复制")
        return 0, 0
    
    logger.info(f"准备复制 {len(to_copy)} 个音频文件，使用 {num_workers} 个工作进程")
    
    copied_count = 0
    failed_count = 0
    
    # 使用多进程复制
    copy_func = partial(_copy_single_audio, output_dir=output_dir, organize_by_prompt=organize_by_prompt)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_func, item): item for item in to_copy}
        
        for future in as_completed(futures):
            success, error_msg = future.result()
            
            if success:
                copied_count += 1
                if copied_count % 100 == 0 or copied_count == len(to_copy):
                    logger.info(f"复制进度: {copied_count}/{len(to_copy)} ({copied_count/len(to_copy)*100:.1f}%)")
            else:
                failed_count += 1
                if error_msg:
                    logger.warning(error_msg)
    
    return copied_count, failed_count


def generate_statistics(results: List[Dict]) -> Dict:
    """生成统计信息 - 简化版"""
    stats = {
        'total': len(results),
        'passed': sum(1 for r in results if r['passed']),
        'filtered': sum(1 for r in results if not r['passed']),
        'cer_failed': sum(1 for r in results if not r['asr']['cer_ok']),
        'sim_failed': sum(1 for r in results if not r['voiceprint']['sim_ok']),
        'pass_rate': 0.0,
    }
    
    if stats['total'] > 0:
        stats['pass_rate'] = stats['passed'] / stats['total'] * 100
    
    # 按原因统计
    reason_counts = defaultdict(int)
    for r in results:
        reason_counts[r['reason']] += 1
    stats['filter_reasons'] = dict(reason_counts)
    
    # CER和相似度统计
    cer_values = [r['asr']['cer'] for r in results]
    sim_values = [r['voiceprint']['similarity'] for r in results]  # 使用max(vad, original)
    
    if cer_values:
        import numpy as np
        stats['cer_stats'] = {
            'mean': float(np.mean(cer_values)),
            'median': float(np.median(cer_values)),
            'std': float(np.std(cer_values)),
            'min': float(np.min(cer_values)),
            'max': float(np.max(cer_values))
        }
    
    if sim_values:
        import numpy as np
        stats['similarity_stats'] = {
            'mean': float(np.mean(sim_values)),
            'median': float(np.median(sim_values)),
            'std': float(np.std(sim_values)),
            'min': float(np.min(sim_values)),
            'max': float(np.max(sim_values))
        }
    
    return stats


def save_results(results: List[Dict], stats: Dict, output_dir: str,
                asr_path: str, vp_path: str, cer_threshold: float, sim_threshold: float):
    """保存结果和统计信息"""
    # 保存完整结果JSON
    result_json_path = os.path.join(output_dir, "merged_filter_results.json")
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'source_files': {
            'asr_result': asr_path,
            'voiceprint_result': vp_path
        },
        'thresholds': {
            'cer_threshold': cer_threshold,
            'similarity_threshold': sim_threshold
        },
        'statistics': stats,
        'filter_results': results
    }
    
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {result_json_path}")
    
    # 保存通过列表
    passed_list_path = os.path.join(output_dir, "passed_list.txt")
    with open(passed_list_path, 'w', encoding='utf-8') as f:
        for r in results:
            if r['passed']:
                f.write(f"{r['tts_path']}\n")
    logger.info(f"通过列表已保存: {passed_list_path}")
    
    # 保存筛除列表
    filtered_list_path = os.path.join(output_dir, "filtered_list.txt")
    with open(filtered_list_path, 'w', encoding='utf-8') as f:
        for r in results:
            if not r['passed']:
                f.write(f"{r['tts_path']}\t{r['reason']}\n")
    logger.info(f"筛除列表已保存: {filtered_list_path}")
    
    # 保存统计摘要
    summary_path = os.path.join(output_dir, "filter_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TTS音频双重筛选结果统计\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("筛选阈值:\n")
        f.write(f"  CER阈值: {cer_threshold}\n")
        f.write(f"  相似度阈值: {sim_threshold}\n\n")
        
        f.write("统计信息:\n")
        f.write(f"  总音频数: {stats['total']}\n")
        f.write(f"  通过筛选: {stats['passed']} ({stats['pass_rate']:.2f}%)\n")
        f.write(f"  被筛除: {stats['filtered']} ({100-stats['pass_rate']:.2f}%)\n\n")
        
        f.write("失败原因分布:\n")
        for reason, count in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
            pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"  {reason}: {count} ({pct:.2f}%)\n")
        
        if 'cer_stats' in stats:
            f.write("\nCER统计:\n")
            f.write(f"  平均值: {stats['cer_stats']['mean']:.4f}\n")
            f.write(f"  中位数: {stats['cer_stats']['median']:.4f}\n")
            f.write(f"  标准差: {stats['cer_stats']['std']:.4f}\n")
            f.write(f"  最小值: {stats['cer_stats']['min']:.4f}\n")
            f.write(f"  最大值: {stats['cer_stats']['max']:.4f}\n")
        
        if 'similarity_stats' in stats:
            f.write("\n相似度统计:\n")
            f.write(f"  平均值: {stats['similarity_stats']['mean']:.4f}\n")
            f.write(f"  中位数: {stats['similarity_stats']['median']:.4f}\n")
            f.write(f"  标准差: {stats['similarity_stats']['std']:.4f}\n")
            f.write(f"  最小值: {stats['similarity_stats']['min']:.4f}\n")
            f.write(f"  最大值: {stats['similarity_stats']['max']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"统计摘要已保存: {summary_path}")


def print_summary(stats: Dict):
    """打印统计摘要到控制台"""
    print("\n" + "=" * 80)
    print("TTS音频双重筛选结果统计")
    print("=" * 80)
    print(f"总音频数:     {stats['total']}")
    print(f"通过筛选:     {stats['passed']} ({stats['pass_rate']:.2f}%)")
    print(f"被筛除:       {stats['filtered']} ({100-stats['pass_rate']:.2f}%)")
    print(f"\n失败原因分布:")
    for reason, count in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
        pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {reason}: {count} ({pct:.2f}%)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="合并ASR和声纹筛选结果，筛选TTS音频")
    parser.add_argument("--asr_result", type=str, required=True,
                       help="ASR筛选结果JSON文件路径")
    parser.add_argument("--voiceprint_result", type=str, required=True,
                       help="声纹筛选结果JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（存放筛选后的音频和结果）")
    parser.add_argument("--cer_threshold", type=float, default=0.05,
                       help="CER阈值（默认: 0.05）")
    parser.add_argument("--similarity_threshold", type=float, default=0.65,
                       help="声纹相似度阈值（默认: 0.65）")
    parser.add_argument("--no_copy_audio", action="store_true",
                       help="不复制音频文件，只生成结果报告")
    parser.add_argument("--flat_structure", action="store_true",
                       help="使用扁平目录结构（默认按prompt组织）")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="复制音频的并行工作进程数（默认: 8）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入文件
    if not os.path.exists(args.asr_result):
        logger.error(f"ASR结果文件不存在: {args.asr_result}")
        return 1
    
    if not os.path.exists(args.voiceprint_result):
        logger.error(f"声纹结果文件不存在: {args.voiceprint_result}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("TTS音频双重筛选")
    print("=" * 80)
    print(f"ASR结果: {args.asr_result}")
    print(f"声纹结果: {args.voiceprint_result}")
    print(f"输出目录: {args.output_dir}")
    print(f"CER阈值: {args.cer_threshold}")
    print(f"相似度阈值: {args.similarity_threshold}")
    print("=" * 80 + "\n")
    
    try:
        # 加载和合并结果
        merger = FilterResultMerger(
            args.asr_result,
            args.voiceprint_result,
            args.cer_threshold,
            args.similarity_threshold
        )
        
        logger.info("加载筛选结果...")
        merger.load_results()
        
        logger.info("合并并筛选结果...")
        merged_results = merger.merge_and_filter()
        
        logger.info("生成统计信息...")
        stats = generate_statistics(merged_results)
        
        # 保存结果
        logger.info("保存结果文件...")
        save_results(merged_results, stats, args.output_dir,
                    args.asr_result, args.voiceprint_result,
                    args.cer_threshold, args.similarity_threshold)
        
        # 复制音频文件
        if not args.no_copy_audio:
            logger.info("复制通过筛选的音频文件...")
            audio_dir = os.path.join(args.output_dir, "audio")
            copied, failed = copy_filtered_audio(
                merged_results, audio_dir,
                organize_by_prompt=not args.flat_structure,
                num_workers=args.num_workers
            )
            logger.info(f"音频复制完成: 成功 {copied}, 失败 {failed}")
        else:
            logger.info("跳过音频复制（--no_copy_audio）")
        
        # 打印摘要
        print_summary(stats)
        
        logger.info(f"完成！结果保存在: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

