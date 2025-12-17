#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于CER值对TTS音频进行分类筛选

功能：
1. 读取ASR筛选结果JSON文件
2. 根据CER值将音频分类到不同的目录（cer0, cer0-0.05, cer0.05-0.1等）
3. 每个音频放在以prompt_id命名的子目录下
4. 每个音频旁边生成一个JSON文件，包含groundtruth、transcription、CER等信息
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
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cer_range_dir(cer: float) -> str:
    """
    根据CER值返回对应的目录名称
    
    分类规则：
    - cer == 0: cer0
    - 0 < cer <= 0.05: cer0-0.05
    - 0.05 < cer <= 0.1: cer0.05-0.1
    - 0.1 < cer <= 0.15: cer0.1-0.15
    - 0.15 < cer <= 0.2: cer0.15-0.2
    - 0.2 < cer <= 0.25: cer0.2-0.25
    - cer > 0.25: cer0.25+
    """
    if cer == 0.0:
        return "cer0"
    elif 0 < cer <= 0.05:
        return "cer0-0.05"
    elif 0.05 < cer <= 0.1:
        return "cer0.05-0.1"
    elif 0.1 < cer <= 0.15:
        return "cer0.1-0.15"
    elif 0.15 < cer <= 0.2:
        return "cer0.15-0.2"
    elif 0.2 < cer <= 0.25:
        return "cer0.2-0.25"
    else:
        return "cer0.25+"


def _process_single_audio(item: Dict, base_output_dir: str, cer_threshold: float, 
                          target_sr: Optional[int] = None) -> Tuple[bool, Optional[str], str]:
    """
    处理单个音频文件（用于多进程）
    
    参数:
        item: 音频信息字典
        base_output_dir: 输出基础目录
        cer_threshold: CER阈值
        target_sr: 目标采样率，如果为None则不重采样，直接复制
    
    返回: (success, error_msg, cer_range_dir)
    """
    try:
        audio_path = item.get('audio_path', '')
        prompt_id = item.get('prompt_id', '')
        voiceprint_id = item.get('voiceprint_id', '')
        cer = item.get('cer', 1.0)
        
        # 只处理CER <= threshold的音频
        if cer > cer_threshold:
            return True, None, ""  # 跳过，不算失败
        
        # 检查必要字段
        if not audio_path or not prompt_id or not voiceprint_id:
            return False, f"缺少必要字段: audio_path={bool(audio_path)}, prompt_id={bool(prompt_id)}, voiceprint_id={bool(voiceprint_id)}", ""
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            return False, f"音频文件不存在: {audio_path}", ""
        
        # 获取CER范围目录
        cer_range_dir = get_cer_range_dir(cer)
        
        # 构建目标路径: base_output_dir/cer_range_dir/prompt_id/voiceprint_id.wav
        target_dir = os.path.join(base_output_dir, cer_range_dir, prompt_id)
        
        # 多进程安全的目录创建
        try:
            os.makedirs(target_dir, exist_ok=True)
        except FileExistsError:
            pass
        except Exception as mkdir_err:
            return False, f"创建目录失败 {target_dir}: {mkdir_err}", cer_range_dir
        
        # 处理音频文件（复制或重采样）
        target_audio_path = os.path.join(target_dir, f"{voiceprint_id}.wav")
        try:
            if target_sr is not None:
                # 重采样到目标采样率
                y, sr = librosa.load(audio_path, sr=None)
                if sr != target_sr:
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type='fft')
                    sf.write(target_audio_path, y_resampled, target_sr)
                else:
                    # 采样率相同，直接复制
                    shutil.copy2(audio_path, target_audio_path)
            else:
                # 不重采样，直接复制
                shutil.copy2(audio_path, target_audio_path)
        except Exception as audio_err:
            return False, f"处理音频文件失败 {audio_path} -> {target_audio_path}: {audio_err}", cer_range_dir
        
        # 创建对应的JSON文件
        json_data = {
            'voiceprint_id': voiceprint_id,
            'prompt_id': prompt_id,
            'audio_path': audio_path,
            'groundtruth_text': item.get('groundtruth_text', ''),
            'transcription': item.get('transcription', ''),
            'normalized_groundtruth': item.get('normalized_groundtruth', ''),
            'normalized_transcription': item.get('normalized_transcription', ''),
            'cer': cer,
            'cer_range': cer_range_dir,
            'language': item.get('language', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = os.path.join(target_dir, f"{voiceprint_id}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except Exception as json_err:
            return False, f"写入JSON文件失败 {json_path}: {json_err}", cer_range_dir
        
        return True, None, cer_range_dir
        
    except Exception as e:
        return False, f"处理失败 {item.get('audio_path', 'unknown')}: {type(e).__name__}: {e}", ""


def process_audio_files(results: List[Dict], output_dir: str, 
                       cer_threshold: float = 0.25,
                       num_workers: int = 16,
                       target_sr: Optional[int] = None) -> Dict:
    """
    处理音频文件，按CER分类复制
    
    返回统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选出需要处理的音频（CER <= threshold）
    to_process = [item for item in results if item.get('cer', 1.0) <= cer_threshold]
    
    if not to_process:
        logger.warning(f"没有CER <= {cer_threshold}的音频需要处理")
        return {
            'total': len(results),
            'processed': 0,
            'skipped': len(results),
            'failed': 0,
            'by_range': {}
        }
    
    logger.info(f"准备处理 {len(to_process)} 个音频文件（CER <= {cer_threshold}），使用 {num_workers} 个工作进程")
    
    processed_count = 0
    failed_count = 0
    skipped_count = len(results) - len(to_process)
    failed_errors = []
    range_counts = defaultdict(int)  # 统计每个CER范围的音频数量
    
    # 动态调整进度输出频率
    if len(to_process) > 10000:
        progress_interval = 50
    elif len(to_process) > 1000:
        progress_interval = 20
    else:
        progress_interval = 10
    
    import time
    start_time = time.time()
    last_progress_time = start_time
    heartbeat_interval = 30
    
    # 使用多进程处理
    process_func = partial(_process_single_audio, base_output_dir=output_dir, 
                          cer_threshold=cer_threshold, target_sr=target_sr)
    
    logger.info("开始提交处理任务...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        max_pending = num_workers * 2
        pending_futures = {}
        item_iter = iter(to_process)
        total_submitted = 0
        
        # 先提交第一批任务
        for _ in range(min(max_pending, len(to_process))):
            try:
                item = next(item_iter)
                future = executor.submit(process_func, item)
                pending_futures[future] = item
                total_submitted += 1
            except StopIteration:
                break
        
        logger.info(f"已提交 {total_submitted}/{len(to_process)} 个处理任务，开始处理...")
        
        # 处理完成的任务
        while pending_futures or total_submitted < len(to_process):
            # 提交新任务
            while len(pending_futures) < max_pending and total_submitted < len(to_process):
                try:
                    item = next(item_iter)
                    future = executor.submit(process_func, item)
                    pending_futures[future] = item
                    total_submitted += 1
                except StopIteration:
                    break
            
            if not pending_futures:
                break
            
            # 等待至少一个任务完成
            done, not_done = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
            
            # 处理所有完成的任务
            for future in done:
                item = pending_futures.pop(future)
                
                try:
                    success, error_msg, cer_range_dir = future.result()
                except Exception as e:
                    success = False
                    error_msg = f"任务执行异常: {type(e).__name__}: {e}"
                    cer_range_dir = ""
                    logger.warning(f"处理任务异常: {error_msg}")
                
                if success:
                    processed_count += 1
                    if cer_range_dir:
                        range_counts[cer_range_dir] += 1
                    
                    current_time = time.time()
                    
                    # 定期输出进度
                    if processed_count % progress_interval == 0 or processed_count == len(to_process):
                        elapsed = current_time - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (len(to_process) - processed_count) / rate if rate > 0 else 0
                        logger.info(f"处理进度: {processed_count}/{len(to_process)} ({processed_count/len(to_process)*100:.1f}%) | "
                                  f"速度: {rate:.1f} 文件/秒 | 预计剩余: {remaining/60:.1f} 分钟")
                        last_progress_time = current_time
                    # 心跳
                    elif current_time - last_progress_time >= heartbeat_interval:
                        elapsed = current_time - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"[心跳] 已处理: {processed_count}/{len(to_process)} | 速度: {rate:.1f} 文件/秒 | 运行时间: {elapsed/60:.1f} 分钟")
                        last_progress_time = current_time
                else:
                    failed_count += 1
                    if len(failed_errors) < 10 and error_msg:
                        failed_errors.append(error_msg)
                    if failed_count % 1000 == 0:
                        logger.warning(f"已失败: {failed_count} 个文件")
    
    total_time = time.time() - start_time
    logger.info(f"处理任务完成，总耗时: {total_time/60:.1f} 分钟")
    
    # 输出失败统计
    if failed_count > 0:
        logger.warning(f"\n处理失败统计: {failed_count}/{len(to_process)} 个文件")
        if failed_errors:
            logger.warning("前几个失败的错误信息:")
            for i, err in enumerate(failed_errors, 1):
                logger.warning(f"  {i}. {err}")
    
    return {
        'total': len(results),
        'processed': processed_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'by_range': dict(range_counts)
    }


def generate_statistics(results: List[Dict], cer_threshold: float) -> Dict:
    """生成统计信息"""
    cer_values = [item.get('cer', 1.0) for item in results]
    
    stats = {
        'total': len(results),
        'within_threshold': sum(1 for cer in cer_values if cer <= cer_threshold),
        'exceed_threshold': sum(1 for cer in cer_values if cer > cer_threshold),
        'cer_threshold': cer_threshold,
        'by_range': defaultdict(int)
    }
    
    # 按CER范围统计
    for item in results:
        cer = item.get('cer', 1.0)
        if cer <= cer_threshold:
            cer_range = get_cer_range_dir(cer)
            stats['by_range'][cer_range] += 1
    
    stats['by_range'] = dict(stats['by_range'])
    
    # CER统计
    if cer_values:
        import numpy as np
        stats['cer_stats'] = {
            'mean': float(np.mean(cer_values)),
            'median': float(np.median(cer_values)),
            'std': float(np.std(cer_values)),
            'min': float(np.min(cer_values)),
            'max': float(np.max(cer_values))
        }
    
    return stats


def save_summary(stats: Dict, process_stats: Dict, output_dir: str, 
                asr_result_path: str, cer_threshold: float):
    """保存统计摘要"""
    summary_path = os.path.join(output_dir, "filter_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TTS音频按CER分类筛选结果统计\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("输入文件:\n")
        f.write(f"  ASR结果: {asr_result_path}\n\n")
        
        f.write("筛选阈值:\n")
        f.write(f"  CER阈值: {cer_threshold}\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  总音频数: {stats['total']}\n")
        f.write(f"  CER <= {cer_threshold}: {stats['within_threshold']} ({stats['within_threshold']/stats['total']*100:.2f}%)\n")
        f.write(f"  CER > {cer_threshold}: {stats['exceed_threshold']} ({stats['exceed_threshold']/stats['total']*100:.2f}%)\n\n")
        
        f.write("按CER范围分布（处理结果）:\n")
        for cer_range in sorted(stats['by_range'].keys()):
            count = stats['by_range'][cer_range]
            pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"  {cer_range}: {count} ({pct:.2f}%)\n")
        
        f.write("\n处理统计:\n")
        f.write(f"  成功处理: {process_stats['processed']}\n")
        f.write(f"  处理失败: {process_stats['failed']}\n")
        f.write(f"  跳过（CER超标）: {process_stats['skipped']}\n")
        
        f.write("\n实际复制到各目录的音频数:\n")
        for cer_range in sorted(process_stats['by_range'].keys()):
            count = process_stats['by_range'][cer_range]
            f.write(f"  {cer_range}: {count}\n")
        
        if 'cer_stats' in stats:
            f.write("\nCER统计:\n")
            f.write(f"  平均值: {stats['cer_stats']['mean']:.4f}\n")
            f.write(f"  中位数: {stats['cer_stats']['median']:.4f}\n")
            f.write(f"  标准差: {stats['cer_stats']['std']:.4f}\n")
            f.write(f"  最小值: {stats['cer_stats']['min']:.4f}\n")
            f.write(f"  最大值: {stats['cer_stats']['max']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"统计摘要已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="基于CER值对TTS音频进行分类筛选")
    parser.add_argument("--asr_result", type=str, required=True,
                       help="ASR筛选结果JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（存放分类后的音频和JSON文件）")
    parser.add_argument("--cer_threshold", type=float, default=0.25,
                       help="CER阈值，只处理CER <= 此值的音频（默认: 0.25）")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="并行工作进程数（默认: 16）")
    parser.add_argument("--target_sr", type=int, default=None,
                       help="目标采样率，如果指定则会将所有音频重采样到此采样率（使用librosa resample fft方法）。如果不指定则不重采样，直接复制（默认: None）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入文件
    if not os.path.exists(args.asr_result):
        logger.error(f"ASR结果文件不存在: {args.asr_result}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("TTS音频按CER分类筛选")
    print("=" * 80)
    print(f"ASR结果: {args.asr_result}")
    print(f"输出目录: {args.output_dir}")
    print(f"CER阈值: {args.cer_threshold}")
    print(f"工作进程数: {args.num_workers}")
    if args.target_sr:
        print(f"目标采样率: {args.target_sr} Hz (将重采样)")
    else:
        print(f"目标采样率: 不重采样（直接复制）")
    print("=" * 80 + "\n")
    
    try:
        # 加载ASR结果
        logger.info(f"加载ASR筛选结果: {args.asr_result}")
        with open(args.asr_result, 'r', encoding='utf-8') as f:
            asr_data = json.load(f)
        
        results = asr_data.get('filter_results', [])
        logger.info(f"加载了 {len(results)} 条ASR结果")
        
        # 生成统计信息
        logger.info("生成统计信息...")
        stats = generate_statistics(results, args.cer_threshold)
        
        # 处理音频文件
        logger.info("开始处理音频文件...")
        if args.target_sr:
            logger.info(f"音频将重采样到 {args.target_sr} Hz (使用librosa resample fft方法)")
        process_stats = process_audio_files(
            results, 
            args.output_dir,
            cer_threshold=args.cer_threshold,
            num_workers=args.num_workers,
            target_sr=args.target_sr
        )
        
        # 保存摘要
        logger.info("保存统计摘要...")
        save_summary(stats, process_stats, args.output_dir, 
                    args.asr_result, args.cer_threshold)
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("处理完成统计")
        print("=" * 80)
        print(f"总音频数: {stats['total']}")
        print(f"CER <= {args.cer_threshold}: {stats['within_threshold']} ({stats['within_threshold']/stats['total']*100:.2f}%)")
        print(f"成功处理: {process_stats['processed']}")
        print(f"处理失败: {process_stats['failed']}")
        print("\n按CER范围分布:")
        for cer_range in sorted(process_stats['by_range'].keys()):
            print(f"  {cer_range}: {process_stats['by_range'][cer_range]}")
        print("=" * 80 + "\n")
        
        logger.info(f"完成！结果保存在: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

