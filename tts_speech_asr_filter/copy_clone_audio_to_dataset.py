#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将filtered_speech目录下的TTS克隆音频拷贝回原数据集目录结构

功能：
1. 从多个数据集的utt2spk文件加载prompt_id到speaker_id的映射
2. 扫描filtered_speech目录下的音频文件
3. 根据prompt_id找到对应的数据集和说话人
4. 将音频拷贝到目标数据集目录对应的说话人目录下
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量用于多进程共享（避免序列化传递）
_global_prompt_to_speaker = None

def _init_worker(prompt_to_speaker):
    """初始化子进程，加载映射到全局变量"""
    global _global_prompt_to_speaker
    _global_prompt_to_speaker = prompt_to_speaker


# 数据集配置
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
        'name': 'king-asr',  # 包含king-asr-612和king-asr-725
        'target_dir': None,  # 需要根据speaker_id判断
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/utt2spk',
    },
    {
        'name': 'speechocean762',
        'target_dir': 'speechocean762',
        'utt2spk_path': '/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/utt2spk',
    }
]


class Utt2SpkMapper:
    """加载和管理utt2spk映射关系"""
    
    def __init__(self, dataset_config: List[Dict]):
        self.dataset_config = dataset_config
        # prompt_id -> (dataset_name, speaker_id)
        self.prompt_to_speaker = {}
        # 统计信息
        self.stats = defaultdict(int)
        
    def load_all_mappings(self):
        """加载所有数据集的utt2spk映射"""
        logger.info("开始加载utt2spk映射...")
        
        for config in self.dataset_config:
            dataset_name = config['name']
            utt2spk_path = config['utt2spk_path']
            
            if not os.path.exists(utt2spk_path):
                logger.warning(f"utt2spk文件不存在: {utt2spk_path}")
                continue
            
            logger.info(f"加载 {dataset_name}: {utt2spk_path}")
            count = self._load_single_mapping(utt2spk_path, dataset_name)
            self.stats[f'{dataset_name}_mappings'] = count
        
        logger.info(f"总共加载 {len(self.prompt_to_speaker)} 个prompt_id映射")
        return len(self.prompt_to_speaker)
    
    def _load_single_mapping(self, utt2spk_path: str, dataset_name: str) -> int:
        """加载单个数据集的utt2spk映射"""
        count = 0
        with open(utt2spk_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    continue
                
                prompt_id, speaker_id = parts
                self.prompt_to_speaker[prompt_id] = (dataset_name, speaker_id)
                count += 1
        
        return count
    
    def get_speaker_info(self, prompt_id: str) -> Optional[Tuple[str, str]]:
        """获取prompt_id对应的数据集和说话人信息"""
        return self.prompt_to_speaker.get(prompt_id)
    
    def determine_target_dir(self, dataset_name: str, speaker_id: str) -> Optional[str]:
        """确定目标子目录名称"""
        # King-ASR需要根据speaker_id判断是612还是725
        if dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                return 'kingasr612'
            elif 'King-ASR-725' in speaker_id:
                return 'king-asr-725'
            else:
                logger.warning(f"无法判断King-ASR子集: {speaker_id}")
                return None
        
        # 其他数据集直接从配置获取
        for config in self.dataset_config:
            if config['name'] == dataset_name:
                return config['target_dir']
        
        return None
    
    def transform_speaker_id(self, dataset_name: str, speaker_id: str, target_subdir: str) -> str:
        """
        转换speaker_id为目标目录中的说话人目录名
        
        规则：
        - childmandarin: 001 -> childmandarin_001
        - chineseenglishchildren: G0001 -> chineseenglishchildren_G0001
        - kingasr612: King-ASR-612_SPEAKER0008 -> kingasr612_0008
        - king-asr-725: King-ASR-725_SPEAKER1001 -> king-asr-725_SPEAKER1001
        - speechocean762: speechocean762_test_0003 -> speechocean762_0003
                         speechocean762_train_9646 -> speechocean762_9646
        """
        if dataset_name == 'childmandarin':
            # 001 -> childmandarin_001
            return f"childmandarin_{speaker_id}"
        
        elif dataset_name == 'chineseenglishchildren':
            # G0001 -> chineseenglishchildren_G0001
            return f"chineseenglishchildren_{speaker_id}"
        
        elif dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                # King-ASR-612_SPEAKER0008 -> kingasr612_0008
                # 提取SPEAKER后面的数字
                if 'SPEAKER' in speaker_id:
                    speaker_num = speaker_id.split('SPEAKER')[-1]
                    return f"kingasr612_{speaker_num}"
                else:
                    logger.warning(f"无法解析King-ASR-612 speaker_id: {speaker_id}")
                    return f"kingasr612_{speaker_id}"
            
            elif 'King-ASR-725' in speaker_id:
                # King-ASR-725_SPEAKER1001 -> king-asr-725_SPEAKER1001
                if 'SPEAKER' in speaker_id:
                    speaker_num = speaker_id.split('SPEAKER')[-1]
                    return f"king-asr-725_SPEAKER{speaker_num}"
                else:
                    logger.warning(f"无法解析King-ASR-725 speaker_id: {speaker_id}")
                    return f"king-asr-725_{speaker_id}"
        
        elif dataset_name == 'speechocean762':
            # speechocean762_test_0003 -> speechocean762_0003
            # speechocean762_train_9646 -> speechocean762_9646
            # 去掉"test_"或"train_"前缀
            if 'test_' in speaker_id:
                speaker_num = speaker_id.split('test_')[-1]
                return f"speechocean762_{speaker_num}"
            elif 'train_' in speaker_id:
                speaker_num = speaker_id.split('train_')[-1]
                return f"speechocean762_{speaker_num}"
            else:
                # 如果speaker_id已经是正确格式或只是数字，确保有前缀
                if speaker_id.startswith('speechocean762_'):
                    return speaker_id
                else:
                    return f"speechocean762_{speaker_id}"
        
        # 默认：直接添加数据集前缀
        return f"{target_subdir}_{speaker_id}"


def scan_filtered_audio(filtered_speech_dirs: List[str]) -> List[Tuple[str, str, str]]:
    """
    扫描filtered_speech目录，收集所有音频文件（优化版：使用glob）
    
    返回: [(prompt_id, voiceprint_id, audio_path), ...]
    """
    import glob
    audio_files = []
    
    for filtered_dir in filtered_speech_dirs:
        audio_dir = os.path.join(filtered_dir, 'audio')
        
        if not os.path.exists(audio_dir):
            logger.warning(f"音频目录不存在: {audio_dir}")
            continue
        
        logger.info(f"扫描目录: {audio_dir}")
        
        # 使用glob更快速地扫描所有.wav文件
        pattern = os.path.join(audio_dir, '*', '*.wav')
        for audio_path in glob.iglob(pattern):
            # 从路径提取prompt_id和voiceprint_id
            # 路径格式: .../audio/<prompt_id>/<voiceprint_id>.wav
            parts = audio_path.split(os.sep)
            filename = parts[-1]
            prompt_id = parts[-2]
            
            # voiceprint_id是去掉.wav后缀的文件名
            voiceprint_id = filename[:-4]
            
            audio_files.append((prompt_id, voiceprint_id, audio_path))
    
    logger.info(f"总共找到 {len(audio_files)} 个音频文件")
    return audio_files


def _copy_single_audio_with_mapping(item: Tuple[str, str, str], 
                                    output_base_dir: str,
                                    dry_run: bool = False,
                                    use_hardlink: bool = False) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    复制单个音频文件到目标目录（用于多进程）
    优化：从全局变量读取映射，完全避免序列化开销
    
    返回: (success, error_msg, info_dict)
    """
    global _global_prompt_to_speaker
    
    try:
        prompt_id, voiceprint_id, audio_path = item
        
        # 从全局变量查找对应的数据集和说话人
        speaker_info = _global_prompt_to_speaker.get(prompt_id)
        if not speaker_info:
            return False, f"未找到映射: prompt_id={prompt_id}", None
        
        dataset_name, speaker_id = speaker_info
        
        # 确定目标子目录（内联逻辑，避免调用mapper方法）
        if dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                target_subdir = 'kingasr612'
            elif 'King-ASR-725' in speaker_id:
                target_subdir = 'king-asr-725'
            else:
                return False, f"无法判断King-ASR子集: {speaker_id}", None
        else:
            # 其他数据集使用固定映射
            dataset_to_dir = {
                'childmandarin': 'childmandarin',
                'chineseenglishchildren': 'chineseenglishchildren',
                'speechocean762': 'speechocean762'
            }
            target_subdir = dataset_to_dir.get(dataset_name)
            if not target_subdir:
                return False, f"未知数据集: {dataset_name}", None
        
        # 转换speaker_id为目标目录中的说话人目录名（内联逻辑）
        if dataset_name == 'childmandarin':
            target_speaker_dirname = f"childmandarin_{speaker_id}"
        elif dataset_name == 'chineseenglishchildren':
            target_speaker_dirname = f"chineseenglishchildren_{speaker_id}"
        elif dataset_name == 'king-asr':
            if 'King-ASR-612' in speaker_id:
                if 'SPEAKER' in speaker_id:
                    speaker_num = speaker_id.split('SPEAKER')[-1]
                    target_speaker_dirname = f"kingasr612_{speaker_num}"
                else:
                    target_speaker_dirname = f"kingasr612_{speaker_id}"
            elif 'King-ASR-725' in speaker_id:
                if 'SPEAKER' in speaker_id:
                    speaker_num = speaker_id.split('SPEAKER')[-1]
                    target_speaker_dirname = f"king-asr-725_SPEAKER{speaker_num}"
                else:
                    target_speaker_dirname = f"king-asr-725_{speaker_id}"
            else:
                return False, f"无法解析speaker_id: {speaker_id}", None
        elif dataset_name == 'speechocean762':
            if 'test_' in speaker_id:
                speaker_num = speaker_id.split('test_')[-1]
                target_speaker_dirname = f"speechocean762_{speaker_num}"
            elif 'train_' in speaker_id:
                speaker_num = speaker_id.split('train_')[-1]
                target_speaker_dirname = f"speechocean762_{speaker_num}"
            elif speaker_id.startswith('speechocean762_'):
                target_speaker_dirname = speaker_id
            else:
                target_speaker_dirname = f"speechocean762_{speaker_id}"
        else:
            target_speaker_dirname = f"{target_subdir}_{speaker_id}"
        
        # 构建目标路径: output_base_dir/<dataset_subdir>/<target_speaker_dirname>/<voiceprint_id>.wav
        target_speaker_dir = os.path.join(output_base_dir, target_subdir, target_speaker_dirname)
        target_path = os.path.join(target_speaker_dir, f"{voiceprint_id}.wav")
        
        # 检查源文件是否存在
        if not os.path.exists(audio_path):
            return False, f"源文件不存在: {audio_path}", None
        
        # Dry run模式只检查不复制
        if dry_run:
            info = {
                'prompt_id': prompt_id,
                'voiceprint_id': voiceprint_id,
                'source': audio_path,
                'target': target_path,
                'dataset': target_subdir,
                'speaker_id': target_speaker_dirname  # 使用转换后的说话人目录名
            }
            return True, None, info
        
        # 创建目标目录（多进程安全）
        try:
            os.makedirs(target_speaker_dir, exist_ok=True)
        except FileExistsError:
            pass
        except Exception as e:
            return False, f"创建目录失败 {target_speaker_dir}: {e}", None
        
        # 复制或链接文件
        try:
            if use_hardlink:
                # 使用硬链接（更快，但不占用额外空间）
                try:
                    os.link(audio_path, target_path)
                except OSError:
                    # 如果硬链接失败（可能跨文件系统），fallback到复制
                    shutil.copy2(audio_path, target_path)
            else:
                # 常规复制
                shutil.copy2(audio_path, target_path)
        except Exception as e:
            return False, f"复制文件失败 {audio_path} -> {target_path}: {e}", None
        
        # 返回成功信息
        info = {
            'prompt_id': prompt_id,
            'voiceprint_id': voiceprint_id,
            'source': audio_path,
            'target': target_path,
            'dataset': target_subdir,
            'speaker_id': target_speaker_dirname  # 使用转换后的说话人目录名
        }
        return True, None, info
        
    except Exception as e:
        return False, f"处理失败: {type(e).__name__}: {e}", None


def copy_audio_files(audio_files: List[Tuple[str, str, str]],
                    mapper: Utt2SpkMapper,
                    output_base_dir: str,
                    num_workers: int = 8,
                    dry_run: bool = False,
                    print_interval: int = 100,
                    use_hardlink: bool = False) -> Tuple[int, int, List[Dict]]:
    """
    并行复制音频文件到目标目录
    
    返回: (success_count, failed_count, copy_records)
    """
    if not audio_files:
        logger.warning("没有音频文件需要复制")
        return 0, 0, []
    
    logger.info(f"准备{'模拟' if dry_run else ''}复制 {len(audio_files)} 个音频文件，使用 {num_workers} 个工作进程")
    if dry_run:
        logger.info(f"dry_run模式：每隔 {print_interval} 条音频打印一个复制示例")
    if use_hardlink and not dry_run:
        logger.info(f"使用硬链接模式（更快，节省空间）")
    
    success_count = 0
    failed_count = 0
    copy_records = []
    failed_errors = []
    
    # 数据集统计
    dataset_stats = defaultdict(int)
    
    # 使用多进程复制（通过初始化器传递映射，完全避免序列化）
    logger.info("正在准备任务...")
    copy_func = partial(_copy_single_audio_with_mapping,
                       output_base_dir=output_base_dir,
                       dry_run=dry_run,
                       use_hardlink=use_hardlink)
    
    logger.info("正在初始化进程池...")
    
    # 使用multiprocessing.Pool.imap_unordered（更快，不需要创建大量Future对象）
    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(mapper.prompt_to_speaker,)) as pool:
        
        logger.info(f"进程池已初始化，开始处理 {len(audio_files)} 个任务...")
        
        # 使用imap_unordered按需处理结果（不需要一次性提交所有任务）
        # chunksize控制每次发送给worker的任务数
        results = pool.imap_unordered(copy_func, audio_files, chunksize=100)
        
        # 处理结果
        for idx, (success, error_msg, info) in enumerate(results, 1):
            if success and info:
                success_count += 1
                copy_records.append(info)
                dataset_stats[info['dataset']] += 1
                
                # dry_run模式下，每隔print_interval条打印一个示例
                if dry_run and success_count % print_interval == 0:
                    logger.info(f"\n[示例 #{success_count}]")
                    logger.info(f"  Prompt ID:     {info['prompt_id']}")
                    logger.info(f"  Voiceprint ID: {info['voiceprint_id']}")
                    logger.info(f"  数据集:        {info['dataset']}")
                    logger.info(f"  说话人:        {info['speaker_id']}")
                    logger.info(f"  源文件:        {info['source']}")
                    logger.info(f"  目标文件:      {info['target']}\n")
                
                # 优化进度输出频率：每100条或每1%输出一次（更频繁的反馈）
                progress_interval = min(100, max(10, len(audio_files) // 100))
                if success_count % progress_interval == 0 or success_count == len(audio_files):
                    elapsed = datetime.now()
                    logger.info(f"{'模拟' if dry_run else ''}复制进度: {success_count}/{len(audio_files)} ({success_count/len(audio_files)*100:.1f}%)")
            else:
                failed_count += 1
                if len(failed_errors) < 10 and error_msg:
                    failed_errors.append(error_msg)
                
                if failed_count % 1000 == 0:
                    logger.warning(f"已失败: {failed_count} 个文件")
    
    # 输出统计信息
    logger.info(f"\n{'模拟' if dry_run else ''}复制完成统计:")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"\n各数据集分布:")
    for dataset, count in sorted(dataset_stats.items()):
        logger.info(f"  {dataset}: {count}")
    
    if failed_count > 0 and failed_errors:
        logger.warning(f"\n前几个失败的错误信息:")
        for i, err in enumerate(failed_errors, 1):
            logger.warning(f"  {i}. {err}")
    
    return success_count, failed_count, copy_records


def save_copy_report(copy_records: List[Dict], stats: Dict, output_dir: str):
    """保存复制报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细记录JSON
    report_json_path = os.path.join(output_dir, "copy_report.json")
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'copy_records': copy_records
    }
    
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"详细报告已保存: {report_json_path}")
    
    # 保存复制列表（source -> target）
    copy_list_path = os.path.join(output_dir, "copy_list.txt")
    with open(copy_list_path, 'w', encoding='utf-8') as f:
        for record in copy_records:
            f.write(f"{record['source']}\t{record['target']}\n")
    logger.info(f"复制列表已保存: {copy_list_path}")
    
    # 保存按数据集分组的统计
    summary_path = os.path.join(output_dir, "copy_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TTS克隆音频复制到数据集目录 - 统计报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  总音频数: {stats['total_files']}\n")
        f.write(f"  成功复制: {stats['success_count']}\n")
        f.write(f"  失败: {stats['failed_count']}\n")
        f.write(f"  成功率: {stats['success_rate']:.2f}%\n\n")
        
        f.write("各数据集分布:\n")
        for dataset, count in sorted(stats.get('dataset_distribution', {}).items()):
            pct = count / stats['success_count'] * 100 if stats['success_count'] > 0 else 0
            f.write(f"  {dataset}: {count} ({pct:.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"统计摘要已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将filtered_speech目录下的TTS克隆音频拷贝回原数据集目录结构"
    )
    parser.add_argument(
        "--source_dirs", 
        type=str, 
        nargs='+',
        required=True,
        help="源filtered_speech目录列表"
    )
    parser.add_argument(
        "--output_base_dir", 
        type=str, 
        required=True,
        help="目标基础目录（包含各数据集子目录）"
    )
    parser.add_argument(
        "--report_dir", 
        type=str, 
        default=None,
        help="报告输出目录（默认：output_base_dir/copy_reports）"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=32,
        help="并行工作进程数（默认: 32，建议根据CPU核心数调整）"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="模拟运行，不实际复制文件"
    )
    parser.add_argument(
        "--print_interval", 
        type=int, 
        default=100,
        help="dry_run模式下，每隔多少条音频打印一个复制示例（默认: 100）"
    )
    parser.add_argument(
        "--use_hardlink", 
        action="store_true",
        help="使用硬链接代替复制（更快，节省空间，但源文件和目标文件共享数据）"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="详细日志"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置报告目录
    if args.report_dir is None:
        args.report_dir = os.path.join(args.output_base_dir, 'copy_reports')
    
    # 检查源目录
    for source_dir in args.source_dirs:
        if not os.path.exists(source_dir):
            logger.error(f"源目录不存在: {source_dir}")
            return 1
    
    # 检查目标目录
    if not os.path.exists(args.output_base_dir):
        logger.error(f"目标基础目录不存在: {args.output_base_dir}")
        return 1
    
    print("\n" + "=" * 80)
    print("TTS克隆音频复制到数据集目录")
    print("=" * 80)
    print(f"源目录: {', '.join(args.source_dirs)}")
    print(f"目标目录: {args.output_base_dir}")
    print(f"报告目录: {args.report_dir}")
    print(f"工作进程: {args.num_workers}")
    print(f"模拟运行: {'是' if args.dry_run else '否'}")
    print("=" * 80 + "\n")
    
    try:
        # 加载utt2spk映射
        mapper = Utt2SpkMapper(DATASET_CONFIG)
        mapping_count = mapper.load_all_mappings()
        
        if mapping_count == 0:
            logger.error("未加载到任何utt2spk映射")
            return 1
        
        # 扫描源目录
        logger.info("扫描源目录...")
        audio_files = scan_filtered_audio(args.source_dirs)
        
        if not audio_files:
            logger.error("未找到任何音频文件")
            return 1
        
        # 复制音频文件
        logger.info(f"{'模拟' if args.dry_run else '开始'}复制音频文件...")
        success_count, failed_count, copy_records = copy_audio_files(
            audio_files,
            mapper,
            args.output_base_dir,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            print_interval=args.print_interval,
            use_hardlink=args.use_hardlink
        )
        
        # 统计信息
        dataset_distribution = defaultdict(int)
        for record in copy_records:
            dataset_distribution[record['dataset']] += 1
        
        stats = {
            'total_files': len(audio_files),
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / len(audio_files) * 100 if len(audio_files) > 0 else 0,
            'dataset_distribution': dict(dataset_distribution)
        }
        
        # 保存报告
        logger.info("保存复制报告...")
        save_copy_report(copy_records, stats, args.report_dir)
        
        print("\n" + "=" * 80)
        print("复制完成")
        print("=" * 80)
        print(f"总音频数:   {stats['total_files']}")
        print(f"成功复制:   {stats['success_count']} ({stats['success_rate']:.2f}%)")
        print(f"失败:       {stats['failed_count']}")
        print(f"报告目录:   {args.report_dir}")
        print("=" * 80 + "\n")
        
        return 0 if failed_count == 0 else 2
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

