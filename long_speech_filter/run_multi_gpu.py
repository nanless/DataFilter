#!/usr/bin/env python3
"""
多GPU长音频处理启动脚本
使用8张GPU并行处理长音频文件
"""

# ⚠️ 重要：必须在任何其他导入之前设置multiprocessing启动方法
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
        print("🔧 设置multiprocessing启动方法为spawn以支持CUDA")
    except RuntimeError as e:
        print(f"⚠️ 无法设置spawn方法: {e}")

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from multi_gpu_processor import MultiGPULongAudioProcessor, MultiGPUConfig
from config import LongAudioProcessingConfig


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description="多GPU长音频处理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理指定目录的音频文件，使用所有GPU
  python run_multi_gpu.py \\
    --input /path/to/input/audio \\
    --output /path/to/output/filtered

  # 只使用前4张GPU，设置质量阈值
  python run_multi_gpu.py \\
    --input /path/to/input/audio \\
    --output /path/to/output/filtered \\
    --num-gpus 4 \\
    --distil-mos-threshold 2.5 \\
    --dnsmos-threshold 2.5

  # 调试模式，详细日志
  python run_multi_gpu.py \\
    --input /path/to/input/audio \\
    --output /path/to/output/filtered \\
    --log-level DEBUG \\
    --log-file multi_gpu_processing.log
        """
    )
    
    # 基本参数
    parser.add_argument('--input', type=str, 
                       default="/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced",
                       help='输入音频目录')
    parser.add_argument('--output', type=str,
                       default="/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced_filtered", 
                       help='输出目录')
    
    # 多GPU配置
    parser.add_argument('--num-gpus', type=int, default=-1,
                       help='使用的GPU数量（-1表示使用所有GPU）')
    parser.add_argument('--max-concurrent', type=int, default=8,
                       help='最大并发处理文件数')
    
    # 质量筛选参数
    parser.add_argument('--distil-mos-threshold', type=float, default=3.0,
                       help='DistilMOS质量阈值')
    parser.add_argument('--dnsmos-threshold', type=float, default=3.0,
                       help='DNSMOS质量阈值')
    parser.add_argument('--dnsmospro-threshold', type=float, default=3.0,
                       help='DNSMOSPro质量阈值')
    parser.add_argument('--min-words', type=int, default=1,
                       help='最少词数要求')
    
    # PyAnnote配置
    parser.add_argument('--use-local-models', action='store_true', default=True,
                       help='使用本地PyAnnote模型')
    parser.add_argument('--no-local-models', action='store_false', dest='use_local_models',
                       help='不使用本地模型，从HuggingFace Hub下载')
    parser.add_argument('--local-model-path', type=str, default="pyannote",
                       help='本地模型路径')
    
    # Whisper配置
    parser.add_argument('--whisper-model', type=str, default="large-v3",
                       help='Whisper模型名称')
    parser.add_argument('--whisper-language', type=str, default=None,
                       help='Whisper语言（None表示自动检测）')
    
    # 日志配置
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    parser.add_argument('--log-file', type=str, default='multi_gpu_processing.log',
                       help='日志文件路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # 显示配置信息
    print("=" * 60)
    print("🚀 多GPU长音频处理系统")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"GPU数量: {args.num_gpus if args.num_gpus > 0 else '全部'}")
    print(f"最大并发: {args.max_concurrent}")
    print(f"质量阈值: DistilMOS≥{args.distil_mos_threshold}, DNSMOS≥{args.dnsmos_threshold}, DNSMOSPro≥{args.dnsmospro_threshold}")
    print(f"最少词数: {args.min_words}")
    print(f"Whisper模型: {args.whisper_model}")
    print(f"PyAnnote本地模型: {'是' if args.use_local_models else '否'}")
    print("=" * 60)
    
    try:
        # 验证输入目录
        if not Path(args.input).exists():
            logger.error(f"输入目录不存在: {args.input}")
            return 1
        
        # 创建输出目录
        Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # 创建基础配置
        base_config = LongAudioProcessingConfig()
        base_config.input_dir = args.input
        base_config.output_dir = args.output
        
        # 更新质量筛选配置
        base_config.quality_filter.distil_mos_threshold = args.distil_mos_threshold
        base_config.quality_filter.dnsmos_threshold = args.dnsmos_threshold
        base_config.quality_filter.dnsmospro_threshold = args.dnsmospro_threshold
        base_config.quality_filter.min_words = args.min_words
        
        # 更新Whisper配置
        base_config.whisper.model_name = args.whisper_model
        base_config.whisper.language = args.whisper_language
        
        # 更新PyAnnote配置
        base_config.speaker_diarization.use_local_models = args.use_local_models
        base_config.speaker_diarization.local_model_path = args.local_model_path
        
        # 创建多GPU配置
        multi_gpu_config = MultiGPUConfig(
            num_gpus=args.num_gpus,
            max_concurrent_files=args.max_concurrent
        )
        
        # 创建多GPU处理器
        logger.info("正在初始化多GPU处理器...")
        processor = MultiGPULongAudioProcessor(base_config, multi_gpu_config)
        
        # 开始处理
        logger.info("开始多GPU并行处理...")
        print(f"\n⏳ 正在处理音频文件，请耐心等待...")
        
        results = processor.process_directory_parallel()
        
        # 输出最终结果
        print(f"\n" + "=" * 60)
        print("🎯 处理完成！")
        print("=" * 60)
        print(f"📁 总文件数: {results['total_files']}")
        print(f"✅ 成功处理: {results['successful_files']}")
        print(f"❌ 失败文件: {results['failed_files']}")
        print(f"📊 成功率: {results['success_rate']:.1f}%")
        print(f"⏱️  总处理时间: {results['processing_time']/60:.1f}分钟")
        
        if results['total_files'] > 0:
            print(f"⚡ 平均每文件: {results['processing_time']/results['total_files']:.1f}秒")
        
        print(f"\n🖥️  GPU使用统计:")
        for gpu_id, stats in results['gpu_stats'].items():
            print(f"  GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件")
        
        if results['failed_files'] > 0:
            print(f"\n⚠️  失败文件详情请查看日志: {args.log_file}")
        
        print(f"\n📂 输出目录: {args.output}")
        print("=" * 60)
        
        # 生成处理报告
        report_file = Path(args.output) / "processing_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"📋 详细报告已保存: {report_file}")
        
        return 0 if results['successful_files'] > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        print(f"\n⏹️  处理已中断")
        return 130
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        print(f"\n❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 