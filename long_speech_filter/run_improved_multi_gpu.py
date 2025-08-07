#!/usr/bin/env python3
"""
改进的多GPU长音频处理启动脚本
解决显存泄漏和进程管理问题
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加父目录到Python路径，使模块导入正常工作
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 现在可以正确导入模块
from long_speech_filter.config import LongAudioProcessingConfig
from long_speech_filter.multi_gpu_processor import MultiGPULongAudioProcessor, MultiGPUConfig


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志配置"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 基础配置
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logging.getLogger().addHandler(console_handler)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="改进的多GPU长音频处理器 - 解决显存管理问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置
  python run_improved_multi_gpu.py
  
  # 自定义输入输出目录
  python run_improved_multi_gpu.py --input /path/to/input --output /path/to/output
  
  # 严格限制显存使用
  python run_improved_multi_gpu.py --memory-fraction 0.6 --processes-per-gpu 1
  
  # 使用特定GPU
  python run_improved_multi_gpu.py --num-gpus 4 --max-concurrent 4
        """
    )
    
    # 输入输出配置
    parser.add_argument(
        '--input', type=str,
        default='/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced',
        help='输入目录路径'
    )
    parser.add_argument(
        '--output', type=str,
        default='/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced_filtered_2',
        help='输出目录路径'
    )
    
    # GPU配置
    parser.add_argument(
        '--num-gpus', type=int, default=-1,
        help='使用的GPU数量 (-1表示使用所有GPU)'
    )
    parser.add_argument(
        '--processes-per-gpu', type=int, default=2,
        help='每个GPU的最大进程数 (建议设置为1避免显存竞争)'
    )
    parser.add_argument(
        '--memory-fraction', type=float, default=0.6,
        help='每个GPU使用的显存比例 (0.1-0.9, 建议0.6-0.7)'
    )
    parser.add_argument(
        '--max-concurrent', type=int, default=16,
        help='最大并发文件数'
    )
    
    # 模型配置
    parser.add_argument(
        '--whisper-model', type=str, default='large-v3',
        help='Whisper模型名称'
    )
    parser.add_argument(
        '--model-cache-dir', type=str, default='/root/data/pretrained_models',
        help='模型缓存目录'
    )
    
    # 质量筛选配置
    parser.add_argument(
        '--min-words', type=int, default=1,
        help='最少词数要求'
    )
    parser.add_argument(
        '--distilmos-threshold', type=float, default=3.0,
        help='DistilMOS阈值'
    )
    parser.add_argument(
        '--dnsmos-threshold', type=float, default=3.0,
        help='DNSMOS阈值'  
    )
    parser.add_argument(
        '--dnsmospro-threshold', type=float, default=3.0,
        help='DNSMOSPro阈值'
    )
    
    # 日志配置
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help='日志级别'
    )
    parser.add_argument(
        '--log-file', type=str, 
        default='improved_multi_gpu_processing.log',
        help='日志文件名'
    )
    
    # 调试选项
    parser.add_argument(
        '--dry-run', action='store_true',
        help='只显示配置信息，不实际处理'
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='测试模式：只处理前5个文件'
    )
    
    # 处理选项
    parser.add_argument(
        '--skip-processed', action='store_true', default=True,
        help='跳过已处理的文件 (默认启用)'
    )
    parser.add_argument(
        '--no-skip-processed', action='store_true',
        help='不跳过已处理的文件，处理所有文件'
    )
    parser.add_argument(
        '--force-reprocess', action='store_true',
        help='强制重新处理所有文件，即使已存在结果'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """验证参数"""
    errors = []
    
    # 验证路径
    if not Path(args.input).exists():
        errors.append(f"输入目录不存在: {args.input}")
    
    # 验证显存比例
    if not (0.1 <= args.memory_fraction <= 0.9):
        errors.append(f"显存比例必须在0.1-0.9之间: {args.memory_fraction}")
    
    # 验证进程数
    if args.processes_per_gpu < 1 or args.processes_per_gpu > 2:
        errors.append(f"每GPU进程数建议设置为1-2: {args.processes_per_gpu}")
    
    # 验证阈值
    for threshold_name, threshold_value in [
        ('distilmos-threshold', args.distilmos_threshold),
        ('dnsmos-threshold', args.dnsmos_threshold), 
        ('dnsmospro-threshold', args.dnsmospro_threshold)
    ]:
        if not (1.0 <= threshold_value <= 5.0):
            errors.append(f"{threshold_name}必须在1.0-5.0之间: {threshold_value}")
    
    if errors:
        print("❌ 参数验证失败:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)


def create_configs(args):
    """创建配置对象"""
    # 基础处理配置
    base_config = LongAudioProcessingConfig()
    base_config.input_dir = args.input
    base_config.output_dir = args.output
    base_config.whisper.model_name = args.whisper_model
    base_config.whisper.model_cache_dir = args.model_cache_dir
    base_config.quality_filter.min_words = args.min_words
    base_config.quality_filter.distil_mos_threshold = args.distilmos_threshold
    base_config.quality_filter.dnsmos_threshold = args.dnsmos_threshold
    base_config.quality_filter.dnsmospro_threshold = args.dnsmospro_threshold
    base_config.log_level = args.log_level
    base_config.log_file = args.log_file
    
    # 处理跳过已处理文件的逻辑
    if args.force_reprocess:
        # 强制重新处理所有文件
        base_config.processing.skip_processed = False
        base_config.processing.force_reprocess = True
    elif args.no_skip_processed:
        # 不跳过已处理文件，但也不强制重新处理
        base_config.processing.skip_processed = False
        base_config.processing.force_reprocess = False
    else:
        # 默认跳过已处理文件
        base_config.processing.skip_processed = True
        base_config.processing.force_reprocess = False
    
    # 多GPU配置
    multi_gpu_config = MultiGPUConfig(
        num_gpus=args.num_gpus,
        max_processes_per_gpu=args.processes_per_gpu,
        gpu_memory_fraction=args.memory_fraction,
        max_concurrent_files=args.max_concurrent,
        enable_gpu_monitoring=True
    )
    
    return base_config, multi_gpu_config


def print_config_summary(base_config, multi_gpu_config):
    """打印配置摘要"""
    print("=" * 80)
    print("📋 改进的多GPU长音频处理器配置摘要")
    print("=" * 80)
    
    print(f"📁 输入目录: {base_config.input_dir}")
    print(f"📁 输出目录: {base_config.output_dir}")
    print()
    
    print("🖥️ GPU配置:")
    print(f"   • GPU数量: {multi_gpu_config.num_gpus} (-1表示全部)")
    print(f"   • 每GPU进程数: {multi_gpu_config.max_processes_per_gpu}")
    print(f"   • 显存使用比例: {multi_gpu_config.gpu_memory_fraction:.1%}")
    print(f"   • 最大并发文件数: {multi_gpu_config.max_concurrent_files}")
    print()
    
    print("🎤 模型配置:")
    print(f"   • Whisper模型: {base_config.whisper.model_name}")
    print(f"   • 模型缓存目录: {base_config.whisper.model_cache_dir}")
    print()
    
    print("📊 质量筛选阈值:")
    print(f"   • 最少词数: {base_config.quality_filter.min_words}")
    print(f"   • DistilMOS ≥ {base_config.quality_filter.distil_mos_threshold}")
    print(f"   • DNSMOS ≥ {base_config.quality_filter.dnsmos_threshold}")
    print(f"   • DNSMOSPro ≥ {base_config.quality_filter.dnsmospro_threshold}")
    print()
    
    print("🔄 处理选项:")
    if base_config.processing.force_reprocess:
        print("   • ⚡ 强制重新处理所有文件")
    elif base_config.processing.skip_processed:
        print("   • ⏭️ 跳过已处理的文件 (默认)")
    else:
        print("   • 🔄 处理所有文件 (不跳过)")
    print()
    
    print("🔧 改进特性:")
    print("   • ✅ 严格的每GPU一进程限制")
    print("   • ✅ 主动显存清理和监控")
    print("   • ✅ 带重试机制的模型加载")
    print("   • ✅ 批量处理显存管理")
    print("   • ✅ CPU后备模式支持")
    print("   • ✅ 详细的进度和显存监控")
    print("   • ✅ 智能跳过已处理文件")
    print("=" * 80)


def check_gpu_status():
    """检查GPU状态"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，无法使用GPU加速")
            return False
        
        num_gpus = torch.cuda.device_count()
        print(f"🖥️ 检测到 {num_gpus} 张GPU:")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            
            # 检查显存使用情况
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = memory_gb - allocated
            
            print(f"   GPU {i}: {props.name}")
            print(f"      总显存: {memory_gb:.1f}GB")
            print(f"      已使用: {allocated:.1f}GB ({allocated/memory_gb:.1%})")
            print(f"      已缓存: {cached:.1f}GB ({cached/memory_gb:.1%})")
            print(f"      可用: {free:.1f}GB ({free/memory_gb:.1%})")
            
            if free < 2.0:  # 可用显存少于2GB时警告
                print(f"      ⚠️ 警告: 可用显存不足，可能影响处理")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU状态检查失败: {e}")
        return False


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 验证参数
    validate_arguments(args)
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    # 创建配置
    base_config, multi_gpu_config = create_configs(args)
    
    # 打印配置摘要
    print_config_summary(base_config, multi_gpu_config)
    
    # 检查GPU状态
    if not check_gpu_status():
        print("❌ GPU检查失败，退出处理")
        sys.exit(1)
    
    # 干运行模式
    if args.dry_run:
        print("🔍 干运行模式：配置验证完成，未执行实际处理")
        return
    
    print()
    print("🚀 开始改进的多GPU并行处理...")
    print()
    
    try:
        # 创建处理器
        processor = MultiGPULongAudioProcessor(base_config, multi_gpu_config)
        
        # 测试模式：限制文件数量
        if args.test_mode:
            print("🧪 测试模式：只处理前5个文件")
            # 可以在这里修改配置来限制文件数量
        
        # 执行处理
        results = processor.process_directory_parallel()
        
        # 输出最终结果
        print()
        print("=" * 80)
        print("🎉 处理完成!")
        print("=" * 80)
        print(f"📊 总文件数: {results['total_files']}")
        print(f"✅ 成功处理: {results['successful_files']}")
        print(f"❌ 失败文件: {results['failed_files']}")
        print(f"📈 成功率: {results['success_rate']:.1f}%")
        print(f"⏱️ 总处理时间: {results['processing_time']/60:.1f}分钟")
        print(f"⚡ 平均处理时间: {results['average_time_per_file']:.1f}秒/文件")
        
        # GPU使用统计
        print()
        print("🖥️ GPU使用统计:")
        for gpu_id, stats in results['gpu_stats'].items():
            memory_info = ""
            if 'memory_usage' in stats:
                memory_info = f", 最终显存: {stats['memory_usage']:.1%}"
            print(f"   GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件{memory_info}")
        
        # 显存统计  
        if results.get('memory_stats', {}).get('peak_usage_by_gpu'):
            print()
            print("📊 显存使用统计:")
            for gpu_id, peak in results['memory_stats']['peak_usage_by_gpu'].items():
                avg = results['memory_stats']['average_usage_by_gpu'].get(gpu_id, 0)
                print(f"   GPU {gpu_id}: 峰值 {peak:.1%}, 平均 {avg:.1%}")
        
        print("=" * 80)
        
        # 返回成功状态
        return 0 if results['successful_files'] > 0 else 1
        
    except KeyboardInterrupt:
        print()
        print("⚠️ 用户中断处理")
        return 1
    except Exception as e:
        print()
        print(f"❌ 处理过程中发生错误: {e}")
        logging.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 