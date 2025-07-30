#!/usr/bin/env python3
"""
长音频处理启动脚本

用法:
    python run_processing.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--config CONFIG_FILE]

参数:
    --input: 输入目录，默认为配置文件中的路径
    --output: 输出目录，默认为配置文件中的路径  
    --config: 配置文件路径（可选）
    --workers: 并行处理的工作线程数，默认为4
    --log-level: 日志级别（DEBUG, INFO, WARNING, ERROR），默认为INFO
"""

import argparse
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from config import LongAudioProcessingConfig
from long_audio_processor import LongAudioProcessor

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="长音频处理系统 - 说话人分离和质量筛选",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 使用默认配置
    python run_processing.py
    
    # 指定输入输出目录
    python run_processing.py --input /path/to/input --output /path/to/output
    
    # 使用4个工作线程并设置详细日志
    python run_processing.py --workers 4 --log-level DEBUG
    
    # 自定义质量阈值
    python run_processing.py --distilmos-threshold 3.5 --dnsmos-threshold 3.0
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str,
        help='输入目录路径，包含待处理的长音频文件'
    )
    
    parser.add_argument(
        '--output',
        type=str, 
        help='输出目录路径，用于保存筛选后的音频'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径（可选）'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行处理的工作线程数（默认: 4）'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别（默认: INFO）'
    )
    
    parser.add_argument(
        '--auth-token',
        type=str,
        help='Hugging Face认证令牌（用于pyannote模型，如果使用本地模型则不需要）'
    )
    
    parser.add_argument(
        '--use-local-models',
        action='store_true',
        default=True,
        help='使用本地pyannote模型（默认: True）'
    )
    
    parser.add_argument(
        '--no-local-models',
        action='store_true',
        help='不使用本地模型，从Hugging Face Hub下载（需要auth-token）'
    )
    
    parser.add_argument(
        '--local-model-path',
        type=str,
        default='pyannote',
        help='本地pyannote模型目录路径（默认: pyannote）'
    )
    
    # VAD相关参数
    parser.add_argument(
        '--vad-threshold',
        type=float,
        help='VAD检测阈值（0.0-1.0，默认: 0.5）'
    )
    
    parser.add_argument(
        '--min-speech-duration',
        type=float,
        help='最短语音时长（秒，默认: 0.5）'
    )
    
    # 质量筛选相关参数
    parser.add_argument(
        '--distilmos-threshold',
        type=float,
        help='DistilMOS评分阈值（1.0-5.0，默认: 3.0）'
    )
    
    parser.add_argument(
        '--dnsmos-threshold',
        type=float,
        help='DNSMOS评分阈值（1.0-5.0，默认: 3.0）'
    )
    
    parser.add_argument(
        '--dnsmospro-threshold',
        type=float,
        help='DNSMOSPro评分阈值（1.0-5.0，默认: 3.0）'
    )
    
    parser.add_argument(
        '--min-words',
        type=int,
        help='最少词数要求（默认: 1）'
    )
    
    # Whisper相关参数
    parser.add_argument(
        '--whisper-model',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper模型大小（默认: large-v3）'
    )
    
    parser.add_argument(
        '--language',
        help='目标语言代码（如: zh, en, ja），默认自动检测'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        help='计算设备（默认: auto）'
    )
    
    # 处理选项
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，只检查配置不实际处理'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='遇到错误时继续处理其他文件'
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """验证命令行参数"""
    errors = []
    
    # 验证输入目录（如果指定）
    if args.input and not Path(args.input).exists():
        errors.append(f"输入目录不存在: {args.input}")
    
    # 验证工作线程数
    if args.workers <= 0:
        errors.append("工作线程数必须大于0")
    
    # 验证阈值参数
    if args.vad_threshold is not None and not (0.0 <= args.vad_threshold <= 1.0):
        errors.append("VAD阈值必须在0.0-1.0之间")
    
    if args.distilmos_threshold is not None and not (1.0 <= args.distilmos_threshold <= 5.0):
        errors.append("DistilMOS阈值必须在1.0-5.0之间")
    
    if args.dnsmos_threshold is not None and not (1.0 <= args.dnsmos_threshold <= 5.0):
        errors.append("DNSMOS阈值必须在1.0-5.0之间")
    
    if args.dnsmospro_threshold is not None and not (1.0 <= args.dnsmospro_threshold <= 5.0):
        errors.append("DNSMOSPro阈值必须在1.0-5.0之间")
    
    if args.min_words is not None and args.min_words < 0:
        errors.append("最少词数不能为负数")
    
    return errors

def create_config_from_args(args):
    """从命令行参数创建配置"""
    config = LongAudioProcessingConfig()
    
    # 更新配置
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.auth_token:
        config.speaker_diarization.auth_token = args.auth_token
    
    # 本地模型配置
    if args.no_local_models:
        config.speaker_diarization.use_local_models = False
        # 如果不使用本地模型但没有提供token，给出警告
        if not args.auth_token:
            print("警告: 不使用本地模型时需要提供--auth-token参数")
    else:
        config.speaker_diarization.use_local_models = True
    
    if args.local_model_path:
        config.speaker_diarization.local_model_path = args.local_model_path
    
    config.processing.max_workers = args.workers
    config.log_level = args.log_level
    
    # VAD配置
    if args.vad_threshold is not None:
        config.vad.threshold = args.vad_threshold
    if args.min_speech_duration is not None:
        config.vad.min_speech_duration = args.min_speech_duration
    
    # 质量筛选配置
    if args.distilmos_threshold is not None:
        config.quality_filter.distil_mos_threshold = args.distilmos_threshold
    if args.dnsmos_threshold is not None:
        config.quality_filter.dnsmos_threshold = args.dnsmos_threshold
    if args.dnsmospro_threshold is not None:
        config.quality_filter.dnsmospro_threshold = args.dnsmospro_threshold
    if args.min_words is not None:
        config.quality_filter.min_words = args.min_words
    
    # Whisper配置
    if args.whisper_model:
        if args.whisper_model.startswith('openai/'):
            config.whisper.model_name = args.whisper_model
        else:
            config.whisper.model_name = f"openai/whisper-{args.whisper_model}"
    if args.language:
        config.whisper.language = args.language
    if args.device:
        config.whisper.device = args.device
    
    return config

def print_config_summary(config):
    """打印配置摘要"""
    print("当前配置:")
    print(f"  输入目录: {config.input_dir}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  并行工作线程: {config.processing.max_workers}")
    print(f"  日志级别: {config.log_level}")
    print()
    print("说话人分离配置:")
    print(f"  使用本地模型: {'是' if config.speaker_diarization.use_local_models else '否'}")
    if config.speaker_diarization.use_local_models:
        print(f"  本地模型路径: {config.speaker_diarization.local_model_path}")
        print(f"  分离模型: {config.speaker_diarization.diarization_model}")
    else:
        print(f"  HF Token: {'已设置' if config.speaker_diarization.auth_token else '未设置'}")
    print()
    print("VAD配置:")
    print(f"  阈值: {config.vad.threshold}")
    print(f"  最短语音时长: {config.vad.min_speech_duration}秒")
    print()
    print("Whisper配置:")
    print(f"  模型: {config.whisper.model_name}")
    print(f"  语言: {config.whisper.language or '自动检测'}")
    print(f"  设备: {config.whisper.device}")
    print()
    print("质量筛选配置:")
    print(f"  DistilMOS阈值: {config.quality_filter.distil_mos_threshold}")
    print(f"  DNSMOS阈值: {config.quality_filter.dnsmos_threshold}")
    print(f"  DNSMOSPro阈值: {config.quality_filter.dnsmospro_threshold}")
    print(f"  最少词数: {config.quality_filter.min_words}")

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=== 长音频处理系统 ===")
    print("功能:")
    print("1. 使用ten-vad和pyannote-audio进行说话人聚类")
    print("2. 基于说话人信息分割音频")
    print("3. 使用whisper + MOS评分进行质量筛选")
    print("4. 按结构化目录保存结果")
    print()
    
    try:
        # 验证参数
        errors = validate_arguments(args)
        if errors:
            print("参数验证失败:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 打印配置摘要
        print_config_summary(config)
        
        # 检查输入目录
        if not Path(config.input_dir).exists():
            print(f"错误: 输入目录不存在: {config.input_dir}")
            sys.exit(1)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 试运行模式
        if args.dry_run:
            print("\n=== 试运行模式 ===")
            # 查找音频文件数量
            from long_audio_processor import LongAudioProcessor
            temp_processor = LongAudioProcessor(config)
            audio_files = temp_processor.find_audio_files(config.input_dir)
            print(f"找到 {len(audio_files)} 个音频文件")
            print("配置验证通过，退出试运行模式")
            return
        
        print("开始处理...")
        print("-" * 50)
        
        # 创建处理器并执行
        from long_audio_processor import LongAudioProcessor
        processor = LongAudioProcessor(config)
        stats = processor.process_directory()
        
        print("-" * 50)
        print("处理完成!")
        print(f"总文件数: {stats['total_files']}")
        print(f"成功处理: {stats['successful_files']}")
        print(f"处理失败: {stats['failed_files']}")
        print(f"总说话人数: {stats['total_speakers']}")
        print(f"总音频片段: {stats['total_segments']}")
        print(f"通过筛选: {stats['passed_segments']}")
        print(f"总处理时间: {stats['total_processing_time']:.2f} 秒")
        
        if stats['total_segments'] > 0:
            pass_rate = stats['passed_segments'] / stats['total_segments'] * 100
            print(f"通过率: {pass_rate:.1f}%")
        
        if stats['total_audio_duration'] > 0:
            speech_ratio = stats['total_speech_time'] / stats['total_audio_duration'] * 100
            print(f"语音比例: {speech_ratio:.1f}%")
        
        print(f"\n结果保存在: {config.output_dir}")
        print(f"详细报告: {config.output_dir}/final_report.json")
        
    except KeyboardInterrupt:
        print("\n处理被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 