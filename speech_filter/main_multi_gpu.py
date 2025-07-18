#!/usr/bin/env python3
"""
多GPU语音筛选Pipeline - 主入口文件
支持4张GPU并行处理
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config_from_yaml, merge_cli_args_with_config, save_config_to_yaml
from multi_gpu_pipeline import MultiGPUPipeline
from utils import setup_logging, validate_input_directory, create_output_directory, format_processing_summary

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='多GPU语音筛选Pipeline - 从大量音频文件中筛选出高质量的语音数据（4张GPU并行）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_multi_gpu.py /path/to/audio_files -o /path/to/filtered_audio
  python main_multi_gpu.py input_dir -o output_dir --language zh --num-gpus 4
  python main_multi_gpu.py input_dir -o output_dir --distilmos-threshold 3.5 --dnsmos-threshold 3.0
  python main_multi_gpu.py input_dir -o output_dir --skip-processed --num-gpus 4
        """
    )
    
    # 必需参数
    parser.add_argument('input_dir', help='输入音频文件目录')
    
    # 输出配置
    parser.add_argument('-o', '--output-dir', default='filtered_audio', help='输出目录 (默认: filtered_audio)')
    
    # 多GPU配置
    parser.add_argument('--num-gpus', type=int, default=4, help='使用的GPU数量 (默认: 4)')
    
    # 配置文件相关
    parser.add_argument('--config', help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--language-preset', choices=['japanese', 'chinese', 'english'], help='语言预设配置')
    parser.add_argument('--save-config', help='保存当前配置到指定文件')
    
    # VAD配置
    parser.add_argument('--vad-threshold', type=float, help='TEN VAD阈值 (默认: 0.5)')
    parser.add_argument('--vad-hop-size', type=int, help='TEN VAD跳跃大小 (默认: 256)')
    parser.add_argument('--min-speech-duration', type=float, help='最短语音时长（秒）')
    parser.add_argument('--max-speech-duration', type=float, help='最长语音时长（秒）')
    
    # Whisper配置
    parser.add_argument('--whisper-model', default='large-v3', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help='Whisper模型 (默认: large-v3)')
    parser.add_argument('--language', help='目标语言 (默认: 自动检测)')
    parser.add_argument('--min-words', type=int, help='最少词数要求')
    parser.add_argument('--model-cache-dir', default='/root/data/pretrained_models', help='模型缓存目录')
    
    # 音质评估配置
    parser.add_argument('--distilmos-threshold', type=float, help='DistilMOS阈值 (默认: 3.0)')
    parser.add_argument('--dnsmos-threshold', type=float, help='DNSMOS阈值 (默认: 3.0)')
    parser.add_argument('--dnsmospro-threshold', type=float, help='DNSMOSPro阈值 (默认: 3.0)')
    parser.add_argument('--disable-distilmos', action='store_true', help='禁用DistilMOS')
    parser.add_argument('--disable-dnsmos', action='store_true', help='禁用DNSMOS')
    parser.add_argument('--disable-dnsmospro', action='store_true', help='禁用DNSMOSPro')
    
    # 处理配置
    parser.add_argument('--sample-rate', type=int, help='重采样率 (默认: 16000)')
    parser.add_argument('--formats', nargs='+', default=['.wav', '.mp3', '.flac', '.m4a'], help='支持的音频格式')
    parser.add_argument('--skip-processed', action='store_true', help='跳过已处理的文件')
    parser.add_argument('--force-reprocess', action='store_true', help='强制重新处理所有文件')
    
    # 输出控制
    parser.add_argument('--export-transcriptions', action='store_true', help='导出转录文本')
    parser.add_argument('--export-quality-report', action='store_true', help='导出音质报告')
    parser.add_argument('--generate-html-report', action='store_true', help='生成HTML报告')
    parser.add_argument('--detailed-results', action='store_true', help='实时保存详细结果')
    parser.add_argument('--results-file', default='processing_results.json', help='结果文件名')
    parser.add_argument('--log-file', default='processing.log', help='日志文件名')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    return parser.parse_args()

def validate_arguments(args):
    """验证命令行参数"""
    errors = []
    
    # 验证输入目录
    if not validate_input_directory(args.input_dir):
        errors.append(f"输入目录不存在或无效: {args.input_dir}")
    
    # 验证GPU数量
    if args.num_gpus <= 0:
        errors.append("GPU数量必须大于0")
    
    # 验证音质阈值
    if args.distilmos_threshold and not (1.0 <= args.distilmos_threshold <= 5.0):
        errors.append("DistilMOS阈值必须在1.0-5.0之间")
    
    if args.dnsmos_threshold and not (1.0 <= args.dnsmos_threshold <= 5.0):
        errors.append("DNSMOS阈值必须在1.0-5.0之间")
    
    if args.dnsmospro_threshold and not (1.0 <= args.dnsmospro_threshold <= 5.0):
        errors.append("DNSMOSPro阈值必须在1.0-5.0之间")
    
    # 验证VAD阈值
    if args.vad_threshold and not (0.0 <= args.vad_threshold <= 1.0):
        errors.append("VAD阈值必须在0.0-1.0之间")
    
    # 验证采样率
    if args.sample_rate and args.sample_rate <= 0:
        errors.append("采样率必须大于0")
    
    return errors

def create_config_from_args(args):
    """从命令行参数创建配置"""
    # 加载基础配置
    config = load_config_from_yaml(args.config, args.language_preset)
    
    # 合并命令行参数
    config = merge_cli_args_with_config(config, args)
    
    # 如果需要保存配置文件
    if args.save_config:
        save_config_to_yaml(config, args.save_config)
        print(f"配置已保存到: {args.save_config}")
    
    return config

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 验证参数
        errors = validate_arguments(args)
        if errors:
            print("参数验证失败:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 设置日志
        log_file = os.path.join(args.output_dir, "logs", args.log_file)
        setup_logging(log_file, args.log_level)
        
        # 创建输出目录
        if not create_output_directory(args.output_dir):
            print(f"创建输出目录失败: {args.output_dir}", file=sys.stderr)
            sys.exit(1)
        
        # 确定是否跳过已处理文件
        skip_processed = args.skip_processed and not args.force_reprocess
        
        # 打印配置信息
        if not args.quiet:
            print("="*60)
            print("               多GPU语音筛选Pipeline")
            print("="*60)
            print(f"输入目录:           {args.input_dir}")
            print(f"输出目录:           {args.output_dir}")
            print(f"使用GPU数量:        {args.num_gpus}")
            print(f"跳过已处理文件:     {'是' if skip_processed else '否'}")
            print(f"Whisper模型:        {args.whisper_model}")
            print(f"目标语言:           {args.language or '自动检测'}")
            print(f"TEN VAD阈值:        {args.vad_threshold or config.vad.threshold}")
            print(f"TEN VAD跳跃大小:    {args.vad_hop_size or config.vad.hop_size}")
            print(f"DistilMOS阈值:      {args.distilmos_threshold or config.audio_quality.distil_mos_threshold}")
            print(f"DNSMOS阈值:         {args.dnsmos_threshold or config.audio_quality.dnsmos_threshold}")
            print(f"DNSMOSPro阈值:      {args.dnsmospro_threshold or config.audio_quality.dnsmospro_threshold}")
            print(f"支持格式:           {', '.join(args.formats)}")
            print("="*60)
            print()
        
        # 创建多GPU pipeline并开始处理
        multi_gpu_pipeline = MultiGPUPipeline(config, num_gpus=args.num_gpus, skip_processed=skip_processed)
        
        start_time = time.time()
        stats = multi_gpu_pipeline.process_directory(args.input_dir, args.output_dir)
        processing_time = time.time() - start_time
        
        # 打印处理统计
        if not args.quiet:
            multi_gpu_pipeline.print_statistics()
            print(f"\n总耗时: {processing_time:.2f}秒")
        
        # 导出额外报告
        if args.export_transcriptions:
            if not args.quiet:
                print("\n导出转录文本...")
            multi_gpu_pipeline.export_transcriptions(args.output_dir)
        
        if args.export_quality_report:
            if not args.quiet:
                print("\n导出音质报告...")
            multi_gpu_pipeline.export_quality_report(args.output_dir)
        
        if args.generate_html_report:
            if not args.quiet:
                print("\n生成HTML报告...")
            # HTML报告保存在上级目录
            html_path = os.path.join(os.path.dirname(args.output_dir), 'multi_gpu_report.html')
            try:
                from utils import generate_report_html
                # 使用存储的所有结果
                if generate_report_html(stats, [result.__dict__ for result in multi_gpu_pipeline.all_results], html_path):
                    if not args.quiet:
                        print(f"HTML报告已生成: {html_path}")
            except Exception as e:
                if not args.quiet:
                    print(f"生成HTML报告失败: {e}")
        
        if not args.quiet:
            print("\n处理完成！")
            print("="*60)
            print(f"输出目录: {args.output_dir}")
            print(f"处理统计: {os.path.dirname(args.output_dir)}/multi_gpu_stats.json")
            print(f"详细结果: 每个音频文件旁边的.json文件")
            print(f"日志文件: {os.path.dirname(args.output_dir)}/logs/")
            print("="*60)
        
        # 返回成功状态
        sys.exit(0 if stats['processed_files'] > 0 else 1)
    
    except KeyboardInterrupt:
        print("\n处理被用户中断", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 