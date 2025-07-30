#!/usr/bin/env python3

import sys
sys.path.insert(0, '/root/code/github_repos/DataFilter/long_speech_filter')

from config import LongAudioProcessingConfig
from multi_gpu_processor import MultiGPULongAudioProcessor
import copy

def debug_gpu_configs():
    """调试GPU配置传递"""
    print("=== 调试多GPU配置传递 ===")
    
    # 创建基础配置
    base_config = LongAudioProcessingConfig()
    print(f"基础配置设备: {base_config.whisper.device}")
    
    # 模拟多GPU处理器的配置创建
    for gpu_id in range(3):  # 测试前3个GPU
        print(f"\n--- GPU {gpu_id} ---")
        
        # 复制配置
        gpu_config = copy.deepcopy(base_config)
        
        # 设置GPU相关配置
        gpu_config.whisper.device = f"cuda:{gpu_id}"
        gpu_config.processing.temp_dir = f"temp_gpu_{gpu_id}"
        gpu_config._gpu_device = f"cuda:{gpu_id}"
        
        print(f"Whisper设备: {gpu_config.whisper.device}")
        print(f"临时目录: {gpu_config.processing.temp_dir}")
        print(f"GPU设备属性: {getattr(gpu_config, '_gpu_device', '未设置')}")
        
        # 检查说话人分离配置
        print(f"使用本地模型: {gpu_config.speaker_diarization.use_local_models}")
        print(f"使用PyAnnote: {gpu_config.speaker_diarization.use_pyannote}")
        
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_gpu_configs() 