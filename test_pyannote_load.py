#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '/root/code/github_repos/DataFilter/long_speech_filter')

from contextlib import contextmanager
from pyannote.audio import Pipeline
import torch

@contextmanager
def change_dir(path):
    """临时切换工作目录"""
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)

def test_pyannote_loading():
    """测试PyAnnote本地模型加载"""
    print("=== 测试PyAnnote本地模型加载 ===")
    
    # 模型路径
    base_dir = "/root/code/github_repos/DataFilter/long_speech_filter"
    config_path = os.path.join(base_dir, "pyannote/speaker-diarization-3.1/config.yaml")
    
    print(f"基础目录: {base_dir}")
    print(f"配置文件: {config_path}")
    print(f"配置文件存在: {os.path.exists(config_path)}")
    
    try:
        with change_dir(base_dir):
            print(f"工作目录切换到: {os.getcwd()}")
            
            # 检查相对路径
            relative_config_path = os.path.relpath(config_path, base_dir)
            print(f"相对配置路径: {relative_config_path}")
            
            # 检查模型文件是否存在
            segmentation_path = "pyannote/segmentation-3.0/pytorch_model.bin"
            embedding_path = "pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
            
            print(f"分割模型存在: {os.path.exists(segmentation_path)}")
            print(f"嵌入模型存在: {os.path.exists(embedding_path)}")
            
            # 尝试加载管道
            print("尝试加载管道...")
            pipeline = Pipeline.from_pretrained(relative_config_path)
            print("✅ 管道加载成功!")
            
            # 测试GPU设置
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                pipeline.to(device)
                print(f"✅ 管道已移动到GPU: cuda:0")
            
            return True
            
    except Exception as e:
        print(f"❌ 管道加载失败: {e}")
        return False

if __name__ == "__main__":
    success = test_pyannote_loading()
    if success:
        print("\n🎉 PyAnnote本地模型加载测试成功!")
    else:
        print("\n💥 PyAnnote本地模型加载测试失败!") 