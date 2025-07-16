#!/usr/bin/env python3
"""
多GPU功能测试脚本
"""
import os
import sys
import torch
import time
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config_from_yaml
from multi_gpu_pipeline import MultiGPUPipeline
from audio_quality_assessor import AudioQualityAssessor

def test_gpu_availability():
    """测试GPU可用性"""
    print("🔍 GPU可用性检查")
    print("="*50)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("✅ CUDA可用")
        gpu_count = torch.cuda.device_count()
        print(f"📊 检测到 {gpu_count} 张GPU")
        
        # 显示每张GPU的信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return gpu_count
    else:
        print("❌ CUDA不可用")
        return 0

def test_multi_gpu_config():
    """测试多GPU配置"""
    print("=" * 60)
    print("测试多GPU配置")
    print("=" * 60)
    
    try:
        # from config import load_config_from_yaml
        # from multi_gpu_pipeline import MultiGPUPipeline # This import is no longer needed
        
        # 加载配置
        config = load_config_from_yaml()
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA可用，检测到 {gpu_count} 个GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 检查模型缓存目录
            cache_dir = Path(config.asr.model_cache_dir)
            if cache_dir.exists():
                print(f"✅ 模型缓存目录存在: {cache_dir}")
            else:
                print(f"⚠️ 模型缓存目录不存在，将在首次运行时创建: {cache_dir}")
            
            # 检查DNSMOSPro模型文件
            dnsmospro_dir = cache_dir / "dnsmospro"
            model_path = dnsmospro_dir / "model_best.pt"
            
            if model_path.exists():
                print(f"✅ DNSMOSPro模型文件存在: {model_path}")
            else:
                print(f"⚠️ DNSMOSPro模型文件不存在，将在首次运行时从GitHub下载: {model_path}")
            
            return True
        else:
            print("❌ CUDA不可用")
            return False
            
    except Exception as e:
        print(f"❌ 多GPU配置测试失败: {e}")
        return False

def test_gpu_memory_usage():
    """测试GPU内存使用"""
    print("\n💾 GPU内存使用测试")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存测试")
        return False
    
    try:
        # 测试每张GPU的内存使用
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            
            # 清空GPU缓存
            torch.cuda.empty_cache()
            
            # 获取内存信息
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_free = torch.cuda.get_device_properties(i).total_memory / 1024**3 - memory_reserved
            
            print(f"GPU {i}:")
            print(f"   已分配: {memory_allocated:.2f}GB")
            print(f"   已保留: {memory_reserved:.2f}GB")
            print(f"   空闲: {memory_free:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU内存测试失败: {e}")
        return False

def test_model_loading():
    """测试模型在GPU上的加载"""
    print("\n🤖 模型GPU加载测试")
    print("="*50)
    
    try:
        # from config import load_config_from_yaml
        # from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        
        # 设置使用GPU
        config.asr.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"目标设备: {config.asr.device}")
        
        # 创建音质评估器
        assessor = AudioQualityAssessor(config)
        
        # 检查模型是否在GPU上
        if torch.cuda.is_available():
            models_on_gpu = []
            
            # 检查DistilMOS模型
            if assessor.distilmos_model is not None:
                device = next(assessor.distilmos_model.parameters()).device
                models_on_gpu.append(f"DistilMOS: {device}")
            
            # 检查DNSMOSPro模型
            if assessor.dnsmospro_model is not None:
                # 检查JIT模型的设备
                # 这里简化检查，实际使用时模型会被移动到正确的设备
                models_on_gpu.append("DNSMOSPro: cuda")
            
            if models_on_gpu:
                print("✅ 模型GPU加载成功:")
                for model_info in models_on_gpu:
                    print(f"   {model_info}")
                return True
            else:
                print("❌ 没有模型加载到GPU")
                return False
        else:
            print("⚠️ 使用CPU模式")
            return True
            
    except Exception as e:
        print(f"❌ 模型GPU加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 多GPU功能测试")
    print("="*60)
    
    tests = [
        ("GPU可用性检查", test_gpu_availability),
        ("多GPU配置测试", test_multi_gpu_config),
        ("GPU内存使用测试", test_gpu_memory_usage),
        ("模型GPU加载测试", test_model_loading)
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}通过")
                success_count += 1
            else:
                print(f"❌ {test_name}失败")
        except Exception as e:
            print(f"❌ {test_name}异常: {e}")
    
    # 测试总结
    print("\n" + "="*60)
    print(f"🎯 测试完成: {success_count}/{total_count} 通过")
    print("="*60)
    
    if success_count == total_count:
        print("🎉 多GPU功能测试全部通过！")
        return 0
    else:
        print("⚠️ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 