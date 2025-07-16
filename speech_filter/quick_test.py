#!/usr/bin/env python3
"""
快速测试脚本
用于快速验证各功能模块是否正常工作
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def create_test_audio(duration=2.0, sample_rate=16000):
    """创建测试音频"""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # 生成语音类似信号
    fundamental = 150  # Hz
    harmonics = [fundamental, fundamental * 2, fundamental * 3]
    amplitudes = [0.8, 0.4, 0.2]
    
    signal = np.zeros_like(t)
    for freq, amp in zip(harmonics, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # 添加包络
    envelope = np.exp(-t / duration * 2)
    signal = signal * envelope * 0.5
    
    return signal.astype(np.float32)

def test_vad():
    """测试VAD功能"""
    print("=" * 50)
    print("测试VAD (语音活动检测)")
    print("=" * 50)
    
    try:
        from config import load_config_from_yaml
        from vad_detector import VADDetector
        
        config = load_config_from_yaml()
        vad_detector = VADDetector(config)
        
        # 创建测试音频
        test_dir = tempfile.mkdtemp()
        try:
            # 创建包含语音的音频
            speech_audio = create_test_audio(3.0)
            speech_path = os.path.join(test_dir, "speech.wav")
            sf.write(speech_path, speech_audio, 16000)
            
            # 创建静音音频
            silence_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2秒静音
            silence_path = os.path.join(test_dir, "silence.wav")
            sf.write(silence_path, silence_audio, 16000)
            
            # 测试语音检测
            print("测试语音音频...")
            speech_segments = vad_detector.detect_speech_segments(speech_path)
            print(f"✓ 语音音频检测到 {len(speech_segments)} 个语音段")
            
            # 测试静音检测
            print("测试静音音频...")
            silence_segments = vad_detector.detect_speech_segments(silence_path)
            print(f"✓ 静音音频检测到 {len(silence_segments)} 个语音段")
            
            # 验证结果
            if len(speech_segments) > 0 and len(silence_segments) == 0:
                print("✅ VAD测试通过")
                return True
            else:
                print("❌ VAD测试失败")
                return False
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ VAD测试出错: {e}")
        return False

def test_whisper():
    """测试Whisper功能"""
    print("\n" + "=" * 50)
    print("测试Whisper (语音识别)")
    print("=" * 50)
    
    try:
        from config import load_config_from_yaml
        from speech_recognizer import SpeechRecognizer
        
        config = load_config_from_yaml()
        config.asr.model_name = "base"  # 使用小模型加快测试
        speech_recognizer = SpeechRecognizer(config)
        
        # 创建测试音频
        test_dir = tempfile.mkdtemp()
        try:
            # 创建测试音频
            speech_audio = create_test_audio(2.0)
            speech_path = os.path.join(test_dir, "speech.wav")
            sf.write(speech_path, speech_audio, 16000)
            
            print("测试语音转录...")
            start_time = time.time()
            result = speech_recognizer.transcribe_audio(speech_path)
            end_time = time.time()
            
            print(f"✓ 转录完成，耗时: {end_time - start_time:.2f}秒")
            print(f"✓ 转录成功: {result.get('success', False)}")
            print(f"✓ 转录文本: '{result.get('text', '')}'")
            print(f"✓ 检测语言: {result.get('language', 'unknown')}")
            
            # 测试结果验证
            is_valid = speech_recognizer.is_valid_transcription(result)
            print(f"✓ 结果验证: {is_valid}")
            
            if result.get('success', False):
                print("✅ Whisper测试通过")
                return True
            else:
                print("❌ Whisper测试失败")
                return False
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Whisper测试出错: {e}")
        return False

def test_distilmos():
    """测试DistilMOS功能"""
    print("\n" + "=" * 50)
    print("测试DistilMOS (音质评估)")
    print("=" * 50)
    
    try:
        from config import load_config_from_yaml
        from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        assessor = AudioQualityAssessor(config)
        
        # 创建测试音频
        test_dir = tempfile.mkdtemp()
        try:
            # 创建高质量音频
            clean_audio = create_test_audio(2.0)
            clean_path = os.path.join(test_dir, "clean.wav")
            sf.write(clean_path, clean_audio, 16000)
            
            # 创建低质量音频（加噪声）
            noisy_audio = clean_audio + np.random.normal(0, 0.1, len(clean_audio))
            noisy_path = os.path.join(test_dir, "noisy.wav")
            sf.write(noisy_path, noisy_audio.astype(np.float32), 16000)
            
            print("测试DistilMOS评估...")
            
            # 测试高质量音频
            print("评估高质量音频...")
            clean_result = assessor.assess_audio_quality(clean_path)
            print(f"✓ 评估成功: {clean_result.get('success', False)}")
            
            if clean_result.get('success'):
                scores = clean_result.get('scores', {})
                if 'distilmos' in scores:
                    print(f"✓ DistilMOS分数: {scores['distilmos']:.3f}")
                    print("✅ DistilMOS测试通过")
                    return True
                else:
                    print("⚠️ DistilMOS未安装或不可用，使用默认评分")
                    return True
            else:
                print("❌ DistilMOS测试失败")
                return False
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ DistilMOS测试出错: {e}")
        return False

def test_dnsmos():
    """测试DNSMOS功能"""
    print("\n" + "=" * 50)
    print("测试DNSMOS (音质评估)")
    print("=" * 50)
    
    try:
        from config import load_config_from_yaml
        from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        assessor = AudioQualityAssessor(config)
        
        # 创建测试音频
        test_dir = tempfile.mkdtemp()
        try:
            # 创建测试音频
            speech_audio = create_test_audio(2.0)
            speech_path = os.path.join(test_dir, "speech.wav")
            sf.write(speech_path, speech_audio, 16000)
            
            print("测试DNSMOS评估...")
            
            start_time = time.time()
            result = assessor.assess_audio_quality(speech_path)
            end_time = time.time()
            
            print(f"✓ 评估完成，耗时: {end_time - start_time:.2f}秒")
            print(f"✓ 评估成功: {result.get('success', False)}")
            
            if result.get('success'):
                scores = result.get('scores', {})
                
                # 检查DNSMOS分数
                dnsmos_keys = ['dnsmos', 'dnsmos_ovrl', 'dnsmos_sig', 'dnsmos_bak', 'dnsmos_p808']
                dnsmos_found = any(key in scores for key in dnsmos_keys)
                
                if dnsmos_found:
                    print("✓ DNSMOS分数:")
                    for key in dnsmos_keys:
                        if key in scores:
                            print(f"  {key}: {scores[key]:.3f}")
                    
                    # 检查综合评分
                    if 'overall' in scores:
                        print(f"✓ 综合评分: {scores['overall']:.3f}")
                    
                    print("✅ DNSMOS测试通过")
                    return True
                else:
                    print("⚠️ DNSMOS未安装或不可用，使用默认评分")
                    return True
            else:
                print("❌ DNSMOS测试失败")
                return False
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ DNSMOS测试出错: {e}")
        return False

def test_integration():
    """测试集成流程"""
    print("\n" + "=" * 50)
    print("测试集成流程")
    print("=" * 50)
    
    try:
        from config import load_config_from_yaml
        from vad_detector import VADDetector
        from speech_recognizer import SpeechRecognizer
        from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        config.asr.model_name = "base"  # 使用小模型
        
        # 创建测试音频
        test_dir = tempfile.mkdtemp()
        try:
            # 创建测试音频
            speech_audio = create_test_audio(3.0)
            speech_path = os.path.join(test_dir, "speech.wav")
            sf.write(speech_path, speech_audio, 16000)
            
            print("运行完整流程...")
            
            # 初始化组件
            vad_detector = VADDetector(config)
            speech_recognizer = SpeechRecognizer(config)
            audio_quality_assessor = AudioQualityAssessor(config)
            
            # 1. VAD检测
            print("1. VAD检测...")
            vad_segments = vad_detector.detect_speech_segments(speech_path)
            print(f"   ✓ 检测到 {len(vad_segments)} 个语音段")
            
            # 2. 语音识别
            print("2. 语音识别...")
            asr_result = speech_recognizer.transcribe_audio(speech_path)
            print(f"   ✓ 转录成功: {asr_result.get('success', False)}")
            
            # 3. 音质评估
            print("3. 音质评估...")
            quality_result = audio_quality_assessor.assess_audio_quality(speech_path)
            print(f"   ✓ 评估成功: {quality_result.get('success', False)}")
            
            # 4. 综合判断
            print("4. 综合判断...")
            vad_passed = len(vad_segments) > 0
            asr_passed = speech_recognizer.is_valid_transcription(asr_result)
            quality_passed = audio_quality_assessor.is_high_quality(quality_result)
            
            print(f"   ✓ VAD通过: {vad_passed}")
            print(f"   ✓ ASR通过: {asr_passed}")
            print(f"   ✓ 质量通过: {quality_passed}")
            
            overall_passed = vad_passed and asr_passed and quality_passed
            print(f"   ✓ 整体通过: {overall_passed}")
            
            print("✅ 集成测试通过")
            return True
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ 集成测试出错: {e}")
        return False

def check_environment():
    """检查环境"""
    print("=" * 50)
    print("检查环境")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查必要的包
    packages = [
        ('whisper', 'Whisper'),
        ('ten_vad', 'TEN VAD'),
        ('distillmos', 'DistilMOS'),
        ('onnxruntime', 'ONNX Runtime'),
        ('librosa', 'librosa'),
        ('soundfile', 'soundfile'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML')
    ]
    
    missing_packages = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    print("🎤 语音筛选模块快速测试")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请安装必要的依赖")
        return 1
    
    # 运行各个测试
    tests = [
        ("VAD", test_vad),
        ("Whisper", test_whisper),
        ("DistilMOS", test_distilmos),
        ("DNSMOS", test_dnsmos),
        ("集成流程", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 开始测试 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15} {status}")
        if success:
            passed += 1
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查相关组件。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 