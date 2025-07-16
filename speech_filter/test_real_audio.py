#!/usr/bin/env python3
"""
真实音频文件测试脚本
从指定目录读取真实音频文件进行各功能模块测试
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import List, Optional
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_test_audio_file():
    """查找测试音频文件"""
    # 配置测试目录
    test_directory = "/root/group-shared/voiceprint/data/speech/speech_enhancement/starrail_3.3/日语 - Japanese"
    
    print(f"📁 测试目录: {test_directory}")
    print(f"🔍 搜索音频文件: {test_directory}")
    
    if not os.path.exists(test_directory):
        print(f"❌ 测试目录不存在: {test_directory}")
        return None
    
    # 查找音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for root, dirs, files in os.walk(test_directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("❌ 未找到音频文件")
        return None
    
    print(f"✅ 找到 {len(audio_files)} 个音频文件")
    
    # 选择一个测试文件（取第一个）
    test_file = audio_files[0]
    print(f"🎯 选择测试文件: {os.path.basename(test_file)}")
    
    return test_file

def get_audio_info(audio_path: str) -> dict:
    """
    获取音频文件信息
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        音频信息字典
    """
    try:
        import librosa
        import soundfile as sf
        
        # 获取基本信息
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        channels = info.channels
        
        # 获取文件大小
        file_size = os.path.getsize(audio_path)
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': channels,
            'file_size': file_size,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception as e:
        logger.error(f"获取音频信息失败: {audio_path}, 错误: {e}")
        return {}

def test_vad_with_real_audio(audio_path: str):
    """使用真实音频测试VAD"""
    print("\n" + "="*50)
    print("测试VAD (语音活动检测) - 真实音频")
    print("="*50)
    
    try:
        from config import load_config_from_yaml
        from vad_detector import VADDetector
        
        config = load_config_from_yaml()
        vad_detector = VADDetector(config)
        
        print(f"📁 测试音频: {os.path.basename(audio_path)}")
        
        # 获取音频信息
        audio_info = get_audio_info(audio_path)
        if audio_info:
            print(f"   时长: {audio_info['duration']:.2f}秒")
            print(f"   采样率: {audio_info['sample_rate']}Hz")
            print(f"   声道数: {audio_info['channels']}")
            print(f"   文件大小: {audio_info['file_size']/1024/1024:.2f}MB")
        
        # VAD检测
        print("\n🎯 开始VAD检测...")
        start_time = time.time()
        
        # 详细检测
        result = vad_detector.detect_speech_segments_detailed(audio_path)
        
        end_time = time.time()
        
        print(f"⏱️  检测耗时: {end_time - start_time:.2f}秒")
        print(f"✅ 检测成功: {result.success}")
        
        if result.success:
            print(f"📊 检测结果:")
            print(f"   语音段数量: {len(result.segments)}")
            print(f"   总语音时长: {result.total_voice_duration:.2f}秒")
            
            if result.segments:
                print("   语音段详情:")
                for i, (start, end) in enumerate(result.segments[:5]):  # 只显示前5个
                    print(f"     段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
                
                if len(result.segments) > 5:
                    print(f"     ... 还有 {len(result.segments) - 5} 个语音段")
                
                # 计算语音占比
                if audio_info and audio_info.get('duration'):
                    voice_ratio = result.total_voice_duration / audio_info['duration'] * 100
                    print(f"   语音占比: {voice_ratio:.1f}%")
            
            print("✅ VAD测试通过")
            return True
        else:
            print(f"❌ VAD检测失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ VAD测试出错: {e}")
        return False

def test_whisper_with_real_audio(audio_path: str):
    """使用真实音频测试Whisper"""
    print("\n" + "="*50)
    print("测试Whisper (语音识别) - 真实音频")
    print("="*50)
    
    try:
        from config import load_config_from_yaml
        from speech_recognizer import SpeechRecognizer
        
        config = load_config_from_yaml()
        # 设置日语语言和合适的模型
        config.asr.language = "ja"
        config.asr.model_name = "large-v3"  # 使用大模型以获得更好的日语识别效果
        
        speech_recognizer = SpeechRecognizer(config)
        
        print(f"📁 测试音频: {os.path.basename(audio_path)}")
        print(f"🌐 目标语言: 日语 (ja)")
        print(f"🤖 模型: {config.asr.model_name}")
        
        # 语音识别
        print("\n🎯 开始语音识别...")
        start_time = time.time()
        
        result = speech_recognizer.transcribe_audio(audio_path)
        
        end_time = time.time()
        
        print(f"⏱️  识别耗时: {end_time - start_time:.2f}秒")
        print(f"✅ 识别成功: {result.get('success', False)}")
        
        if result.get('success', False):
            text = result.get('text', '').strip()
            language = result.get('language', 'unknown')
            word_count = result.get('word_count', 0)
            
            print(f"📊 识别结果:")
            print(f"   检测语言: {language}")
            print(f"   词数: {word_count}")
            print(f"   转录文本: '{text}'")
            
            # 验证结果
            is_valid = speech_recognizer.is_valid_transcription(result)
            print(f"   结果验证: {'✅ 有效' if is_valid else '❌ 无效'}")
            
            # 语言匹配检查
            if language == 'ja':
                print("   语言匹配: ✅ 匹配")
            else:
                print(f"   语言匹配: ⚠️ 不匹配 (期望: ja, 实际: {language})")
            
            print("✅ Whisper测试通过")
            return True
        else:
            error_msg = result.get('error', '未知错误')
            print(f"❌ Whisper识别失败: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Whisper测试出错: {e}")
        return False

def test_audio_quality_with_real_audio(audio_file):
    """测试音质评估（DistilMOS & DNSMOS & DNSMOSPro）"""
    print("==================================================")
    print("测试音质评估 (DistilMOS & DNSMOS & DNSMOSPro) - 真实音频")
    print("==================================================")
    
    try:
        from config import load_config_from_yaml
        from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        print(f"配置文件加载成功: {Path(__file__).parent / 'config.yaml'}")
        
        print(f"📁 测试音频: {audio_file.name}")
        print(f"📊 评估工具启用状态:")
        print(f"   DistilMOS: {'✅ 启用' if config.audio_quality.use_distil_mos else '❌ 禁用'}")
        print(f"   DNSMOS: {'✅ 启用' if config.audio_quality.use_dnsmos else '❌ 禁用'}")
        print(f"   DNSMOSPro: {'✅ 启用' if config.audio_quality.use_dnsmospro else '❌ 禁用'}")
        print(f"📊 评估阈值:")
        print(f"   DistilMOS: {config.audio_quality.distil_mos_threshold}")
        print(f"   DNSMOS: {config.audio_quality.dnsmos_threshold}")
        print(f"   DNSMOSPro: {config.audio_quality.dnsmospro_threshold}")
        
        # 创建音质评估器
        assessor = AudioQualityAssessor(config)
        
        print("\n🎯 开始音质评估...")
        start_time = time.time()
        
        # 执行音质评估
        result = assessor.assess_audio_quality(str(audio_file))
        
        end_time = time.time()
        
        print(f"⏱️  评估耗时: {end_time - start_time:.2f}秒")
        
        if result['success']:
            print("✅ 音质评估成功")
            scores = result['scores']
            
            print("📊 评估结果:")
            
            # 显示各个评估工具的结果
            if 'distilmos' in scores:
                print(f"   DistilMOS: {scores['distilmos']:.3f}")
                print(f"     质量等级: {get_quality_level(scores['distilmos'])}")
            
            if 'dnsmos' in scores:
                print(f"   DNSMOS: {scores['dnsmos']:.3f}")
                print(f"     质量等级: {get_quality_level(scores['dnsmos'])}")
            
            if 'dnsmospro' in scores:
                print(f"   DNSMOSPro: {scores['dnsmospro']:.3f}")
                print(f"     质量等级: {get_quality_level(scores['dnsmospro'])}")
            
            # 综合评分
            overall_score = scores.get('overall', 0)
            print(f"   综合评分: {overall_score:.3f}")
            print(f"     综合质量: {get_quality_level(overall_score)}")
            
            # 质量判定
            is_high_quality = assessor.is_high_quality(result)
            print(f"🏆 质量判定: {'✅ 通过' if is_high_quality else '❌ 不通过'}")
            
        else:
            print("❌ 音质评估失败")
            if 'error' in result:
                print(f"   错误: {result['error']}")
            return False
        
        print("✅ 音质评估测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 音质评估测试失败: {e}")
        return False

def get_quality_level(score: float) -> str:
    """
    根据评分判断质量等级
    
    Args:
        score: 评分
        
    Returns:
        质量等级字符串
    """
    if score >= 4.0:
        return "优秀"
    elif score >= 3.0:
        return "良好"
    elif score >= 2.0:
        return "一般"
    else:
        return "较差"

def test_dnsmospro_repository_config():
    """测试DNSMOSPro仓库配置"""
    print("==================================================")
    print("测试DNSMOSPro仓库配置")
    print("==================================================")
    
    try:
        # 加载配置
        from config import load_config_from_yaml
        from audio_quality_assessor import AudioQualityAssessor
        
        config = load_config_from_yaml()
        
        # 检查配置
        print(f"📁 模型缓存目录: {config.asr.model_cache_dir}")
        print(f"📊 DNSMOSPro启用状态: {'✅ 启用' if config.audio_quality.use_dnsmospro else '❌ 禁用'}")
        
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
            print("   下载链接: https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/NISQA/model_best.pt")
        
        # 测试模型加载
        print("\n🔧 测试模型加载...")
        try:
            assessor = AudioQualityAssessor(config)
            
            if assessor.distilmos_model is not None:
                print("✅ Distill-MOS模型加载成功")
            else:
                print("❌ Distill-MOS模型加载失败")
            
            if assessor.dnsmos_compute_score is not None:
                print("✅ DNSMOS模型加载成功")
            else:
                print("❌ DNSMOS模型加载失败")
            
            if assessor.dnsmospro_model is not None:
                print("✅ DNSMOSPro模型加载成功")
            else:
                print("❌ DNSMOSPro模型加载失败")
            
        except Exception as e:
            print(f"❌ 模型加载测试失败: {e}")
            return False
        
        print("✅ DNSMOSPro仓库配置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ DNSMOSPro仓库配置测试失败: {e}")
        return False


def test_complete_pipeline_with_real_audio(audio_path):
    """使用真实音频测试完整处理流程"""
    print("\n" + "="*50)
    print("测试完整处理流程 - 真实音频")
    print("="*50)
    
    try:
        from config import load_config_from_yaml
        from pipeline import SpeechFilterPipeline
        
        config = load_config_from_yaml()
        pipeline = SpeechFilterPipeline(config)
        
        print(f"📁 测试音频: {os.path.basename(audio_path)}")
        
        # 创建临时输出目录
        temp_output_dir = "temp_test_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # 获取输入目录
            input_dir = os.path.dirname(audio_path)
            
            # 处理单个文件
            print("\n🚀 开始完整流程处理...")
            start_time = time.time()
            
            result = pipeline._process_single_file(audio_path, input_dir, temp_output_dir)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"⏱️  处理耗时: {elapsed_time:.2f}秒")
            print(f"✅ 处理成功: {result.passed}")
            
            # 显示VAD结果
            if result.vad_segments:
                print(f"📊 VAD检测: 发现 {len(result.vad_segments)} 个语音段")
                total_speech_time = sum(end - start for start, end in result.vad_segments)
                print(f"   总语音时长: {total_speech_time:.2f}秒")
            
            # 显示转录结果
            if result.transcription:
                print(f"📝 语音识别: 成功 = {result.transcription.get('success', False)}")
                if result.transcription.get('text'):
                    print(f"   转录文本: {result.transcription['text'][:100]}...")
                    print(f"   检测语言: {result.transcription.get('language', 'unknown')}")
                    print(f"   词数: {result.transcription.get('word_count', 0)}")
            
            # 显示音质评估结果
            if result.quality_scores and 'scores' in result.quality_scores:
                scores = result.quality_scores['scores']
                print(f"🎵 音质评估:")
                if 'distilmos' in scores:
                    print(f"   DistilMOS: {scores['distilmos']:.3f}")
                if 'dnsmos' in scores:
                    print(f"   DNSMOS: {scores['dnsmos']:.3f}")
                if 'dnsmospro' in scores:
                    print(f"   DNSMOSPro: {scores['dnsmospro']:.3f}")
                if 'overall' in scores:
                    print(f"   综合评分: {scores['overall']:.3f}")
            
            # 显示最终结果
            if result.passed:
                print("🎉 音频文件通过所有筛选条件")
                
                # 检查输出文件是否存在
                relative_path = os.path.relpath(audio_path, input_dir)
                output_file = os.path.join(temp_output_dir, relative_path)
                if os.path.exists(output_file):
                    print(f"✅ 输出文件已生成: {output_file}")
                else:
                    print(f"❌ 输出文件未生成: {output_file}")
            else:
                print("❌ 音频文件未通过筛选")
                if result.error_message:
                    print(f"   失败原因: {result.error_message}")
            
            print("✅ 完整流程测试通过")
            return True
            
        finally:
            # 清理临时目录
            if os.path.exists(temp_output_dir):
                import shutil
                shutil.rmtree(temp_output_dir)
                
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎤 真实音频文件测试 - 完整音质评估系统")
    print("="*60)
    print("📊 测试内容:")
    print("   • DNSMOSPro仓库配置验证")
    print("   • VAD语音活动检测")
    print("   • Whisper语音识别")
    print("   • 三合一音质评估 (DistilMOS + DNSMOS + DNSMOSPro)")
    print("   • 完整流程集成测试")
    print("="*60)
    
    # 查找测试音频文件
    test_file = find_test_audio_file()
    if not test_file:
        print("❌ 未找到测试音频文件")
        return 1
    
    # 测试项目列表
    tests = [
        ("DNSMOSPro配置", lambda: test_dnsmospro_repository_config()),
        ("VAD检测", lambda: test_vad_with_real_audio(test_file)),
        ("Whisper识别", lambda: test_whisper_with_real_audio(test_file)),
        ("音质评估", lambda: test_audio_quality_with_real_audio(Path(test_file))),
        ("完整流程", lambda: test_complete_pipeline_with_real_audio(test_file))
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for i, (test_name, test_func) in enumerate(tests, 1):
        print(f"\n🔍 开始测试: {test_name}")
        print(f"进度: {i}/{total_count}")
        
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name}测试通过")
                success_count += 1
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    # 测试总结
    print("\n" + "="*60)
    print(f"🎯 测试完成: {success_count}/{total_count} 通过")
    print("="*60)
    
    if success_count == total_count:
        print("🎉 所有测试通过！真实音频处理系统工作正常")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关配置")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 