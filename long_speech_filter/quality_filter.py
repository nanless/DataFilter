"""
音频质量筛选模块
集成Whisper语音识别和多种MOS质量评估
基于现有speech_filter框架
"""
import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# 添加speech_filter到路径
speech_filter_path = os.path.join(os.path.dirname(__file__), '..', 'speech_filter')

# 导入speech_filter模块的功能
try:
    # 临时添加speech_filter路径到sys.path开头
    if speech_filter_path not in sys.path:
        sys.path.insert(0, speech_filter_path)
        
    from speech_recognizer import SpeechRecognizer, ASRResult
    from audio_quality_assessor import AudioQualityAssessor, AudioQualityResult
    
    # 使用绝对路径导入speech_filter的config
    import importlib.util
    config_spec = importlib.util.spec_from_file_location(
        "speech_filter_config", 
        os.path.join(speech_filter_path, "config.py")
    )
    speech_filter_config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(speech_filter_config)
    
    ASRConfig = speech_filter_config.ASRConfig
    AudioQualityConfig = speech_filter_config.AudioQualityConfig
    ProcessingConfig = speech_filter_config.ProcessingConfig
    
    # 移除添加的path避免影响其他导入
    if speech_filter_path in sys.path:
        sys.path.remove(speech_filter_path)
        
    SPEECH_FILTER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入speech_filter模块: {e}")
    print("请确保speech_filter模块在正确的路径下，或安装相关依赖")
    SPEECH_FILTER_AVAILABLE = False
    
    # 创建占位符类以避免导入错误
    class SpeechRecognizer:
        def __init__(self, config): pass
        def transcribe_audio_detailed(self, path): 
            class Result:
                success = False
                error_message = "speech_filter模块不可用"
            return Result()
    
    class AudioQualityAssessor:
        def __init__(self, config): pass
        def assess_audio_quality_detailed(self, path):
            class Result:
                success = False
                error_message = "speech_filter模块不可用"
                scores = {}
            return Result()
    
    class ASRConfig:
        def __init__(self, **kwargs): 
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AudioQualityConfig:
        def __init__(self, **kwargs): 
            for k, v in kwargs.items():
                setattr(self, k, v)
        
    class ProcessingConfig:
        def __init__(self, **kwargs): 
            self.sample_rate = 16000  # 默认采样率
            for k, v in kwargs.items():
                setattr(self, k, v)

logger = logging.getLogger(__name__)

@dataclass
class AudioSegmentQuality:
    """音频片段质量信息"""
    audio_path: str
    passed: bool
    transcription: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class LongAudioQualityFilter:
    """长音频质量筛选器"""
    
    def __init__(self, config):
        self.config = config
        
        logger.info("🔧 正在初始化长音频质量筛选器...")
        
        # 检查speech_filter模块可用性
        if not SPEECH_FILTER_AVAILABLE:
            logger.error("❌ speech_filter模块不可用!")
            logger.error("   这将导致所有音频片段被标记为失败")
            logger.error("   请检查:")
            logger.error("   1. speech_filter目录是否存在于: ../speech_filter")
            logger.error("   2. 必要依赖是否已安装: pip install gin-config torch torchaudio transformers")
            logger.error("   3. speech_filter模块是否完整")
        else:
            logger.info("✅ speech_filter模块可用")
        
        # 创建兼容的配置对象
        self.asr_config = self._create_asr_config()
        self.quality_config = self._create_quality_config()
        self.processing_config = self._create_processing_config()
        
        # 创建一个兼容的配置对象用于初始化模块
        self.compat_config = self._create_compat_config()
        
        # 记录配置信息
        logger.info("📋 质量筛选配置:")
        logger.info(f"   🎤 Whisper模型: {self.config.whisper.model_name}")
        logger.info(f"   🔤 最少词数: {self.config.quality_filter.min_words}")
        logger.info(f"   📊 质量阈值:")
        if self.config.quality_filter.use_distil_mos:
            logger.info(f"      • DistilMOS ≥ {self.config.quality_filter.distil_mos_threshold}")
        if self.config.quality_filter.use_dnsmos:
            logger.info(f"      • DNSMOS ≥ {self.config.quality_filter.dnsmos_threshold}")
        if self.config.quality_filter.use_dnsmospro:
            logger.info(f"      • DNSMOSPro ≥ {self.config.quality_filter.dnsmospro_threshold}")
        
        # 初始化语音识别器
        logger.info("🎤 正在初始化Whisper语音识别器...")
        try:
            self.speech_recognizer = SpeechRecognizer(self.compat_config)
            logger.info("✅ 成功初始化语音识别器")
        except Exception as e:
            logger.error(f"❌ 初始化语音识别器失败: {e}")
            logger.exception("详细错误信息:")
            self.speech_recognizer = None
            
        # 初始化音质评估器
        logger.info("🎵 正在初始化MOS音质评估器...")
        try:
            self.quality_assessor = AudioQualityAssessor(self.compat_config)
            logger.info("✅ 成功初始化音质评估器")
        except Exception as e:
            logger.error(f"❌ 初始化音质评估器失败: {e}")
            logger.exception("详细错误信息:")
            self.quality_assessor = None
        
        # 最终状态检查
        if self.speech_recognizer and self.quality_assessor:
            logger.info("🎉 质量筛选器初始化完成 - 所有组件正常")
        else:
            logger.error("⚠️ 质量筛选器初始化不完整:")
            if not self.speech_recognizer:
                logger.error("   ❌ 语音识别器不可用")
            if not self.quality_assessor:
                logger.error("   ❌ 音质评估器不可用")
            logger.error("   这将导致所有音频片段评估失败")
    
    def _create_asr_config(self) -> ASRConfig:
        """创建ASR配置"""
        return ASRConfig(
            model_name=self.config.whisper.model_name,
            language=self.config.whisper.language,
            batch_size=self.config.whisper.batch_size,
            device=self.config.whisper.device,
            min_words=self.config.quality_filter.min_words,
            model_cache_dir=self.config.whisper.model_cache_dir
        )
    
    def _create_quality_config(self) -> AudioQualityConfig:
        """创建音质配置"""
        quality_config = AudioQualityConfig(
            distil_mos_threshold=self.config.quality_filter.distil_mos_threshold,
            dnsmos_threshold=self.config.quality_filter.dnsmos_threshold,
            dnsmospro_threshold=self.config.quality_filter.dnsmospro_threshold,
            use_distil_mos=self.config.quality_filter.use_distil_mos,
            use_dnsmos=self.config.quality_filter.use_dnsmos,
            use_dnsmospro=self.config.quality_filter.use_dnsmospro
        )
        
        # 如果配置中有GPU设备信息，传递给音质配置
        if hasattr(self.config, '_gpu_device'):
            quality_config.device = self.config._gpu_device
        
        return quality_config
    
    def _create_processing_config(self) -> ProcessingConfig:
        """创建处理配置"""
        return ProcessingConfig(
            supported_formats=self.config.processing.supported_formats,
            sample_rate=self.config.processing.sample_rate
        )
    
    def _create_compat_config(self):
        """创建兼容的配置对象"""
        class CompatConfig:
            def __init__(self, asr_config, quality_config, processing_config):
                self.asr = asr_config
                self.audio_quality = quality_config
                self.processing = processing_config
                self.sample_rate = processing_config.sample_rate
        
        return CompatConfig(self.asr_config, self.quality_config, self.processing_config)
    
    def evaluate_audio_segment(self, audio_path: str) -> AudioSegmentQuality:
        """
        评估单个音频片段的质量
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            AudioSegmentQuality: 质量评估结果
        """
        logger.info(f"🔍 开始评估音频片段: {Path(audio_path).name}")
        
        result = AudioSegmentQuality(
            audio_path=audio_path,
            passed=False
        )
        
        try:
            # 检查文件是否存在
            if not Path(audio_path).exists():
                result.error_message = f"音频文件不存在: {audio_path}"
                logger.error(f"❌ {result.error_message}")
                return result
            
            # 步骤1: 语音识别
            if not self.speech_recognizer:
                result.error_message = "语音识别器未初始化 - speech_filter模块不可用"
                logger.error(f"❌ {result.error_message}")
                return result
                
            logger.info("🎤 步骤1/2: 进行Whisper语音识别...")
            asr_result = self.speech_recognizer.transcribe_audio_detailed(audio_path)
            
            if not asr_result.success:
                result.error_message = f"Whisper识别失败: {asr_result.error_message}"
                logger.error(f"❌ {result.error_message}")
                return result
            
            # 记录识别结果的详细信息
            result.transcription = {
                'text': asr_result.text,
                'language': asr_result.language,
                'word_count': asr_result.word_count,
                'confidence': asr_result.confidence,
                'segments': asr_result.segments
            }
            
            logger.info(f"✅ Whisper识别成功:")
            logger.info(f"   📝 识别文本: '{asr_result.text[:50]}{'...' if len(asr_result.text) > 50 else ''}'")
            logger.info(f"   🔤 语言: {asr_result.language}")
            logger.info(f"   📊 词数: {asr_result.word_count}")
            confidence_str = f"{asr_result.confidence:.3f}" if asr_result.confidence is not None else "N/A"
            logger.info(f"   🎯 置信度: {confidence_str}")
            
            # 检查是否满足文字识别要求
            if self.config.quality_filter.require_text and not asr_result.text.strip():
                result.error_message = "未识别到文字内容"
                logger.warning(f"⚠️ {result.error_message}")
                return result
                
            if asr_result.word_count < self.config.quality_filter.min_words:
                result.error_message = f"词数不足，要求至少{self.config.quality_filter.min_words}词，实际{asr_result.word_count}词"
                logger.warning(f"⚠️ {result.error_message}")
                return result
            
            logger.info(f"✅ 文本要求检查通过 - 词数{asr_result.word_count}≥{self.config.quality_filter.min_words}")
            
            # 步骤2: 音质评估
            if not self.quality_assessor:
                result.error_message = "音质评估器未初始化 - speech_filter模块不可用"
                logger.error(f"❌ {result.error_message}")
                return result
                
            logger.info("🎵 步骤2/2: 进行MOS音质评估...")
            quality_result = self.quality_assessor.assess_audio_quality_detailed(audio_path)
            
            if not quality_result.success:
                result.error_message = f"MOS音质评估失败: {quality_result.error_message}"
                logger.error(f"❌ {result.error_message}")
                return result
            
            result.quality_scores = quality_result.scores
            
            # 详细记录MOS评分
            logger.info(f"✅ MOS音质评分结果:")
            for metric, score in quality_result.scores.items():
                logger.info(f"   📈 {metric}: {score:.3f}")
            
            # 检查各项MOS分数是否满足阈值
            quality_passed = True
            failed_metrics = []
            passed_metrics = []
            
            if self.config.quality_filter.use_distil_mos:
                distilmos_score = quality_result.scores.get('distilmos', 0.0)
                threshold = self.config.quality_filter.distil_mos_threshold
                if distilmos_score < threshold:
                    quality_passed = False
                    failed_metrics.append(f"DistilMOS({distilmos_score:.3f} < {threshold})")
                else:
                    passed_metrics.append(f"DistilMOS({distilmos_score:.3f} ≥ {threshold})")
            
            if self.config.quality_filter.use_dnsmos:
                # 查找可能的DNSMOS键名
                dnsmos_score = (quality_result.scores.get('dnsmos_overall') or 
                               quality_result.scores.get('dnsmos_ovrl') or 
                               quality_result.scores.get('dnsmos') or 0.0)
                threshold = self.config.quality_filter.dnsmos_threshold
                if dnsmos_score < threshold:
                    quality_passed = False
                    failed_metrics.append(f"DNSMOS({dnsmos_score:.3f} < {threshold})")
                else:
                    passed_metrics.append(f"DNSMOS({dnsmos_score:.3f} ≥ {threshold})")
            
            if self.config.quality_filter.use_dnsmospro:
                dnsmospro_score = (quality_result.scores.get('dnsmospro_overall') or
                                  quality_result.scores.get('dnsmospro') or 0.0)
                threshold = self.config.quality_filter.dnsmospro_threshold
                if dnsmospro_score < threshold:
                    quality_passed = False
                    failed_metrics.append(f"DNSMOSPro({dnsmospro_score:.3f} < {threshold})")
                else:
                    passed_metrics.append(f"DNSMOSPro({dnsmospro_score:.3f} ≥ {threshold})")
            
            # 记录阈值检查结果
            if passed_metrics:
                logger.info(f"✅ 通过的指标: {', '.join(passed_metrics)}")
            
            if not quality_passed:
                result.error_message = f"音质评分不满足要求: {', '.join(failed_metrics)}"
                logger.warning(f"⚠️ {result.error_message}")
                return result
            
            # 所有检查通过
            result.passed = True
            logger.info(f"🎉 音频片段质量评估全部通过: {Path(audio_path).name}")
            
        except Exception as e:
            logger.error(f"💥 评估音频片段时发生异常: {e}")
            logger.exception("详细异常信息:")
            result.error_message = str(e)
        
        return result
    
    def batch_evaluate_segments(self, audio_files: List[str]) -> List[AudioSegmentQuality]:
        """
        批量评估音频片段
        
        Args:
            audio_files: 音频文件路径列表
            
        Returns:
            List[AudioSegmentQuality]: 评估结果列表
        """
        results = []
        
        logger.info(f"🎯 开始批量评估 {len(audio_files)} 个音频片段")
        
        # 检查模块可用性
        if not SPEECH_FILTER_AVAILABLE:
            logger.error("❌ speech_filter模块不可用，所有音频将被标记为失败")
            logger.error("   请安装相关依赖：pip install gin-config")
            
        passed_count = 0
        failed_reasons = {}
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"📋 评估进度: {i}/{len(audio_files)} ({i/len(audio_files)*100:.1f}%)")
            result = self.evaluate_audio_segment(audio_file)
            results.append(result)
            
            # 统计结果
            if result.passed:
                passed_count += 1
                logger.info(f"✅ 第{i}个音频通过评估")
            else:
                # 统计失败原因
                reason = result.error_message or "未知错误"
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                logger.info(f"❌ 第{i}个音频未通过评估: {reason}")
        
        # 输出详细的汇总统计
        pass_rate = passed_count / len(results) * 100 if results else 0
        logger.info(f"")
        logger.info(f"📊 === 批量评估完成汇总 ===")
        logger.info(f"✅ 通过数量: {passed_count}/{len(results)} ({pass_rate:.1f}%)")
        logger.info(f"❌ 失败数量: {len(results) - passed_count}")
        
        if failed_reasons:
            logger.info(f"🔍 失败原因统计:")
            for reason, count in sorted(failed_reasons.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   • {reason}: {count}次 ({count/len(results)*100:.1f}%)")
        
        # 如果通过率为0，给出诊断建议
        if passed_count == 0 and len(results) > 0:
            logger.error(f"")
            logger.error(f"⚠️ === 0%通过率诊断 ===")
            if not SPEECH_FILTER_AVAILABLE:
                logger.error(f"🔧 主要问题: speech_filter模块不可用")
                logger.error(f"   解决方案: 确保speech_filter目录存在且包含正确的模块")
                logger.error(f"   检查依赖: pip install gin-config torch torchaudio transformers")
            else:
                logger.error(f"🔧 可能的问题:")
                logger.error(f"   1. 质量阈值设置过高")
                logger.error(f"   2. 音频质量确实较差")
                logger.error(f"   3. 模型加载或推理问题")
        
        return results
    
    def filter_and_save_results(self, evaluation_results: List[AudioSegmentQuality], 
                               output_dir: str, audio_id: str, speaker_id: str) -> Dict[str, Any]:
        """
        筛选并保存通过质量检查的音频片段
        
        Args:
            evaluation_results: 评估结果列表
            output_dir: 输出目录
            audio_id: 音频ID
            speaker_id: 说话人ID
            
        Returns:
            保存结果统计
        """
        # 创建输出目录结构
        speaker_output_dir = Path(output_dir) / audio_id / speaker_id
        speaker_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        metadata_files = []
        
        # 生成唯一的文件名前缀，包含时间戳和进程信息
        import time
        import os
        import uuid
        
        # 获取进程ID和时间戳用于生成唯一前缀
        process_id = getattr(self.config, '_process_id', os.getpid())
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        unique_prefix = f"{timestamp}_{process_id}"
        
        segment_counter = 1
        for result in evaluation_results:
            if result.passed:
                # 复制音频文件
                src_path = Path(result.audio_path)
                dst_filename = f"segment_{unique_prefix}_{segment_counter:03d}.wav"
                dst_path = speaker_output_dir / dst_filename
                
                # 如果文件已存在，添加随机后缀（极少发生）
                if dst_path.exists():
                    random_suffix = str(uuid.uuid4())[:8]
                    dst_filename = f"segment_{unique_prefix}_{segment_counter:03d}_{random_suffix}.wav"
                    dst_path = speaker_output_dir / dst_filename
                
                try:
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    saved_files.append(str(dst_path))
                    
                    # 保存元数据
                    metadata = {
                        'segment_id': f"{unique_prefix}_{segment_counter:03d}",
                        'segment_counter': segment_counter,
                        'process_id': process_id,
                        'timestamp': timestamp,
                        'original_path': str(src_path),
                        'saved_path': str(dst_path),
                        'audio_id': audio_id,
                        'speaker_id': speaker_id,
                        'transcription': result.transcription,
                        'quality_scores': result.quality_scores,
                        'evaluation_passed': True,
                        'processing_timestamp': self._get_timestamp()
                    }
                    
                    # 清理元数据，确保可以JSON序列化
                    clean_metadata = self._clean_for_json_serialization(metadata)
                    
                    metadata_filename = f"segment_{unique_prefix}_{segment_counter:03d}.json"
                    metadata_path = speaker_output_dir / metadata_filename
                    
                    # 如果元数据文件已存在，使用同样的随机后缀
                    if metadata_path.exists():
                        if 'random_suffix' in locals():
                            metadata_filename = f"segment_{unique_prefix}_{segment_counter:03d}_{random_suffix}.json"
                            metadata_path = speaker_output_dir / metadata_filename
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_metadata, f, ensure_ascii=False, indent=2)
                    
                    metadata_files.append(str(metadata_path))
                    segment_counter += 1
                    
                except Exception as e:
                    logger.error(f"保存文件失败 {src_path} -> {dst_path}: {e}")
        
        summary = {
            'audio_id': audio_id,
            'speaker_id': speaker_id,
            'total_segments': len(evaluation_results),
            'passed_segments': len(saved_files),
            'saved_files': saved_files,
            'metadata_files': metadata_files,
            'output_directory': str(speaker_output_dir),
            'pass_rate': len(saved_files) / len(evaluation_results) if evaluation_results else 0.0,
            'unique_prefix': unique_prefix,  # 添加唯一前缀信息
            'process_id': process_id
        }
        
        logger.info(f"保存完成 - {audio_id}/{speaker_id}: {len(saved_files)}/{len(evaluation_results)} 个片段通过筛选 ({summary['pass_rate']:.1%})")
        
        return summary
    
    def evaluate_audio_array(self, audio_array, metadata: Dict[str, Any]) -> AudioSegmentQuality:
        """
        评估音频数组质量（不保存为文件）
        
        Args:
            audio_array: 音频数组
            metadata: 音频元数据
            
        Returns:
            AudioSegmentQuality: 质量评估结果
        """
        try:
            import tempfile
            import soundfile as sf
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # 保存音频数组到临时文件
                sf.write(temp_path, audio_array, metadata.get('sample_rate', self.config.processing.sample_rate))
                
                # 评估音频质量
                result = self.evaluate_audio_segment(temp_path)
                
                # 更新路径信息
                result.audio_path = f"segment_{metadata.get('segment_id', 0)}"
                
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return result
                
        except Exception as e:
            logger.error(f"评估音频数组质量失败: {e}")
            return AudioSegmentQuality(
                audio_path=f"segment_{metadata.get('segment_id', 0)}",
                passed=False,
                error_message=str(e)
            )
    
    def process_speaker_audio_segments(self, speaker_audio_segments: Dict[str, List], 
                                     audio_id: str, output_dir: str) -> Dict[str, Any]:
        """
        处理说话人音频片段
        
        Args:
            speaker_audio_segments: {speaker_id: [(audio_array, metadata), ...]}
            audio_id: 音频ID
            output_dir: 输出目录
            
        Returns:
            处理结果统计
        """
        all_results = {}
        total_segments = 0
        total_passed = 0
        
        logger.info(f"🎭 开始处理 {len(speaker_audio_segments)} 个说话人的音频片段")
        
        for i, (speaker_id, segments) in enumerate(speaker_audio_segments.items(), 1):
            logger.info(f"👤 处理说话人 {i}/{len(speaker_audio_segments)}: {speaker_id} ({len(segments)} 个片段)")
            
            # 评估每个音频片段
            evaluation_results = self.batch_evaluate_segments([
                self._save_temp_audio(audio_array, metadata) 
                for audio_array, metadata in segments
            ])
            
            # 保存通过质量检查的片段
            save_result = self.save_audio_segments(
                segments, evaluation_results, output_dir, audio_id, speaker_id
            )
            
            all_results[speaker_id] = save_result
            total_segments += save_result['total_segments']
            total_passed += save_result['passed_segments']
            
            logger.info(f"✅ 说话人 {speaker_id} 完成: {save_result['passed_segments']}/{save_result['total_segments']} 个片段通过 ({save_result['pass_rate']:.1%})")
        
        summary = {
            'audio_id': audio_id,
            'speaker_results': all_results,
            'total_segments': total_segments,
            'total_passed': total_passed,
            'overall_pass_rate': total_passed / total_segments if total_segments > 0 else 0.0
        }
        
        logger.info(f"")
        logger.info(f"🎯 === {audio_id} 处理完成汇总 ===")
        logger.info(f"👥 处理说话人: {len(speaker_audio_segments)} 个")
        logger.info(f"🎵 总音频片段: {total_segments} 个")
        logger.info(f"✅ 通过筛选: {total_passed} 个")
        logger.info(f"📊 总通过率: {summary['overall_pass_rate']:.1%}")
        
        # 按说话人显示详细结果
        for speaker_id, result in all_results.items():
            logger.info(f"   👤 {speaker_id}: {result['passed_segments']}/{result['total_segments']} ({result['pass_rate']:.1%})")
        
        return summary
    
    def _save_temp_audio(self, audio_array, metadata):
        """保存临时音频文件用于评估"""
        import tempfile
        import soundfile as sf
        import os
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        # 写入音频数据
        sf.write(temp_path, audio_array, metadata.get('sample_rate', self.config.processing.sample_rate))
        
        return temp_path
    
    def save_audio_segments(self, segments, evaluation_results, output_dir: str, 
                           audio_id: str, speaker_id: str) -> Dict[str, Any]:
        """
        保存音频片段和元数据
        
        Args:
            segments: [(audio_array, metadata), ...]
            evaluation_results: 评估结果列表
            output_dir: 输出目录
            audio_id: 音频ID
            speaker_id: 说话人ID
            
        Returns:
            保存结果统计
        """
        # 创建输出目录结构
        speaker_output_dir = Path(output_dir) / audio_id / speaker_id
        speaker_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        metadata_files = []
        
        # 生成唯一的文件名前缀，包含时间戳和进程信息
        import time
        import os
        import uuid
        
        # 获取进程ID和时间戳用于生成唯一前缀
        process_id = getattr(self.config, '_process_id', os.getpid())
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        unique_prefix = f"{timestamp}_{process_id}"
        
        segment_counter = 1
        for (audio_array, metadata), result in zip(segments, evaluation_results):
            if result.passed:
                try:
                    import soundfile as sf
                    
                    # 生成唯一的文件名：时间戳_进程id_计数器
                    dst_filename = f"segment_{unique_prefix}_{segment_counter:03d}.wav"
                    dst_path = speaker_output_dir / dst_filename
                    
                    # 如果文件已存在，添加随机后缀（极少发生）
                    if dst_path.exists():
                        random_suffix = str(uuid.uuid4())[:8]
                        dst_filename = f"segment_{unique_prefix}_{segment_counter:03d}_{random_suffix}.wav"
                        dst_path = speaker_output_dir / dst_filename
                    
                    sf.write(str(dst_path), audio_array, metadata.get('sample_rate', self.config.processing.sample_rate))
                    saved_files.append(str(dst_path))
                    
                    # 保存完整元数据
                    full_metadata = {
                        'segment_id': f"{unique_prefix}_{segment_counter:03d}",
                        'segment_counter': segment_counter,
                        'process_id': process_id,
                        'timestamp': timestamp,
                        'saved_path': str(dst_path),
                        'audio_id': audio_id,
                        'speaker_id': speaker_id,
                        'original_metadata': metadata,
                        'transcription': result.transcription,
                        'quality_scores': result.quality_scores,
                        'evaluation_passed': True,
                        'processing_timestamp': self._get_timestamp()
                    }
                    
                    # 清理元数据，确保可以JSON序列化
                    clean_full_metadata = self._clean_for_json_serialization(full_metadata)
                    
                    metadata_filename = f"segment_{unique_prefix}_{segment_counter:03d}.json"
                    metadata_path = speaker_output_dir / metadata_filename
                    
                    # 如果元数据文件已存在，使用同样的随机后缀
                    if metadata_path.exists():
                        if 'random_suffix' in locals():
                            metadata_filename = f"segment_{unique_prefix}_{segment_counter:03d}_{random_suffix}.json"
                            metadata_path = speaker_output_dir / metadata_filename
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_full_metadata, f, ensure_ascii=False, indent=2)
                    
                    metadata_files.append(str(metadata_path))
                    segment_counter += 1
                    
                except Exception as e:
                    logger.error(f"保存音频片段失败: {e}")
        
        summary = {
            'speaker_id': speaker_id,
            'total_segments': len(evaluation_results),
            'passed_segments': len(saved_files),
            'saved_files': saved_files,
            'metadata_files': metadata_files,
            'output_directory': str(speaker_output_dir),
            'pass_rate': len(saved_files) / len(evaluation_results) if evaluation_results else 0.0,
            'unique_prefix': unique_prefix,  # 添加唯一前缀信息
            'process_id': process_id
        }
        
        return summary
    
    def _clean_for_json_serialization(self, data):
        """清理数据使其可以JSON序列化"""
        import math
        
        if isinstance(data, dict):
            return {k: self._clean_for_json_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json_serialization(item) for item in data]
        elif isinstance(data, float):
            # 处理NaN和无穷大
            if math.isnan(data) or math.isinf(data):
                return None
            return data
        elif data is None or isinstance(data, (str, int, bool)):
            return data
        else:
            # 对于其他不可序列化的类型，转换为字符串
            try:
                return str(data)
            except:
                return None

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat() 