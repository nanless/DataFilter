"""
长音频说话人分离模块
整合ten-vad和pyannote-audio进行说话人聚类和音频分割
基于现有speech_filter框架和speaker_counter实现
"""
import os
import sys
import numpy as np
import librosa
import torch
import torchaudio
import soundfile as sf
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from contextlib import contextmanager

# 添加speech_filter路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_filter'))

@contextmanager
def change_dir(target_dir: str):
    """
    上下文管理器，临时切换工作目录

    Args:
        target_dir: 目标目录路径
    """
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)

# 导入ten-vad
try:
    from ten_vad import TenVad
except ImportError:
    print("警告: 未找到TEN VAD，请安装ten-vad包")
    TenVad = None

# 导入pyannote.audio
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    import pyannote.audio
except ImportError:
    print("警告: 未找到pyannote.audio，请安装pyannote-audio包")
    Pipeline = None
    Annotation = None
    Segment = None

# 导入现有的VAD检测器
try:
    from vad_detector import VADDetector, VADResult
except ImportError:
    print("警告: 无法导入VAD检测器，请确保speech_filter模块可用")
    VADDetector = None

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """说话人片段信息"""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    audio_segment: Optional[np.ndarray] = None

@dataclass
class SpeakerDiarizationResult:
    """说话人分离结果"""
    segments: List[SpeakerSegment]
    speaker_count: int
    audio_duration: float
    total_speech_time: float
    speech_ratio: float
    speaker_durations: Dict[str, float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.speaker_durations is None:
            self.speaker_durations = {}
            for segment in self.segments:
                if segment.speaker_id not in self.speaker_durations:
                    self.speaker_durations[segment.speaker_id] = 0.0
                self.speaker_durations[segment.speaker_id] += segment.duration

class LongAudioSpeakerDiarizer:
    """长音频说话人分离器"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.processing.sample_rate
        self.min_segment_duration = config.speaker_diarization.min_segment_duration
        self.auth_token = config.speaker_diarization.auth_token
        self.device = self._get_device(config.whisper.device)
        
        # VAD配置
        self.vad_threshold = config.vad.threshold
        self.vad_hop_size = config.vad.hop_size
        self.min_speech_duration = config.vad.min_speech_duration
        self.use_ten_vad = config.vad.use_ten_vad
        
        # 初始化TEN VAD
        self.ten_vad = None
        if TenVad and self.use_ten_vad:
            try:
                self.ten_vad = self._create_ten_vad(self.sample_rate)
                logger.info("成功初始化TEN VAD")
            except Exception as e:
                logger.error(f"初始化TEN VAD失败: {e}")
        
        # 初始化pyannote pipeline
        self.diarization_pipeline = None
        if Pipeline and config.speaker_diarization.use_pyannote:
            try:
                logger.info(f"开始加载PyAnnote管道到设备: {self.device}")
                self._load_pipeline()
                if self.diarization_pipeline:
                    logger.info(f"✅ 成功初始化pyannote说话人分离管道到设备: {self.device}")
                else:
                    logger.warning(f"⚠️ PyAnnote管道加载后为None，使用fallback模式")
            except Exception as e:
                logger.error(f"❌ 初始化pyannote分离管道失败 (设备: {self.device}): {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                
        # 如果有现有的VAD检测器可用，也初始化它作为备用
        self.vad_detector = None
        if VADDetector:
            try:
                # 创建兼容的配置对象
                vad_config = self._create_vad_config()
                self.vad_detector = VADDetector(vad_config)
                logger.info("成功初始化现有VAD检测器作为备用")
            except Exception as e:
                logger.warning(f"初始化现有VAD检测器失败: {e}")
    
    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _create_ten_vad(self, sample_rate: int) -> TenVad:
        """创建TEN VAD实例"""
        # 参考speaker_counter.py的方法计算hop_size
        frame_size_ms = 16  # 默认帧大小16ms
        hop_size = int(frame_size_ms * sample_rate / 1000)
        return TenVad(hop_size=hop_size, threshold=self.vad_threshold)
    
    def _create_vad_config(self):
        """创建VAD配置对象"""
        class VADConfig:
            def __init__(self, config):
                self.sample_rate = config.processing.sample_rate
                self.vad = config.vad
        return VADConfig(self.config)
    
    def _load_pipeline(self):
        """加载pyannote说话人分离管道"""
        try:
            # 优先尝试从本地加载模型
            if self.config.speaker_diarization.use_local_models:
                local_model_path = self.config.get_local_model_path(
                    self.config.speaker_diarization.diarization_model
                )
                
                if os.path.exists(local_model_path):
                    config_path = os.path.join(local_model_path, "config.yaml")
                    if os.path.exists(config_path):
                        logger.info(f"从本地配置文件加载pyannote模型: {config_path}")
                        
                        # 使用工作目录切换的方式加载本地模型（参考speaker_counter.py）
                        # 切换到long_speech_filter目录，这样相对路径能正确解析
                        base_dir = os.path.dirname(__file__)  # long_speech_filter目录
                        try:
                            with change_dir(base_dir):
                                relative_config_path = os.path.relpath(config_path, base_dir)
                                
                                if self.auth_token:
                                    self.diarization_pipeline = Pipeline.from_pretrained(
                                        relative_config_path,
                                        use_auth_token=self.auth_token
                                    )
                                else:
                                    # 尝试不使用token加载
                                    try:
                                        self.diarization_pipeline = Pipeline.from_pretrained(relative_config_path)
                                    except Exception as e:
                                        logger.warning(f"无token加载失败: {e}")
                                        logger.warning("本地配置文件可能引用HF Hub模型，需要认证token")
                                        self._create_fallback_pipeline()
                                        return
                                
                                logger.info("成功从本地配置文件创建pyannote说话人分离管道")
                        except Exception as e:
                            logger.warning(f"使用本地配置文件创建pipeline失败: {e}")
                            self._create_fallback_pipeline()
                            return
                    else:
                        logger.warning(f"本地模型目录缺少config.yaml: {local_model_path}")
                        self._create_fallback_pipeline()
                        return
                else:
                    logger.warning(f"本地模型路径不存在: {local_model_path}")
                    self._create_fallback_pipeline()
                    return
            else:
                # 配置为不使用本地模型，直接从HF Hub加载
                self._load_pipeline_from_hub()
            
            # 设置设备
            if self.diarization_pipeline and hasattr(self.diarization_pipeline, 'to'):
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    # 支持指定GPU ID，如 "cuda:0", "cuda:1" 等
                    device = torch.device(self.device)
                    self.diarization_pipeline.to(device)
                    logger.info(f"PyAnnote管道已移动到GPU: {self.device}")
                elif self.device == "cuda" and torch.cuda.is_available():
                    self.diarization_pipeline.to(torch.device("cuda"))
                    logger.info(f"PyAnnote管道已移动到GPU: {self.device}")
                else:
                    self.diarization_pipeline.to(torch.device("cpu"))
                    logger.info("PyAnnote管道已移动到CPU")
        
        except Exception as e:
            logger.error(f"加载pyannote模型失败: {e}")
            self._create_fallback_pipeline()
    
    def _load_pipeline_from_hub(self):
        """从Hugging Face Hub加载模型（需要token）"""
        if not self.auth_token:
            raise ValueError("从Hugging Face Hub加载pyannote模型需要认证token，请设置auth_token")
            
        logger.info("从Hugging Face Hub加载pyannote模型...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.auth_token
        )
    
    def _create_fallback_pipeline(self):
        """创建备用处理方案（不使用pyannote）"""
        logger.warning("创建备用处理方案，pyannote说话人分离功能将不可用")
        self.diarization_pipeline = None
    
    def _apply_ten_vad_preprocessing(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """应用TEN VAD预处理"""
        if not self.ten_vad:
            return audio
            
        try:
            # 确保音频是numpy数组
            if isinstance(audio, torch.Tensor):
                audio_np = audio.numpy()
            else:
                audio_np = audio
            
            # 如果是多通道，取第一个通道
            if len(audio_np.shape) > 1:
                audio_np = audio_np[0]
            
            # 使用TEN VAD检测语音段
            speech_segments = self.ten_vad(audio_np, sample_rate=sample_rate)
            
            # 创建语音标志数组
            speech_flags = np.zeros(len(audio_np), dtype=bool)
            for segment in speech_segments:
                start_idx = int(segment['start'])
                end_idx = int(segment['end'])
                speech_flags[start_idx:end_idx] = True
            
            # 应用语音掩码
            filtered_audio = audio_np * speech_flags
            
            return torch.FloatTensor(filtered_audio)
            
        except Exception as e:
            logger.warning(f"TEN VAD预处理失败，使用原始音频: {e}")
            return audio
    
    def detect_speech_activity(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        使用多种方法检测语音活动
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            语音片段列表 [(start, end), ...]
        """
        try:
            # 优先使用TEN VAD
            if self.ten_vad:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                # 使用TEN VAD检测语音片段
                try:
                    # TEN VAD需要int16类型的音频数据
                    # 将float32转换为int16（确保音频在 [-1, 1] 范围内）
                    audio_normalized = np.clip(audio, -1.0, 1.0)
                    audio_int16 = (audio_normalized * 32767).astype(np.int16)
                    
                    # 计算hop_size（参考speaker_counter.py的方法）
                    frame_size_ms = 16  # 默认帧大小16ms
                    hop_size = int(frame_size_ms * sr / 1000)
                    num_frames = len(audio_int16) // hop_size
                    
                    # 逐帧处理，获取语音标志
                    speech_flags = []
                    for i in range(num_frames):
                        audio_frame = audio_int16[i * hop_size: (i + 1) * hop_size]
                        if len(audio_frame) == hop_size:
                            # TEN VAD返回(probability, flag)元组
                            out_probability, out_flag = self.ten_vad.process(audio_frame)
                            speech_flags.append(out_flag)
                        else:
                            speech_flags.append(0)
                    
                    # 将语音标志转换为时间段
                    time_segments = []
                    current_start = None
                    
                    for i, flag in enumerate(speech_flags):
                        start_time = i * hop_size / sr
                        end_time = (i + 1) * hop_size / sr
                        
                        # 使用标志判断是否为语音
                        is_speech = flag > 0
                        
                        if is_speech and current_start is None:
                            # 开始语音段
                            current_start = start_time
                        elif not is_speech and current_start is not None:
                            # 结束语音段
                            if end_time - current_start >= self.min_speech_duration:
                                time_segments.append((current_start, end_time))
                            current_start = None
                    
                    # 处理最后一个语音段
                    if current_start is not None:
                        final_time = len(audio) / sr
                        if final_time - current_start >= self.min_speech_duration:
                            time_segments.append((current_start, final_time))
                    
                    logger.info(f"TEN VAD检测到 {len(time_segments)} 个语音片段")
                    return time_segments
                    
                except Exception as e:
                    logger.warning(f"TEN VAD检测出错，尝试简单方法: {e}")
                    # 如果标准方法失败，返回整个音频作为一个片段
                    duration = len(audio) / sr
                    if duration >= self.min_speech_duration:
                        return [(0.0, duration)]
                    else:
                        return []
            
            # 备用：使用现有VAD检测器
            elif self.vad_detector:
                segments = self.vad_detector.detect_speech_segments(audio_path)
                logger.info(f"备用VAD检测到 {len(segments)} 个语音片段")
                return segments
            
            else:
                logger.error("没有可用的VAD检测器")
                return []
                
        except Exception as e:
            logger.error(f"语音活动检测失败: {e}")
            return []
    
    def perform_speaker_diarization(self, audio_path: str) -> SpeakerDiarizationResult:
        """
        执行说话人分离
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            SpeakerDiarizationResult: 分离结果
        """
        if not self.diarization_pipeline:
            logger.warning("pyannote分离管道未初始化，返回单说话人结果")
            # 获取音频时长
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                audio_duration = len(audio) / sr
                
                # 创建单个说话人的片段（整个音频）
                single_segment = SpeakerSegment(
                    speaker_id="SPEAKER_00",
                    start_time=0.0,
                    end_time=audio_duration,
                    duration=audio_duration
                )
                
                return SpeakerDiarizationResult(
                    segments=[single_segment],
                    speaker_count=1,
                    audio_duration=audio_duration,
                    total_speech_time=audio_duration,
                    speech_ratio=1.0,
                    success=True
                )
            except Exception as e:
                return SpeakerDiarizationResult(
                    segments=[],
                    speaker_count=0,
                    audio_duration=0.0,
                    total_speech_time=0.0,
                    speech_ratio=0.0,
                    success=False,
                    error_message=f"无法处理音频文件: {e}"
                )
            
        try:
            logger.info(f"开始对音频文件进行说话人分离: {audio_path}")
            
            # 获取音频时长
            audio, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(audio) / sr
            
            # 执行说话人分离
            diarization = self.diarization_pipeline(audio_path)
            
            # 解析结果
            segments = []
            speaker_labels = set()
            total_speech_time = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_labels.add(speaker)
                
                # 过滤过短的片段
                if turn.duration >= self.min_segment_duration:
                    segment = SpeakerSegment(
                        speaker_id=speaker,
                        start_time=turn.start,
                        end_time=turn.end,
                        duration=turn.duration
                    )
                    segments.append(segment)
                    total_speech_time += turn.duration
            
            # 按时间排序
            segments.sort(key=lambda x: x.start_time)
            
            speech_ratio = total_speech_time / audio_duration if audio_duration > 0 else 0.0
            
            logger.info(f"说话人分离完成，检测到 {len(speaker_labels)} 个说话人，{len(segments)} 个片段")
            logger.info(f"音频时长: {audio_duration:.2f}s, 语音时长: {total_speech_time:.2f}s, 语音比例: {speech_ratio:.2%}")
            
            return SpeakerDiarizationResult(
                segments=segments,
                speaker_count=len(speaker_labels),
                audio_duration=audio_duration,
                total_speech_time=total_speech_time,
                speech_ratio=speech_ratio,
                success=True
            )
            
        except Exception as e:
            logger.error(f"说话人分离失败: {e}")
            return SpeakerDiarizationResult(
                segments=[],
                speaker_count=0,
                audio_duration=0.0,
                total_speech_time=0.0,
                speech_ratio=0.0,
                success=False,
                error_message=str(e)
            )
    
    def process_with_vad_and_diarization(self, audio_path: str) -> Dict[str, Any]:
        """
        结合VAD和说话人分离的综合处理
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            综合处理结果
        """
        results = {
            'audio_path': audio_path,
            'vad_segments': [],
            'speaker_segments': [],
            'success': False,
            'error_message': None
        }
        
        try:
            # 步骤1: VAD检测
            logger.info("步骤1: 进行VAD检测")
            vad_segments = self.detect_speech_activity(audio_path)
            results['vad_segments'] = vad_segments
            
            if not vad_segments:
                results['error_message'] = "VAD未检测到语音片段"
                return results
            
            # 步骤2: 说话人分离
            logger.info("步骤2: 进行说话人分离")
            diarization_result = self.perform_speaker_diarization(audio_path)
            
            if not diarization_result.success:
                results['error_message'] = diarization_result.error_message
                return results
            
            # 整合结果
            results['speaker_segments'] = [
                {
                    'speaker_id': seg.speaker_id,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration
                }
                for seg in diarization_result.segments
            ]
            results['speaker_count'] = diarization_result.speaker_count
            results['audio_duration'] = diarization_result.audio_duration
            results['total_speech_time'] = diarization_result.total_speech_time
            results['speech_ratio'] = diarization_result.speech_ratio
            results['speaker_durations'] = diarization_result.speaker_durations
            results['success'] = True
            
            logger.info(f"综合处理完成，VAD检测到 {len(vad_segments)} 个语音片段，"
                       f"说话人分离检测到 {diarization_result.speaker_count} 个说话人")
            
        except Exception as e:
            logger.error(f"综合处理失败: {e}")
            results['error_message'] = str(e)
            
        return results
        
    def split_audio_by_speakers(self, audio_path: str, speaker_segments: List[Dict], 
                               output_dir: str) -> Dict[str, List[str]]:
        """
        根据说话人信息分割音频
        
        Args:
            audio_path: 原始音频路径
            speaker_segments: 说话人片段信息
            output_dir: 输出目录
            
        Returns:
            每个说话人的音频文件路径列表 {speaker_id: [file_paths]}
        """
        try:
            # 加载原始音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 创建输出目录
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 按说话人分组
            speaker_files = {}
            segment_counter = {}
            
            for segment in speaker_segments:
                speaker_id = segment['speaker_id']
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # 在VAD时间戳前后各扩展0.5秒的静音
                padding_duration = 0.3  # 0.5秒
                
                # 计算扩展后的音频索引
                start_time_extended = max(0, start_time - padding_duration)
                end_time_extended = end_time + padding_duration
                
                start_sample = int(start_time_extended * sr)
                end_sample = int(end_time_extended * sr)
                
                # 确保索引在有效范围内
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                if start_sample >= end_sample:
                    logger.warning(f"跳过无效片段: {speaker_id} {start_time}-{end_time}")
                    continue
                
                # 提取音频片段
                audio_segment = audio[start_sample:end_sample]
                
                # 检查音频片段长度
                if len(audio_segment) < int(0.1 * sr):  # 至少0.1秒
                    logger.warning(f"跳过过短片段: {speaker_id} {start_time}-{end_time}")
                    continue
                
                # 生成文件名
                if speaker_id not in segment_counter:
                    segment_counter[speaker_id] = 0
                    speaker_files[speaker_id] = []
                
                segment_counter[speaker_id] += 1
                timestamp = f"{start_time:.2f}-{end_time:.2f}"
                filename = f"{speaker_id}_segment_{segment_counter[speaker_id]:03d}_{timestamp}.wav"
                output_path = os.path.join(output_dir, filename)
                
                # 保存音频片段
                sf.write(output_path, audio_segment, sr)
                speaker_files[speaker_id].append(output_path)
                
                logger.debug(f"保存音频片段: {output_path}")
                
            total_segments = sum(len(files) for files in speaker_files.values())
            logger.info(f"音频分割完成，生成了 {total_segments} 个片段，涉及 {len(speaker_files)} 个说话人")
            
            # 记录每个说话人的片段数量
            for speaker_id, files in speaker_files.items():
                logger.info(f"说话人 {speaker_id}: {len(files)} 个片段")
            
            return speaker_files
            
        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            return {}
    
    def extract_speaker_audio_segments(self, audio_path: str, speaker_segments: List[Dict]) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
        """
        提取说话人音频片段（返回音频数组而不是文件）
        
        Args:
            audio_path: 原始音频路径
            speaker_segments: 说话人片段信息
            
        Returns:
            {speaker_id: [(audio_array, metadata), ...]}
        """
        try:
            # 加载原始音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            speaker_audio_segments = {}
            
            for i, segment in enumerate(speaker_segments):
                speaker_id = segment['speaker_id']
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # 在VAD时间戳前后各扩展0.5秒的静音
                padding_duration = 0.3  # 0.5秒
                
                # 计算扩展后的音频索引
                start_time_extended = max(0, start_time - padding_duration)
                end_time_extended = end_time + padding_duration
                
                start_sample = int(start_time_extended * sr)
                end_sample = int(end_time_extended * sr)
                
                # 确保索引在有效范围内
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                if start_sample >= end_sample:
                    continue
                
                # 提取音频片段
                audio_segment = audio[start_sample:end_sample]
                
                # 检查音频片段长度
                if len(audio_segment) < int(0.1 * sr):  # 至少0.1秒
                    continue
                
                # 创建元数据，记录原始时间和扩展时间
                metadata = {
                    'segment_id': i,
                    'speaker_id': speaker_id,
                    'start_time': start_time,  # 原始VAD检测时间
                    'end_time': end_time,      # 原始VAD检测时间
                    'duration': end_time - start_time,  # 原始持续时间
                    'extended_start_time': start_time_extended,  # 扩展后开始时间
                    'extended_end_time': end_time_extended,      # 扩展后结束时间
                    'extended_duration': end_time_extended - start_time_extended,  # 扩展后持续时间
                    'padding_duration': padding_duration,        # 填充时长
                    'sample_rate': sr,
                    'original_audio': audio_path
                }
                
                # 存储到结果中
                if speaker_id not in speaker_audio_segments:
                    speaker_audio_segments[speaker_id] = []
                
                speaker_audio_segments[speaker_id].append((audio_segment, metadata))
            
            logger.info(f"提取了 {sum(len(segments) for segments in speaker_audio_segments.values())} 个音频片段")
            return speaker_audio_segments
            
        except Exception as e:
            logger.error(f"提取音频片段失败: {e}")
            return {} 