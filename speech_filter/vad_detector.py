"""
VAD (Voice Activity Detection) 检测模块
使用TEN VAD模型进行语音活动检测
"""
import numpy as np
import librosa
import torch
import torchaudio
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

# 导入TEN VAD类
try:
    from ten_vad import TenVad
except ImportError:
    print("Warning: TEN VAD not found. Please install ten-vad package.")
    TenVad = None

logger = logging.getLogger(__name__)

@dataclass
class VADResult:
    """VAD检测结果"""
    segments: List[Tuple[float, float]]
    total_voice_duration: float
    success: bool
    error_message: Optional[str] = None

class VADDetector:
    """语音活动检测器"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.threshold = config.vad.threshold
        self.min_speech_duration = config.vad.min_speech_duration
        self.max_speech_duration = config.vad.max_speech_duration
        self.hop_size = config.vad.hop_size
        
        # 检查TEN VAD是否可用
        if TenVad is None:
            raise ImportError("TEN VAD not available. Please install ten-vad package.")
            
    def detect_speech_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        检测音频中的语音段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 使用TEN VAD检测语音段
            timestamps, total_voice_duration = self._get_ten_vad_timestamps(y, sr)
            
            # 过滤时长不符合要求的段
            filtered_segments = []
            for start, end in timestamps:
                duration = end - start
                if self.min_speech_duration <= duration <= self.max_speech_duration:
                    filtered_segments.append((start, end))
            
            logger.info(f"VAD检测完成：{audio_path}，检测到{len(filtered_segments)}个语音段，总语音时长：{total_voice_duration:.2f}秒")
            return filtered_segments
            
        except Exception as e:
            logger.error(f"VAD检测失败：{audio_path}，错误：{str(e)}")
            return []
    
    def detect_speech_segments_detailed(self, audio_path: str) -> VADResult:
        """
        检测音频中的语音段（返回详细结果）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            VADResult对象
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 使用TEN VAD检测语音段
            timestamps, total_voice_duration = self._get_ten_vad_timestamps(y, sr)
            
            # 过滤时长不符合要求的段
            filtered_segments = []
            for start, end in timestamps:
                duration = end - start
                if self.min_speech_duration <= duration <= self.max_speech_duration:
                    filtered_segments.append((start, end))
            
            logger.info(f"VAD检测完成：{audio_path}，检测到{len(filtered_segments)}个语音段，总语音时长：{total_voice_duration:.2f}秒")
            
            return VADResult(
                segments=filtered_segments,
                total_voice_duration=total_voice_duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"VAD检测失败：{audio_path}，错误：{str(e)}")
            return VADResult(
                segments=[],
                total_voice_duration=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _get_ten_vad_timestamps(self, audio_data: np.ndarray, sr: int) -> Tuple[List[List[float]], float]:
        """
        使用TEN VAD获取语音活动检测时间戳
        
        Args:
            audio_data: 音频数据 (numpy array)
            sr: 采样率
            
        Returns:
            时间戳列表 [start, end] 和总语音活动时长
        """
        try:
            # 转换为int16格式用于TEN VAD
            data_int16 = (audio_data * 32768).astype(np.int16)
            
            # 创建TEN VAD实例
            ten_vad_instance = TenVad(self.hop_size, self.threshold)
            
            num_frames = data_int16.shape[0] // self.hop_size
            frame_duration = self.hop_size / sr  # 每帧的时间长度（秒）
            
            # 逐帧处理音频
            speech_flags = []
            for i in range(num_frames):
                audio_frame = data_int16[i * self.hop_size: (i + 1) * self.hop_size]
                out_probability, out_flag = ten_vad_instance.process(audio_frame)
                speech_flags.append(out_flag)

            # 将语音标志转换为时间戳
            timestamps = []
            total_voice_duration = 0.0
            
            # 查找连续的语音段
            in_speech = False
            start_frame = 0
            
            for i, flag in enumerate(speech_flags):
                if flag == 1 and not in_speech:  # 语音开始
                    in_speech = True
                    start_frame = i
                elif flag == 0 and in_speech:  # 语音结束
                    in_speech = False
                    start_time = start_frame * frame_duration
                    end_time = i * frame_duration
                    timestamps.append([round(start_time, 3), round(end_time, 3)])
                    total_voice_duration += (end_time - start_time)
            
            # 处理语音一直持续到结尾的情况
            if in_speech:
                start_time = start_frame * frame_duration
                end_time = len(speech_flags) * frame_duration
                timestamps.append([round(start_time, 3), round(end_time, 3)])
                total_voice_duration += (end_time - start_time)
            
            return timestamps, round(total_voice_duration, 3)
            
        except Exception as e:
            logger.error(f"TEN VAD处理错误：{e}")
            return [], 0.0
    
    def extract_speech_audio(self, audio_path: str, segments: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        根据VAD结果提取语音音频段
        
        Args:
            audio_path: 音频文件路径
            segments: 语音段时间列表
            
        Returns:
            提取的音频段列表
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            audio_segments = []
            for start_time, end_time in segments:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # 确保索引在有效范围内
                start_sample = max(0, start_sample)
                end_sample = min(len(y), end_sample)
                
                if start_sample < end_sample:
                    segment_audio = y[start_sample:end_sample]
                    audio_segments.append(segment_audio)
            
            return audio_segments
            
        except Exception as e:
            logger.error(f"提取语音段失败：{audio_path}，错误：{str(e)}")
            return []
            
    def is_speech(self, audio_path: str) -> bool:
        """
        判断音频是否包含语音
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            是否包含语音
        """
        segments = self.detect_speech_segments(audio_path)
        return len(segments) > 0 