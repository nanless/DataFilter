"""
语音识别模块
使用whisper-large-v3模型进行语音转录
"""
import os
import torch
import whisper
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import tempfile
import shutil
import warnings
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import threading
import time
from contextlib import contextmanager

# 抑制警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass
class ASRResult:
    """语音识别结果"""
    text: str
    language: str
    word_count: int
    segments: list
    success: bool
    error_message: Optional[str] = None
    confidence: Optional[float] = None

class SpeechRecognizer:
    """语音识别器，基于Whisper模型"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.asr.model_name
        self.language = config.asr.language
        self.device = config.asr.device
        self.model_cache_dir = config.asr.model_cache_dir
        self.sample_rate = config.processing.sample_rate
        
        # 线程锁，确保模型加载和推理的线程安全
        self._model_lock = threading.Lock()
        self._model = None
        self._model_loading = False
        
        # 错误重试配置
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载Whisper模型"""
        with self._model_lock:
            if self._model is not None:
                return
            
            if self._model_loading:
                # 等待其他线程完成加载
                while self._model_loading:
                    time.sleep(0.1)
                return
            
            try:
                self._model_loading = True
                logger.info(f"正在加载Whisper模型: {self.model_name}")
                
                # 确保模型缓存目录存在
                whisper_cache_dir = os.path.join(self.model_cache_dir, 'whisper_modes')
                os.makedirs(whisper_cache_dir, exist_ok=True)
                
                # 加载模型
                # 在多进程环境中，确保使用正确的设备映射
                if torch.cuda.is_available():
                    # 如果CUDA可用，使用当前可见的第一个GPU
                    actual_device = "cuda:0"
                else:
                    actual_device = "cpu"
                
                self._model = whisper.load_model(
                    self.model_name,
                    device=actual_device,
                    download_root=whisper_cache_dir
                )
                
                # 设置模型为评估模式
                self._model.eval()
                
                logger.info(f"Whisper模型加载成功，缓存目录: {whisper_cache_dir}")
                
            except Exception as e:
                logger.error(f"Whisper模型加载失败: {str(e)}")
                self._model = None
                raise
            finally:
                self._model_loading = False
    
    def _reload_model(self):
        """重新加载模型"""
        with self._model_lock:
            logger.info("重新加载Whisper模型...")
            if self._model is not None:
                del self._model
                self._model = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._load_model()
    
    @contextmanager
    def _get_model(self):
        """获取模型的上下文管理器，确保线程安全"""
        with self._model_lock:
            if self._model is None:
                raise RuntimeError("模型未加载")
            yield self._model
    
    def _preprocess_audio(self, audio_path: str) -> Optional[str]:
        """
        预处理音频文件，转换为临时wav文件
        
        Args:
            audio_path: 原始音频文件路径
            
        Returns:
            临时wav文件路径，失败返回None
        """
        try:
            # 读取音频文件
            audio_data, sr = librosa.load(audio_path, sr=None)
            
            # 检查音频数据
            if len(audio_data) == 0:
                logger.warning(f"音频文件为空: {audio_path}")
                return None
            
            # 重采样到目标采样率
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # 音频归一化
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 检查音频长度
            duration = len(audio_data) / self.sample_rate
            if duration < 0.1:  # 至少0.1秒
                logger.warning(f"音频文件过短: {audio_path}, 时长: {duration:.2f}秒")
                return None
            
            # if duration > 30.0:  # 最多30秒
            #     logger.warning(f"音频文件过长，截取前30秒: {audio_path}")
            #     audio_data = audio_data[:30 * self.sample_rate]
            
            # 创建临时文件
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_audio.wav")
            
            # 保存到临时文件
            sf.write(temp_file, audio_data, self.sample_rate)
            
            return temp_file
            
        except Exception as e:
            logger.error(f"音频预处理失败: {audio_path}, 错误: {str(e)}")
            return None
    
    def _cleanup_temp_file(self, temp_file: str):
        """清理临时文件"""
        try:
            if temp_file and os.path.exists(temp_file):
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {temp_file}, 错误: {str(e)}")
    
    def _safe_transcribe(self, audio_path: str, attempt: int = 1) -> Dict[str, Any]:
        """
        安全的转录方法，包含错误处理和重试逻辑
        
        Args:
            audio_path: 音频文件路径
            attempt: 尝试次数
            
        Returns:
            转录结果字典
        """
        temp_file = None
        try:
            # 预处理音频
            temp_file = self._preprocess_audio(audio_path)
            if temp_file is None:
                return {
                    'text': '',
                    'language': self.language or 'unknown',
                    'success': False,
                    'error': '音频预处理失败'
                }
            
            # 使用模型进行转录
            with self._get_model() as model:
                # 设置转录选项
                options = {
                    'language': self.language,
                    'task': 'transcribe',
                    'fp16': False,  # 避免精度问题
                    'verbose': False
                }
                
                # 执行转录
                result = model.transcribe(temp_file, **options)
                
                # 提取文本
                text = result.get('text', '').strip()
                detected_language = result.get('language', self.language or 'unknown')
                segments = result.get('segments', [])
                
                # 计算词数
                word_count = len(text.split()) if text else 0
                
                logger.info(f"转录完成：{audio_path}，检测语言：{detected_language}，词数：{word_count}")
                
                return {
                    'text': text,
                    'language': detected_language,
                    'segments': segments,
                    'word_count': word_count,
                    'success': True
                }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"转录尝试 {attempt} 失败：{temp_file or audio_path}，错误：{error_msg}")
            
            # 检查是否是已知的多线程问题
            if any(keyword in error_msg for keyword in [
                'key.size(1) == value.size(1)',
                'Linear(in_features=',
                'cannot reshape tensor',
                'CUDA error',
                'RuntimeError'
            ]):
                logger.warning(f"转录尝试 {attempt} 失败：{audio_path}，错误：{error_msg}，正在重试...")
                
                # 检查错误类型并采取对应措施
                if 'key.size(1) == value.size(1)' in error_msg:
                    logger.info("检测到维度不匹配错误，尝试重新加载模型")
                    time.sleep(self.retry_delay)
                    self._reload_model()
                elif 'Linear(in_features=' in error_msg:
                    logger.info("检测到模型层错误，重新初始化模型")
                    time.sleep(self.retry_delay)
                    self._reload_model()
                elif 'cannot reshape tensor' in error_msg:
                    logger.info("检测到张量reshape错误，跳过此文件")
                    return {
                        'text': '',
                        'language': self.language or 'unknown',
                        'success': False,
                        'error': '张量reshape错误，音频数据可能损坏'
                    }
                elif 'CUDA' in error_msg and attempt == 1:
                    logger.info("检测到CUDA错误，清理缓存后重试")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(self.retry_delay)
                
                # 如果还有重试次数，递归重试
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # 递增延迟
                    return self._safe_transcribe(audio_path, attempt + 1)
            
            # 最终失败
            return {
                'text': '',
                'language': self.language or 'unknown',
                'success': False,
                'error': error_msg
            }
            
        finally:
            # 清理临时文件
            if temp_file:
                self._cleanup_temp_file(temp_file)
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录结果字典
        """
        return self._safe_transcribe(audio_path)
    
    def transcribe_audio_detailed(self, audio_path: str) -> ASRResult:
        """
        转录音频文件（返回详细结果）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            ASRResult对象
        """
        result = self.transcribe_audio(audio_path)
        
        return ASRResult(
            text=result.get("text", ""),
            language=result.get("language", "unknown"),
            word_count=result.get("word_count", 0),
            segments=result.get("segments", []),
            success=result.get("success", False),
            error_message=result.get("error", None)
        )
    
    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        转录音频数组
        
        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            
        Returns:
            转录结果字典
        """
        temp_file = None
        try:
            # 确保音频长度足够
            if len(audio_array) < sample_rate * 0.1:  # 至少100ms
                return {
                    "text": "",
                    "language": "unknown",
                    "word_count": 0,
                    "segments": [],
                    "success": False,
                    "error": "音频太短"
                }
            
            # 归一化音频
            audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # 创建临时文件
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_audio.wav")
            sf.write(temp_file, audio_array, sample_rate)
            
            # 使用模型进行转录
            with self._get_model() as model:
                # 设置转录选项
                options = {
                    "language": self.language,
                    "task": "transcribe",
                    "verbose": False,
                    "fp16": False
                }
                
                # 执行转录
                result = model.transcribe(temp_file, **options)
                
                # 提取文本
                text = result.get("text", "").strip()
                language = result.get("language", "unknown")
                word_count = len(text.split()) if text else 0
                
                return {
                    "text": text,
                    "language": language,
                    "word_count": word_count,
                    "segments": result.get("segments", []),
                    "success": True
                }
            
        except Exception as e:
            logger.error(f"转录音频数组失败：{str(e)}")
            return {
                "text": "",
                "language": "unknown",
                "word_count": 0,
                "segments": [],
                "success": False,
                "error": str(e)
            }
        finally:
            # 清理临时文件
            if temp_file:
                self._cleanup_temp_file(temp_file)
    
    def is_valid_transcription(self, transcription_result: Dict[str, Any]) -> bool:
        """
        判断转录结果是否有效
        
        Args:
            transcription_result: 转录结果
            
        Returns:
            是否有效
        """
        if not transcription_result.get("success", False):
            return False
        
        text = transcription_result.get("text", "").strip()
        word_count = transcription_result.get("word_count", 0)
        
        # 检查是否有文本内容
        if not text:
            return False
        
        # 检查词数是否达到最低要求（从配置中获取）
        min_words = getattr(self.config.asr, 'min_words', 1)
        if word_count < min_words:
            return False
        
        # 检查语言（如果指定了特定语言）
        if self.language:
            detected_language = transcription_result.get("language", "unknown")
            if detected_language != self.language:
                logger.debug(f"语言不匹配：期望{self.language}，检测到{detected_language}")
                return False
        
        return True
    
    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        return list(whisper.tokenizer.LANGUAGES.keys())
    
    def detect_language(self, audio_path: str) -> str:
        """
        检测音频语言
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            检测到的语言代码
        """
        try:
            with self._get_model() as model:
                # 加载音频
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                
                # 生成梅尔频谱
                mel = whisper.log_mel_spectrogram(audio).to(self.device)
                
                # 检测语言
                _, probs = model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                
                logger.info(f"检测到语言：{detected_language}，置信度：{probs[detected_language]:.2f}")
                return detected_language
            
        except Exception as e:
            logger.error(f"语言检测失败：{audio_path}，错误：{str(e)}")
            return "unknown" 