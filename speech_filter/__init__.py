"""
语音筛选Pipeline包

一个基于多AI模型的语音筛选工具，用于从大量音频文件中筛选出高质量的语音数据。

主要模块:
- pipeline: 核心处理流程
- config: 配置管理
- vad_detector: VAD检测
- speech_recognizer: 语音识别
- audio_quality: 音质评估
- utils: 工具函数
"""

# 导入主要类和函数
from .pipeline import SpeechFilterPipeline, ProcessingResult
from .config import PipelineConfig, VADConfig, ASRConfig, AudioQualityConfig, ProcessingConfig
from .vad_detector import VADDetector, VADResult
from .speech_recognizer import SpeechRecognizer, ASRResult
from .audio_quality_assessor import AudioQualityAssessor, AudioQualityResult
from .utils import setup_logging, format_duration, format_file_size

# 定义公共API
__all__ = [
    # 核心类
    'SpeechFilterPipeline',
    'ProcessingResult',
    
    # 配置类
    'PipelineConfig',
    'VADConfig', 
    'ASRConfig',
    'AudioQualityConfig',
    'ProcessingConfig',
    
    # 检测器类
    'VADDetector',
    'VADResult',
    'SpeechRecognizer',
    'ASRResult',
    'AudioQualityAssessor',
    'AudioQualityResult',
    
    # 工具函数
    'setup_logging',
    'format_duration',
    'format_file_size',
] 