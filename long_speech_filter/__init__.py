"""
长音频处理器 (Long Audio Processor)

一个基于深度学习的长音频处理系统，支持：
- 说话人分离 (ten-vad + pyannote-audio)
- 音频分割和质量筛选 (Whisper + MOS评估)
- 多GPU并行处理和显存管理优化
- 结构化存储 (长音频id/说话人id/句子id)

使用方法:
    from long_speech_filter import LongAudioProcessingConfig, MultiGPULongAudioProcessor
    
    config = LongAudioProcessingConfig()
    processor = MultiGPULongAudioProcessor(config)
    results = processor.process_directory_parallel()
"""

from .config import (
    LongAudioProcessingConfig,
    LongAudioVADConfig,
    SpeakerDiarizationConfig,
    QualityFilterConfig,
    WhisperConfig,
    ProcessingConfig
)

from .speaker_diarization import (
    LongAudioSpeakerDiarizer,
    SpeakerSegment,
    SpeakerDiarizationResult
)

from .quality_filter import (
    LongAudioQualityFilter,
    AudioSegmentQuality
)

from .long_audio_processor import (
    LongAudioProcessor,
    ProcessingResult
)

from .multi_gpu_processor import (
    MultiGPULongAudioProcessor,
    MultiGPUConfig
)

__version__ = "2.0.0"
__author__ = "DataFilter Team"

__all__ = [
    # 配置类
    'LongAudioProcessingConfig',
    'LongAudioVADConfig', 
    'SpeakerDiarizationConfig',
    'QualityFilterConfig',
    'WhisperConfig',
    'ProcessingConfig',
    'MultiGPUConfig',
    
    # 核心处理类
    'LongAudioProcessor',
    'MultiGPULongAudioProcessor',
    'LongAudioSpeakerDiarizer',
    'LongAudioQualityFilter',
    
    # 结果类
    'ProcessingResult',
    'SpeakerSegment',
    'SpeakerDiarizationResult',
    'AudioSegmentQuality',
] 