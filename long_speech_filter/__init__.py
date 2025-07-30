"""
长音频处理模块

一个用于处理长音频文件的完整流程，包含：
- 说话人分离（使用ten-vad + pyannote-audio）
- 音频分割
- 质量筛选（whisper + dnsmos + dnsmospro + distilmos）
- 结构化存储

主要功能:
1. 使用ten-vad和pyannote-audio进行说话人聚类
2. 基于说话人信息分割音频
3. 对每个音频片段进行多维度质量评估
4. 按照 长音频id/说话人id/句子id 的结构存储通过筛选的音频
5. 为每条音频保存完整的元数据信息

使用方法:
    from long_speech_filter import LongAudioProcessor, LongAudioProcessingConfig
    
    config = LongAudioProcessingConfig()
    processor = LongAudioProcessor(config)
    stats = processor.process_directory()
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

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    # 配置类
    'LongAudioProcessingConfig',
    'LongAudioVADConfig', 
    'SpeakerDiarizationConfig',
    'QualityFilterConfig',
    'WhisperConfig',
    'ProcessingConfig',
    
    # 核心处理类
    'LongAudioProcessor',
    'LongAudioSpeakerDiarizer',
    'LongAudioQualityFilter',
    
    # 结果类
    'ProcessingResult',
    'SpeakerSegment',
    'SpeakerDiarizationResult',
    'AudioSegmentQuality',
] 