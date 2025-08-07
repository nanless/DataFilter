"""
长音频处理配置模块
定义所有配置参数
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class LongAudioVADConfig:
    """VAD检测配置"""
    threshold: float = 0.5
    hop_size: int = 256
    min_speech_duration: float = 0.5
    max_speech_duration: float = 30.0
    min_silence_duration: float = 0.1
    use_ten_vad: bool = True

@dataclass
class SpeakerDiarizationConfig:
    """说话人分离配置"""
    use_pyannote: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    min_segment_duration: float = 1.0
    auth_token: Optional[str] = None
    # 本地模型路径配置
    use_local_models: bool = True
    local_model_path: str = "pyannote"  # 相对于当前目录的路径
    diarization_model: str = "speaker-diarization-3.1"
    segmentation_model: str = "segmentation-3.0"
    embedding_model: str = "wespeaker-voxceleb-resnet34-LM"

@dataclass
class QualityFilterConfig:
    """质量筛选配置"""
    require_text: bool = True
    min_words: int = 1
    distil_mos_threshold: float = 3.0
    dnsmos_threshold: float = 3.0
    dnsmospro_threshold: float = 3.0
    use_distil_mos: bool = True
    use_dnsmos: bool = True
    use_dnsmospro: bool = True

@dataclass
class WhisperConfig:
    """Whisper配置"""
    model_name: str = "large-v3"  # 修复Whisper模型名称
    language: Optional[str] = None  # None表示自动检测  
    batch_size: int = 16
    device: str = "cuda"
    model_cache_dir: str = "/root/data/pretrained_models"

@dataclass
class ProcessingConfig:
    """处理配置"""
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.flac', '.m4a'])
    sample_rate: int = 16000
    max_workers: int = 4
    temp_dir: str = "temp"
    skip_processed: bool = True  # 默认跳过已处理的文件
    force_reprocess: bool = False  # 强制重新处理，即使已存在结果

@dataclass
class LongAudioProcessingConfig:
    """长音频处理主配置"""
    input_dir: str = "/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced"
    output_dir: str = "/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced_filtered"
    
    vad: LongAudioVADConfig = field(default_factory=LongAudioVADConfig)
    speaker_diarization: SpeakerDiarizationConfig = field(default_factory=SpeakerDiarizationConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    log_level: str = "INFO"
    log_file: str = "long_audio_processing.log"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LongAudioProcessingConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'vad': self.vad.__dict__,
            'speaker_diarization': self.speaker_diarization.__dict__,
            'quality_filter': self.quality_filter.__dict__,
            'whisper': self.whisper.__dict__,
            'processing': self.processing.__dict__,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def get_local_model_path(self, model_name: str) -> str:
        """获取本地模型的完整路径"""
        import os
        from pathlib import Path
        
        base_path = Path(__file__).parent / self.speaker_diarization.local_model_path
        model_path = base_path / model_name
        return str(model_path.absolute()) 