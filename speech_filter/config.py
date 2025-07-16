"""
语音筛选Pipeline配置文件
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class VADConfig:
    """VAD检测配置"""
    threshold: float = 0.5                 # TEN VAD阈值
    hop_size: int = 256                    # 跳跃大小（样本数，默认256=16ms at 16kHz）
    min_speech_duration: float = 0.5       # 最短语音时长（秒）
    max_speech_duration: float = 30.0      # 最长语音时长（秒）
    min_silence_duration: float = 0.1      # 最小静音持续时间（秒）
    
@dataclass
class ASRConfig:
    """语音识别配置"""
    model_name: str = "openai/whisper-large-v3"
    language: Optional[str] = None  # None表示自动检测，可选: "zh", "en", "ja"
    batch_size: int = 16
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    min_words: int = 1  # 最少词数要求
    model_cache_dir: str = "/root/data/pretrained_models"  # 模型缓存目录
    
@dataclass
class AudioQualityConfig:
    """音质评估配置"""
    distil_mos_threshold: float = 3.0    # DistilMos评分阈值
    dnsmos_threshold: float = 3.0        # DNSMOS评分阈值
    dnsmospro_threshold: float = 3.0     # DNSMOSPro评分阈值
    use_distil_mos: bool = True          # 是否使用DistilMos
    use_dnsmos: bool = True              # 是否使用DNSMOS
    use_dnsmospro: bool = True           # 是否使用DNSMOSPro
    
@dataclass
class ProcessingConfig:
    """处理配置"""
    supported_formats: List[str] = None
    max_duration: float = 30.0           # 最大音频时长（秒）
    min_duration: float = 0.5            # 最小音频时长（秒）
    sample_rate: int = 16000             # 重采样率
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a']

@dataclass
class PipelineConfig:
    """Pipeline总配置"""
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio_quality: AudioQualityConfig = field(default_factory=AudioQualityConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # 输出配置
    output_dir: str = "filtered_audio"
    results_file: str = "processing_results.json"
    log_file: str = "pipeline.log"
    
    # 并行处理配置
    num_workers: int = 4
    batch_size: int = 8
    
    # 向后兼容的属性访问
    @property
    def sample_rate(self):
        return self.processing.sample_rate
    
    @property
    def supported_formats(self):
        return self.processing.supported_formats

# 向后兼容的别名
Config = PipelineConfig


def load_config_from_yaml(config_path: str = None, language: str = None) -> PipelineConfig:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        language: 特定语言配置（如'japanese', 'chinese', 'english'）
        
    Returns:
        PipelineConfig对象
    """
    # 确定配置文件路径
    if config_path is None:
        # 默认配置文件路径
        current_dir = Path(__file__).parent
        config_path = current_dir / "config.yaml"
    
    config_path = Path(config_path)
    
    # 检查配置文件是否存在
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("使用默认配置")
        return PipelineConfig()
    
    try:
        # 加载YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 创建配置对象
        config = PipelineConfig()
        
        # 加载基础配置
        if 'vad' in yaml_config:
            vad_config = yaml_config['vad']
            config.vad.threshold = vad_config.get('threshold', config.vad.threshold)
            config.vad.hop_size = vad_config.get('hop_size', config.vad.hop_size)
            config.vad.min_speech_duration = vad_config.get('min_speech_duration', config.vad.min_speech_duration)
            config.vad.max_speech_duration = vad_config.get('max_speech_duration', config.vad.max_speech_duration)
            config.vad.min_silence_duration = vad_config.get('min_silence_duration', config.vad.min_silence_duration)
        
        if 'asr' in yaml_config:
            asr_config = yaml_config['asr']
            config.asr.model_name = asr_config.get('model_name', config.asr.model_name)
            config.asr.language = asr_config.get('language', config.asr.language)
            config.asr.batch_size = asr_config.get('batch_size', config.asr.batch_size)
            config.asr.device = asr_config.get('device', config.asr.device)
            config.asr.min_words = asr_config.get('min_words', config.asr.min_words)
            config.asr.model_cache_dir = asr_config.get('model_cache_dir', config.asr.model_cache_dir)
        
        if 'audio_quality' in yaml_config:
            aq_config = yaml_config['audio_quality']
            config.audio_quality.distil_mos_threshold = aq_config.get('distil_mos_threshold', config.audio_quality.distil_mos_threshold)
            config.audio_quality.dnsmos_threshold = aq_config.get('dnsmos_threshold', config.audio_quality.dnsmos_threshold)
            config.audio_quality.dnsmospro_threshold = aq_config.get('dnsmospro_threshold', config.audio_quality.dnsmospro_threshold)
            config.audio_quality.use_distil_mos = aq_config.get('use_distil_mos', config.audio_quality.use_distil_mos)
            config.audio_quality.use_dnsmos = aq_config.get('use_dnsmos', config.audio_quality.use_dnsmos)
            config.audio_quality.use_dnsmospro = aq_config.get('use_dnsmospro', config.audio_quality.use_dnsmospro)
        
        if 'processing' in yaml_config:
            proc_config = yaml_config['processing']
            config.processing.supported_formats = proc_config.get('supported_formats', config.processing.supported_formats)
            config.processing.max_duration = proc_config.get('max_duration', config.processing.max_duration)
            config.processing.min_duration = proc_config.get('min_duration', config.processing.min_duration)
            config.processing.sample_rate = proc_config.get('sample_rate', config.processing.sample_rate)
        
        if 'output' in yaml_config:
            output_config = yaml_config['output']
            config.output_dir = output_config.get('output_dir', config.output_dir)
            config.results_file = output_config.get('results_file', config.results_file)
            config.log_file = output_config.get('log_file', config.log_file)
        
        if 'parallel' in yaml_config:
            parallel_config = yaml_config['parallel']
            config.num_workers = parallel_config.get('num_workers', config.num_workers)
            config.batch_size = parallel_config.get('batch_size', config.batch_size)
        
        # 如果指定了语言，应用特定语言配置
        if language and 'language_configs' in yaml_config:
            lang_configs = yaml_config['language_configs']
            if language in lang_configs:
                lang_config = lang_configs[language]
                print(f"应用 {language} 语言特定配置")
                
                # 应用语言特定的VAD配置
                if 'vad' in lang_config:
                    vad_config = lang_config['vad']
                    config.vad.threshold = vad_config.get('threshold', config.vad.threshold)
                    config.vad.hop_size = vad_config.get('hop_size', config.vad.hop_size)
                    config.vad.min_speech_duration = vad_config.get('min_speech_duration', config.vad.min_speech_duration)
                    config.vad.max_speech_duration = vad_config.get('max_speech_duration', config.vad.max_speech_duration)
                
                # 应用语言特定的ASR配置
                if 'asr' in lang_config:
                    asr_config = lang_config['asr']
                    config.asr.model_name = asr_config.get('model_name', config.asr.model_name)
                    config.asr.language = asr_config.get('language', config.asr.language)
                    config.asr.batch_size = asr_config.get('batch_size', config.asr.batch_size)
                    config.asr.min_words = asr_config.get('min_words', config.asr.min_words)
                    config.asr.model_cache_dir = asr_config.get('model_cache_dir', config.asr.model_cache_dir)
                
                # 应用语言特定的音质评估配置
                if 'audio_quality' in lang_config:
                    aq_config = lang_config['audio_quality']
                    config.audio_quality.distil_mos_threshold = aq_config.get('distil_mos_threshold', config.audio_quality.distil_mos_threshold)
                    config.audio_quality.dnsmos_threshold = aq_config.get('dnsmos_threshold', config.audio_quality.dnsmos_threshold)
                    config.audio_quality.dnsmospro_threshold = aq_config.get('dnsmospro_threshold', config.audio_quality.dnsmospro_threshold)
                    config.audio_quality.use_distil_mos = aq_config.get('use_distil_mos', config.audio_quality.use_distil_mos)
                    config.audio_quality.use_dnsmos = aq_config.get('use_dnsmos', config.audio_quality.use_dnsmos)
                    config.audio_quality.use_dnsmospro = aq_config.get('use_dnsmospro', config.audio_quality.use_dnsmospro)
            else:
                print(f"警告：未找到语言 '{language}' 的特定配置")
        
        print(f"配置文件加载成功: {config_path}")
        return config
        
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置")
        return PipelineConfig()


def save_config_to_yaml(config: PipelineConfig, config_path: str):
    """
    将配置保存到YAML文件
    
    Args:
        config: PipelineConfig对象
        config_path: 配置文件路径
    """
    config_dict = {
        'vad': {
            'threshold': config.vad.threshold,
            'hop_size': config.vad.hop_size,
            'min_speech_duration': config.vad.min_speech_duration,
            'max_speech_duration': config.vad.max_speech_duration,
            'min_silence_duration': config.vad.min_silence_duration
        },
        'asr': {
            'model_name': config.asr.model_name,
            'language': config.asr.language,
            'batch_size': config.asr.batch_size,
            'device': config.asr.device,
            'min_words': config.asr.min_words,
            'model_cache_dir': config.asr.model_cache_dir
        },
        'audio_quality': {
            'distil_mos_threshold': config.audio_quality.distil_mos_threshold,
            'dnsmos_threshold': config.audio_quality.dnsmos_threshold,
            'dnsmospro_threshold': config.audio_quality.dnsmospro_threshold,
            'use_distil_mos': config.audio_quality.use_distil_mos,
            'use_dnsmos': config.audio_quality.use_dnsmos,
            'use_dnsmospro': config.audio_quality.use_dnsmospro
        },
        'processing': {
            'supported_formats': config.processing.supported_formats,
            'max_duration': config.processing.max_duration,
            'min_duration': config.processing.min_duration,
            'sample_rate': config.processing.sample_rate
        },
        'output': {
            'output_dir': config.output_dir,
            'results_file': config.results_file,
            'log_file': config.log_file
        },
        'parallel': {
            'num_workers': config.num_workers,
            'batch_size': config.batch_size
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        print(f"配置文件保存成功: {config_path}")
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def merge_cli_args_with_config(config: PipelineConfig, args) -> PipelineConfig:
    """
    将命令行参数与配置文件合并，命令行参数优先
    
    Args:
        config: 从配置文件加载的配置
        args: 命令行参数对象
        
    Returns:
        合并后的配置对象
    """
    # VAD配置
    if hasattr(args, 'vad_threshold') and args.vad_threshold is not None:
        config.vad.threshold = args.vad_threshold
    if hasattr(args, 'vad_hop_size') and args.vad_hop_size is not None:
        config.vad.hop_size = args.vad_hop_size
    if hasattr(args, 'min_speech_duration') and args.min_speech_duration is not None:
        config.vad.min_speech_duration = args.min_speech_duration
    if hasattr(args, 'max_speech_duration') and args.max_speech_duration is not None:
        config.vad.max_speech_duration = args.max_speech_duration
    
    # ASR配置
    if hasattr(args, 'whisper_model') and args.whisper_model is not None:
        config.asr.model_name = args.whisper_model
    if hasattr(args, 'language') and args.language is not None:
        config.asr.language = args.language
    if hasattr(args, 'min_words') and args.min_words is not None:
        config.asr.min_words = args.min_words
    if hasattr(args, 'model_cache_dir') and args.model_cache_dir is not None:
        config.asr.model_cache_dir = args.model_cache_dir
    
    # 音质评估配置
    if hasattr(args, 'distilmos_threshold') and args.distilmos_threshold is not None:
        config.audio_quality.distil_mos_threshold = args.distilmos_threshold
    if hasattr(args, 'dnsmos_threshold') and args.dnsmos_threshold is not None:
        config.audio_quality.dnsmos_threshold = args.dnsmos_threshold
    if hasattr(args, 'dnsmospro_threshold') and args.dnsmospro_threshold is not None:
        config.audio_quality.dnsmospro_threshold = args.dnsmospro_threshold
    if hasattr(args, 'disable_distilmos') and args.disable_distilmos:
        config.audio_quality.use_distil_mos = False
    if hasattr(args, 'disable_dnsmos') and args.disable_dnsmos:
        config.audio_quality.use_dnsmos = False
    if hasattr(args, 'disable_dnsmospro') and args.disable_dnsmospro:
        config.audio_quality.use_dnsmospro = False
    
    # 处理配置
    if hasattr(args, 'sample_rate') and args.sample_rate is not None:
        config.processing.sample_rate = args.sample_rate
    if hasattr(args, 'formats') and args.formats is not None:
        config.processing.supported_formats = [fmt.lower() for fmt in args.formats]
    
    # 并行配置
    if hasattr(args, 'workers') and args.workers is not None:
        config.num_workers = args.workers
    
    # 输出配置
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        config.output_dir = args.output_dir
    if hasattr(args, 'results_file') and args.results_file is not None:
        config.results_file = args.results_file
    if hasattr(args, 'log_file') and args.log_file is not None:
        config.log_file = args.log_file
    
    return config 