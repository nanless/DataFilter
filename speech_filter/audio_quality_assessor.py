"""
音质评估模块
使用Microsoft官方的Distill-MOS和DNSMOS模型进行音质评分
"""
import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import warnings
import requests
from pathlib import Path
import tempfile
import shutil
from dnsmospro_utils import stft

# 抑制警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass
class AudioQualityResult:
    """音质评估结果"""
    scores: Dict[str, float]
    success: bool
    audio_path: Optional[str] = None
    error_message: Optional[str] = None
    overall_score: Optional[float] = None
    
    def __post_init__(self):
        if self.overall_score is None and self.scores:
            self.overall_score = self.scores.get('overall', 0.0)

class AudioQualityAssessor:
    """音质评估器，使用Microsoft官方的Distill-MOS和DNSMOS模型"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.distilmos_threshold = config.audio_quality.distil_mos_threshold
        self.dnsmos_threshold = config.audio_quality.dnsmos_threshold
        self.dnsmospro_threshold = config.audio_quality.dnsmospro_threshold
        self.enable_distilmos = config.audio_quality.use_distil_mos
        self.enable_dnsmos = config.audio_quality.use_dnsmos
        self.enable_dnsmospro = config.audio_quality.use_dnsmospro
        
        # 支持指定GPU设备
        self.device = getattr(config.audio_quality, 'device', None)
        
        # 模型缓存目录
        self.model_cache_dir = getattr(config.asr, 'model_cache_dir', '/root/data/pretrained_models')
        
        # 模型相关
        self.distilmos_model = None
        self.dnsmos_compute_score = None
        self.dnsmospro_model = None
        
        # 初始化模型
        self._initialize_models()
    
    def _get_safe_device(self):
        """获取在多进程环境中安全的设备"""
        # 如果指定了设备，使用指定的设备
        if self.device:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                return self.device
            elif self.device == "cpu":
                return "cpu"
        
        # 默认行为
        if torch.cuda.is_available():
            # 在多进程环境中，使用当前可见的第一个GPU
            return "cuda:0"
        else:
            return "cpu"
    
    def _initialize_models(self):
        """初始化音质评估模型"""
        try:
            if self.enable_distilmos:
                self._load_distilmos_model()
            
            if self.enable_dnsmos:
                self._load_dnsmos_model()
            
            if self.enable_dnsmospro:
                self._load_dnsmospro_model()
                
        except Exception as e:
            logger.error(f"音质评估模型初始化失败: {e}")
            logger.warning("将使用基于音频特征的简单音质评估")
    
    def _load_distilmos_model(self):
        """加载Distill-MOS模型"""
        try:
            import distillmos
            logger.info("正在加载Distill-MOS模型...")
            
            self.distilmos_model = distillmos.ConvTransformerSQAModel()
            self.distilmos_model.eval()
            
            # 获取设备信息
            device = self._get_safe_device()
            
            # 将模型移动到指定设备
            if device != 'cpu' and torch.cuda.is_available():
                self.distilmos_model = self.distilmos_model.to(device)
                logger.info(f"Distill-MOS模型已移动到GPU: {device}")
            else:
                self.distilmos_model = self.distilmos_model.to('cpu')
                logger.info("Distill-MOS模型已移动到CPU")
            
            logger.info("Distill-MOS模型加载成功")
            
        except ImportError:
            logger.error("Distill-MOS包未安装，请运行: pip install distillmos")
            self.distilmos_model = None
        except Exception as e:
            logger.error(f"Distill-MOS模型加载失败: {e}")
            self.distilmos_model = None
    
    def _load_dnsmos_model(self):
        """加载DNSMOS模型"""
        try:
            logger.info("正在加载DNSMOS模型...")
            
            # 模型文件路径
            model_dir = Path(self.model_cache_dir) / "dnsmos"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            primary_model_path = model_dir / "sig_bak_ovr.onnx"
            p808_model_path = model_dir / "model_v8.onnx"
            
            # 下载模型文件（如果不存在）
            if not primary_model_path.exists():
                logger.info("正在下载DNSMOS主模型...")
                self._download_dnsmos_model(
                    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
                    primary_model_path
                )
            
            if not p808_model_path.exists():
                logger.info("正在下载DNSMOS P808模型...")
                self._download_dnsmos_model(
                    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx",
                    p808_model_path
                )
            
            # 初始化DNSMOS计算器
            self.dnsmos_compute_score = DNSMOSComputeScore(str(primary_model_path), str(p808_model_path))
            
            logger.info(f"DNSMOS模型加载成功，模型目录: {model_dir}")
            
        except Exception as e:
            logger.error(f"DNSMOS模型加载失败: {e}")
            self.dnsmos_compute_score = None
    
    def _load_dnsmospro_model(self):
        """加载DNSMOSPro模型"""
        try:
            logger.info("正在加载DNSMOSPro模型...")
            
            # 模型文件路径
            model_dir = Path(self.model_cache_dir) / "dnsmospro"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model_best.pt"
            
            # 下载模型文件（如果不存在）
            if not model_path.exists():
                logger.info("正在下载DNSMOSPro模型...")
                self._download_dnsmospro_model(
                    "https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/NISQA/model_best.pt",
                    model_path
                )
            
            # 加载模型
            device = self._get_safe_device()
            
            # 根据设备选择加载位置
            if device != 'cpu' and torch.cuda.is_available():
                self.dnsmospro_model = torch.jit.load(str(model_path), map_location=torch.device(device))
                logger.info(f"DNSMOSPro模型已加载到GPU: {device}")
            else:
                self.dnsmospro_model = torch.jit.load(str(model_path), map_location=torch.device('cpu'))
                logger.info("DNSMOSPro模型已加载到CPU")
            
            # 设置模型为评估模式
            self.dnsmospro_model.eval()
            
            logger.info(f"DNSMOSPro模型加载成功，模型目录: {model_dir}")
            
        except Exception as e:
            logger.error(f"DNSMOSPro模型加载失败: {e}")
            self.dnsmospro_model = None
    
    def _download_dnsmos_model(self, url: str, path: Path):
        """下载DNSMOS模型文件"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"模型文件下载成功: {path}")
            
        except Exception as e:
            logger.error(f"模型文件下载失败: {url} -> {path}, 错误: {e}")
            raise
    
    def _download_dnsmospro_model(self, url: str, path: Path):
        """下载DNSMOSPro模型文件"""
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"DNSMOSPro模型文件下载成功: {path}")
            
        except Exception as e:
            logger.error(f"DNSMOSPro模型文件下载失败: {url} -> {path}, 错误: {e}")
            raise
    
    def assess_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        评估音频质量
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            质量评估结果字典
        """
        try:
            # 计算各种质量指标
            quality_metrics = {}
            
            # 尝试Distill-MOS评分
            if self.enable_distilmos and self.distilmos_model is not None:
                try:
                    distilmos_score = self._calculate_distilmos_score(audio_path)
                    quality_metrics['distilmos'] = distilmos_score
                except Exception as e:
                    logger.warning(f"Distill-MOS评分失败：{audio_path}，错误：{str(e)}")
            
            # 尝试DNSMOS评分
            if self.enable_dnsmos and self.dnsmos_compute_score is not None:
                try:
                    dnsmos_scores = self._calculate_dnsmos_scores(audio_path)
                    quality_metrics.update(dnsmos_scores)
                except Exception as e:
                    logger.warning(f"DNSMOS评分失败：{audio_path}，使用默认评分。错误：{str(e)}")
                    # DNSMOS失败时使用默认评分，但不中断处理
                    quality_metrics.update({
                        'dnsmos_ovrl': 2.0,
                        'dnsmos_sig': 2.0,
                        'dnsmos_bak': 2.0,
                        'dnsmos_p808': 2.0,
                        'dnsmos': 2.0
                    })
            
            # 尝试DNSMOSPro评分
            if self.enable_dnsmospro and self.dnsmospro_model is not None:
                try:
                    dnsmospro_score = self._calculate_dnsmospro_score(audio_path)
                    quality_metrics['dnsmospro'] = dnsmospro_score
                except Exception as e:
                    logger.warning(f"DNSMOSPro评分失败：{audio_path}，错误：{str(e)}")
            
            # 如果没有任何评分成功，使用基础音频特征评分
            if not quality_metrics:
                logger.warning(f"所有音质评估方法都失败，使用基础评分：{audio_path}")
                quality_metrics = self._calculate_basic_audio_score(audio_path)
            
            # 计算综合质量评分
            overall_score = self._calculate_overall_score(quality_metrics)
            quality_metrics['overall'] = overall_score
            
            logger.info(f"音质评估完成：{audio_path}，综合评分：{overall_score:.2f}")
            
            return {
                'scores': quality_metrics,
                'success': True,
                'audio_path': audio_path
            }
            
        except Exception as e:
            logger.error(f"音质评估失败：{audio_path}，错误：{str(e)}")
            return {
                'scores': {'overall': 2.0},  # 使用中等默认评分
                'success': False,
                'error': str(e),
                'audio_path': audio_path
            }
    
    def assess_audio_quality_detailed(self, audio_path: str) -> AudioQualityResult:
        """
        评估音频质量（返回详细结果）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            AudioQualityResult对象
        """
        result = self.assess_audio_quality(audio_path)
        
        return AudioQualityResult(
            scores=result.get('scores', {}),
            success=result.get('success', False),
            audio_path=result.get('audio_path'),
            error_message=result.get('error')
        )
    
    def _calculate_distilmos_score(self, audio_path: str) -> float:
        """
        使用Distill-MOS计算音质评分
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Distill-MOS评分 (1-5)
        """
        try:
            # 获取设备信息
            device = self._get_safe_device()
            
            # 加载音频文件
            x, sr = torchaudio.load(audio_path)
            
            # 如果是多通道，只使用第一个通道
            if x.shape[0] > 1:
                logger.debug(f"音频文件有多个通道，使用第一个通道: {audio_path}")
                x = x[0, None, :]
            
            # 重采样到16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                x = resampler(x)
            
            # 将音频数据移动到与模型相同的设备
            if device != 'cpu' and torch.cuda.is_available():
                x = x.to(device)
            else:
                x = x.to('cpu')
            
            # 使用模型预测
            with torch.no_grad():
                mos = self.distilmos_model(x)
            
            # 转换为标量
            if torch.is_tensor(mos):
                mos = mos.cpu().item()
            
            # 确保评分在1-5范围内
            mos = np.clip(mos, 1.0, 5.0)
            
            return float(mos)
            
        except Exception as e:
            logger.error(f"Distill-MOS评分计算失败：{audio_path}，错误：{str(e)}")
            return 1.0
    
    def _calculate_dnsmos_scores(self, audio_path: str) -> Dict[str, float]:
        """
        使用DNSMOS计算音质评分
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            DNSMOS评分字典
        """
        try:
            # 使用DNSMOS计算器
            result = self.dnsmos_compute_score(audio_path, 16000, is_personalized_MOS=False)
            
            # 提取相关评分
            dnsmos_scores = {
                'dnsmos_ovrl': result.get('OVRL', 1.0),
                'dnsmos_sig': result.get('SIG', 1.0),
                'dnsmos_bak': result.get('BAK', 1.0),
                'dnsmos_p808': result.get('P808_MOS', 1.0),
                'dnsmos': result.get('OVRL', 1.0)  # 使用OVRL作为主要DNSMOS分数
            }
            
            return dnsmos_scores
            
        except Exception as e:
            logger.error(f"DNSMOS评分计算失败：{audio_path}，错误：{str(e)}")
            return {
                'dnsmos_ovrl': 1.0,
                'dnsmos_sig': 1.0,
                'dnsmos_bak': 1.0,
                'dnsmos_p808': 1.0,
                'dnsmos': 1.0
            }
    
    def _calculate_dnsmospro_score(self, audio_path: str) -> float:
        """
        使用DNSMOSPro计算音质评分
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            DNSMOSPro评分 (1-5)
        """
        try:
            # 获取设备信息
            device = self._get_safe_device()
            
            # 加载音频文件
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # 转换为torch张量
            samples = torch.FloatTensor(audio_data)
            
            # 计算STFT
            spec = torch.FloatTensor(stft(samples.numpy()))
            
            # 将张量移动到与模型相同的设备
            if device != 'cpu' and torch.cuda.is_available():
                spec = spec.to(device)
            else:
                spec = spec.to('cpu')
            
            # 模型预测
            with torch.no_grad():
                prediction = self.dnsmospro_model(spec[None, None, ...])
                
                # 从预测结果中提取均值
                if device != 'cpu' and torch.cuda.is_available():
                    mean = prediction[:, 0].cpu().item()
                else:
                    mean = prediction[:, 0].item()
            
            # 使用均值作为评分
            score = np.clip(mean, 1.0, 5.0)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"DNSMOSPro评分计算失败：{audio_path}，错误：{str(e)}")
            return 1.0
    
    def _calculate_overall_score(self, quality_metrics: Dict[str, float]) -> float:
        """计算综合质量评分"""
        if not quality_metrics:
            return 1.0
        
        scores = []
        weights = []
        
        # Distill-MOS评分
        if 'distilmos' in quality_metrics:
            scores.append(quality_metrics['distilmos'])
            weights.append(0.33)
        
        # DNSMOS评分
        if 'dnsmos' in quality_metrics:
            scores.append(quality_metrics['dnsmos'])
            weights.append(0.33)
        
        # DNSMOSPro评分
        if 'dnsmospro' in quality_metrics:
            scores.append(quality_metrics['dnsmospro'])
            weights.append(0.34)
        
        if scores:
            # 加权平均
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 1.0
        
        return np.clip(overall_score, 1.0, 5.0)
    
    def assess_audio_array(self, audio_array: np.ndarray, sample_rate: int = None) -> Dict[str, Any]:
        """
        评估音频数组质量
        
        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            
        Returns:
            质量评估结果字典
        """
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # 创建临时文件
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
            
            try:
                # 保存音频数组到临时文件
                sf.write(temp_audio_path, audio_array, sample_rate)
                
                # 评估音频质量
                result = self.assess_audio_quality(temp_audio_path)
                
                return result
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            logger.error(f"音频数组质量评估失败：{str(e)}")
            return {
                'scores': {},
                'success': False,
                'error': str(e)
            }
    
    def is_high_quality(self, quality_result: Dict[str, Any]) -> bool:
        """
        判断音频是否为高质量
        
        Args:
            quality_result: 质量评估结果
            
        Returns:
            是否为高质量
        """
        if not quality_result.get('success', False):
            return False
        
        scores = quality_result.get('scores', {})
        
        # 检查Distill-MOS评分
        if self.enable_distilmos and 'distilmos' in scores:
            if scores['distilmos'] < self.distilmos_threshold:
                return False
        
        # 检查DNSMOS评分
        if self.enable_dnsmos and 'dnsmos' in scores:
            if scores['dnsmos'] < self.dnsmos_threshold:
                return False
        
        # 检查DNSMOSPro评分
        if self.enable_dnsmospro and 'dnsmospro' in scores:
            if scores['dnsmospro'] < self.dnsmospro_threshold:
                return False
        
        return True


class DNSMOSComputeScore:
    """DNSMOS评分计算器 - 基于官方实现"""
    
    def __init__(self, primary_model_path: str, p808_model_path: str):
        """
        初始化DNSMOS计算器
        
        Args:
            primary_model_path: 主模型路径
            p808_model_path: P808模型路径
        """
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
        # 常量
        self.SAMPLING_RATE = 16000
        self.INPUT_LENGTH = 9.01
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        """计算音频的mel频谱 - 与官方代码完全一致"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size+1, 
            hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T
    
    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """获取多项式拟合值 - 与官方代码完全一致"""
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        
        return sig_poly, bak_poly, ovr_poly
    
    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        """
        计算音频文件的DNSMOS分数 - 与官方代码完全一致
        
        Args:
            fpath: 音频文件路径
            sampling_rate: 采样率
            is_personalized_MOS: 是否使用个性化MOS
            
        Returns:
            包含各种分数的字典
        """
        try:
            # 读取音频文件 - 与官方代码一致
            aud, input_fs = sf.read(fpath)
            fs = sampling_rate
            if input_fs != fs:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
            else:
                audio = aud
            
            actual_audio_len = len(audio)
            len_samples = int(self.INPUT_LENGTH * fs)
            
            # 如果音频太短，重复填充 - 与官方代码一致
            while len(audio) < len_samples:
                audio = np.append(audio, audio)
            
            # 计算跳跃次数 - 与官方代码一致
            num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
            hop_len_samples = fs
            
            # 存储每个片段的预测结果
            predicted_mos_sig_seg_raw = []
            predicted_mos_bak_seg_raw = []
            predicted_mos_ovr_seg_raw = []
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []
            predicted_p808_mos = []
            
            # 处理每个音频片段 - 与官方代码一致
            for idx in range(num_hops):
                audio_seg = audio[int(idx * hop_len_samples) : int((idx + self.INPUT_LENGTH) * hop_len_samples)]
                if len(audio_seg) < len_samples:
                    continue
                
                # 准备输入特征 - 与官方代码完全一致
                input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
                p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
                
                # 运行推理 - 与官方代码一致
                oi = {'input_1': input_features}
                p808_oi = {'input_1': p808_input_features}
                
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                
                # 获取多项式拟合值
                mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                    mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
                )
                
                # 存储结果
                predicted_mos_sig_seg_raw.append(mos_sig_raw)
                predicted_mos_bak_seg_raw.append(mos_bak_raw)
                predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)
                predicted_p808_mos.append(p808_mos)
            
            # 构建结果字典 - 与官方代码一致
            clip_dict = {
                'filename': fpath,
                'len_in_sec': actual_audio_len / fs,
                'sr': fs,
                'num_hops': num_hops,
                'OVRL_raw': np.mean(predicted_mos_ovr_seg_raw),
                'SIG_raw': np.mean(predicted_mos_sig_seg_raw),
                'BAK_raw': np.mean(predicted_mos_bak_seg_raw),
                'OVRL': np.mean(predicted_mos_ovr_seg),
                'SIG': np.mean(predicted_mos_sig_seg),
                'BAK': np.mean(predicted_mos_bak_seg),
                'P808_MOS': np.mean(predicted_p808_mos)
            }
            
            return clip_dict
            
        except Exception as e:
            logger.error(f"DNSMOS计算失败：{fpath}，错误：{str(e)}")
            return {
                'filename': fpath,
                'OVRL': 1.0,
                'SIG': 1.0,
                'BAK': 1.0,
                'P808_MOS': 1.0
            } 