#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local copy of Multilingual WeSpeakerVerification used for speaker similarity.
This removes external sys.path dependency and bundles the required class here.
"""

from __future__ import annotations

import os
import sys

# ========================================
# 必须在导入 torch/torchaudio 之前设置环境变量
# 强制使用 soundfile 后端，避免加载 libtorchaudio.so
# 禁用 torio 的 FFmpeg 扩展，避免段错误
# ========================================
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"

import torch
import numpy as np
import yaml
import json
import time
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import warnings
from unittest.mock import MagicMock

warnings.filterwarnings('ignore')

# ========================================
# Mock torio 的 FFmpeg 扩展，防止段错误
# 必须在任何可能导入 torio 的代码之前执行
# ========================================
_mock_torio_ext = MagicMock()
_mock_torio_ext.ffmpeg = MagicMock()
sys.modules['torio._extension'] = _mock_torio_ext
sys.modules['torio._extension.ffmpeg'] = _mock_torio_ext.ffmpeg


class WeSpeakerVerification:
    """
    多语言说话人验证类
    使用WeSpeaker预训练模型进行说话人验证
    """

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        """
        初始化说话人验证模型

        Args:
            model_dir: WeSpeaker模型目录路径，默认使用预置路径
            device: 计算设备，默认自动选择（cuda/cpu）
        """
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 在CPU模式下显式屏蔽CUDA，避免底层库意外初始化GPU导致崩溃
        if str(self.device).lower() == 'cpu':
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        # 设置模型目录
        if model_dir is None:
            # 默认模型路径（可按需扩展）
            default_paths = [
                Path.cwd() / "samresnet100",
            ]
            for path in default_paths:
                if path.exists():
                    self.model_dir = str(path)
                    break
            else:
                raise ValueError("未找到模型文件，请指定model_dir参数或将模型放在默认路径")
        else:
            self.model_dir = model_dir

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载WeSpeaker模型"""
        # 在CPU模式下，设置 CUDA_VISIBLE_DEVICES 避免意外使用GPU
        if str(getattr(self, "device", "cpu")).lower() == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            
        # 直接导入，有错就抛出
        from wespeaker.cli.speaker import Speaker
        self.model = Speaker(self.model_dir)

        # 设置设备
        if hasattr(self.model, 'set_device'):
            self.model.set_device(self.device)
        else:
            if hasattr(self.model, 'model'):
                self.model.model = self.model.model.to(self.device)
            if hasattr(self.model, 'device'):
                self.model.device = torch.device(self.device)

        # 加载配置以获取模型信息
        config_path = os.path.join(self.model_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.embedding_size = self.config.get('projection_args', {}).get('embed_dim', 192)
        else:
            self.config = {}
            self.embedding_size = 192  # 默认值

    def extract_embedding(self, audio_path: Union[str, Path],
                          return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        提取音频的说话人特征向量

        Args:
            audio_path: 音频文件路径
            return_numpy: 是否返回numpy数组，否则返回torch.Tensor

        Returns:
            embedding: 说话人特征向量
        """
        audio_path = str(audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 使用WeSpeaker提取embedding
        embedding = self.model.extract_embedding(audio_path)

        if embedding is None:
            raise ValueError(f"无法提取embedding: {audio_path}")

        # 确保是numpy数组
        if not isinstance(embedding, np.ndarray):
            embedding = embedding.detach().cpu().numpy()

        # 确保是1维向量
        embedding = embedding.flatten()

        if not return_numpy:
            embedding = torch.from_numpy(embedding).float()

        return embedding

    def compute_similarity(self, embedding1: Union[np.ndarray, torch.Tensor],
                           embedding2: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算两个说话人特征向量的余弦相似度
        """
        # 转换为torch.Tensor
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1).float()
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2).float()

        # 确保是1维向量
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # 计算余弦相似度
        cosine_sim = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        )

        return cosine_sim.item()

    def extract_embedding_array(self, audio: Union[np.ndarray, torch.Tensor], sr: int,
                                return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        直接从内存中的波形数组提取embedding，避免写临时文件。
        使用 WeSpeaker 的 extract_embedding_from_pcm 接口。
        """
        # 转换为 torch.Tensor
        if isinstance(audio, np.ndarray):
            wav_tensor = torch.from_numpy(audio.astype(np.float32))
        else:
            wav_tensor = audio.float()
        
        # 确保是1维
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor[:, 0]
        
        # extract_embedding_from_pcm 需要 2D tensor (channels, samples)
        # 添加 channel 维度
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # (samples,) -> (1, samples)
        
        # 使用 WeSpeaker 的 extract_embedding_from_pcm 方法
        emb = self.model.extract_embedding_from_pcm(wav_tensor, sr)
        
        # 转换为 numpy 数组
        if not isinstance(emb, np.ndarray):
            emb = emb.detach().cpu().numpy()
        
        emb = emb.flatten()
        
        if return_numpy:
            return emb.astype(np.float32)
        return torch.from_numpy(emb.astype(np.float32))

    def verify_speakers(self, audio_path1: Union[str, Path],
                        audio_path2: Union[str, Path],
                        threshold: float = 0.5) -> Tuple[float, dict]:
        """
        验证两个音频是否为同一说话人
        """
        audio_path1 = str(audio_path1)
        audio_path2 = str(audio_path2)

        embedding1 = self.extract_embedding(audio_path1)
        embedding2 = self.extract_embedding(audio_path2)

        similarity = self.compute_similarity(embedding1, embedding2)
        is_same_speaker = similarity > threshold

        details = {
            'audio1': audio_path1,
            'audio2': audio_path2,
            'similarity': similarity,
            'threshold': threshold,
            'is_same_speaker': is_same_speaker,
            'embedding_dim': self.embedding_size
        }

        return similarity, details

    def batch_extract_embeddings(self, audio_paths: list,
                                 batch_size: int = 8,
                                 show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        批量提取音频embeddings
        """
        embeddings_dict = {}
        iterator = audio_paths
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(audio_paths, desc="提取embeddings")

        for audio_path in iterator:
            embedding = self.extract_embedding(audio_path)
            embeddings_dict[str(audio_path)] = embedding

        return embeddings_dict



