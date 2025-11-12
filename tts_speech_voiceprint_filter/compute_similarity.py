#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import random
import uuid

import numpy as np

from typing import Dict, List, Tuple, Optional, Any

# 直接导入，有错就抛出
import torch
import soundfile as sf
import librosa
from ten_vad import TenVad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEN_VAD_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("voiceprint_filter")


def read_wav_scp(paths: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """读取多个 Kaldi wav.scp，返回三种索引：
    - by_id: utt_id -> abs path
    - by_basename: basename -> [abs path]
    - by_stem: stem(without ext) -> [abs path]
    """
    by_id: Dict[str, str] = {}
    by_basename: Dict[str, List[str]] = {}
    by_stem: Dict[str, List[str]] = {}

    for p in paths:
        if not p or not os.path.isfile(p):
            logger.warning(f"wav.scp 不存在: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 格式: utt_id <space> abs/path.wav
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, wav_path = parts[0].strip(), parts[1].strip()
                if utt_id and wav_path:
                    by_id[utt_id] = wav_path
                    base = os.path.basename(wav_path)
                    stem, _ = os.path.splitext(base)
                    by_basename.setdefault(base, []).append(wav_path)
                    by_stem.setdefault(stem, []).append(wav_path)
    logger.info(f"wav.scp 索引完成: by_id={len(by_id)}, by_basename={len(by_basename)}, by_stem={len(by_stem)}")
    return by_id, by_basename, by_stem


def load_json_objects(path: str) -> Optional[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exact_find_source_wav_by_utt(utt_id: str, wav_index_by_id: Dict[str, str]) -> Optional[str]:
    """严格精确匹配：仅按 utt_id 在 wav.scp 索引里查找，不做任何放宽规则。"""
    if not utt_id:
        return None
    p = wav_index_by_id.get(utt_id)
    if p and os.path.isfile(p):
        return p
    return None


def find_source_wav(voiceprint_id: str,
                    hint_utt: Optional[str],
                    hint_path: Optional[str],
                    wav_index_by_id: Dict[str, str],
                    wav_index_by_basename: Dict[str, List[str]],
                    wav_index_by_stem: Dict[str, List[str]]) -> Optional[str]:
    """根据提供的线索在 wav.scp 索引里寻找源音频路径。
    优先级：显式 path > 显式 utt_id > 精确 stem/basename 匹配 > 包含关系匹配。
    """
    # 显式路径优先
    if hint_path and os.path.isfile(hint_path):
        return hint_path
    # 显式 utt_id
    if hint_utt and hint_utt in wav_index_by_id:
        p = wav_index_by_id[hint_utt]
        if os.path.isfile(p):
            return p
    # 精确 stem/basename 匹配
    for key, candidates in [(voiceprint_id, wav_index_by_stem.get(voiceprint_id, [])),
                            (f"{voiceprint_id}.wav", wav_index_by_basename.get(f"{voiceprint_id}.wav", []))]:
        for p in candidates:
            if os.path.isfile(p):
                return p
    # 宽松包含匹配（可能产生歧义，仅取第一个）
    for stem, candidates in wav_index_by_stem.items():
        if voiceprint_id in stem:
            for p in candidates:
                if os.path.isfile(p):
                    return p
    return None


def build_pairs_from_mapping(mapping_dir: str,
                             tts_zero_shot: str,
                             wav_scp_paths: List[str],
                             debug: bool = False,
                             debug_limit: Optional[int] = None) -> List[Tuple[str, str, str, str]]:
    """从映射JSON目录构建 (source_wav, tts_wav, voiceprint_id, prompt_id) 列表。"""
    wav_id, wav_base, wav_stem = read_wav_scp(wav_scp_paths)
    pairs: List[Tuple[str, str, str, str]] = []
    json_files = sorted(glob.glob(os.path.join(mapping_dir, "*.json")))
    if debug:
        random.shuffle(json_files)
    if not json_files:
        logger.error(f"映射目录下未找到JSON: {mapping_dir}")
        return pairs
    total_json = len(json_files)
    if debug and debug_limit and debug_limit > 0:
        logger.info(f"开始构建配对: 发现 {total_json} 个JSON 文件，将随机采样至 {debug_limit} 对（debug）")
    else:
        logger.info(f"开始构建配对: 发现 {total_json} 个JSON 文件")

    reached_limit = False
    for idx, jf in enumerate(json_files, 1):
        pairs_before = len(pairs)
        data = load_json_objects(jf)
        if not isinstance(data, dict):
            continue
        items = list(data.items())
        if debug:
            random.shuffle(items)
        for prompt_id, entries in items:
            if not isinstance(entries, list):
                continue
            entry_iter = list(entries)
            if debug:
                random.shuffle(entry_iter)
            for entry in entry_iter:
                vp_id: Optional[str] = None
                src_path: Optional[str] = None
                src_utt: Optional[str] = None
                tts_path: Optional[str] = None

                if isinstance(entry, dict):
                    vp_id = entry.get("voiceprint_id") or entry.get("voice_id") or entry.get("vp") or None
                    src_path = entry.get("source_path") or entry.get("src") or None
                    src_utt = entry.get("source_utt") or entry.get("utt") or None
                    # tts 路径可显式提供或按约定拼接
                    tts_path = entry.get("tts_path") or entry.get("tts") or None
                elif isinstance(entry, str):
                    # 兼容 "voiceprint_id\ttext" 等旧格式
                    parts = entry.split("\t", 1)
                    vp_id = parts[0].strip() if parts else None
                else:
                    continue

                if not tts_path:
                    # 约定：<zero_shot>/<prompt_id>/<voiceprint_id>.wav
                    tts_path = os.path.join(tts_zero_shot, prompt_id, f"{vp_id}.wav")

                # 严格精确匹配：源音频按JSON字典的key（prompt_id==源utt_id）在 wav.scp 中查找
                src = None
                if src_path:
                    # 若显式给出路径，仍优先使用（严格但直接）
                    if os.path.isfile(src_path):
                        src = src_path
                if src is None:
                    # 若条目内提供 source_utt，则优先；否则使用外层的 prompt_id 作为源utt
                    key_utt = src_utt if src_utt else prompt_id
                    src = exact_find_source_wav_by_utt(key_utt, wav_id)

                if not src or not os.path.isfile(src):
                    logger.debug(f"未找到源音频(精确匹配): utt={src_utt if src_utt else prompt_id}")
                    continue
                if not os.path.isfile(tts_path):
                    logger.debug(f"未找到TTS音频: {tts_path}")
                    continue
                pairs.append((src, tts_path, vp_id, prompt_id))
                if debug and debug_limit and debug_limit > 0 and len(pairs) >= debug_limit:
                    reached_limit = True
                    break
            if reached_limit:
                break
        added = len(pairs) - pairs_before
        logger.debug(f"[{idx}/{total_json}] 处理 {os.path.basename(jf)}: 新增配对 {added}，累计 {len(pairs)}")
        if idx % max(1, total_json // 20) == 0 or idx == total_json or reached_limit:
            logger.info(f"配对进度: {idx}/{total_json} ({idx/total_json:.1%})，累计 {len(pairs)} 对")
        if reached_limit:
            break

    logger.info(f"构建配对完成: 总计 {len(pairs)} 对")
    return pairs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _sanitize_for_filename(text: str) -> str:
    if text is None:
        return "none"
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:150]


def _mask_to_segments(mask: np.ndarray, sr: int) -> List[List[float]]:
    """将样本级掩码转换为以秒计的区间列表[[s, e], ...]"""
    if mask.size == 0:
        return []
    active = (mask > 0.5).astype(np.int8)
    padded = np.concatenate(([0], active, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = []
    for s, e in zip(starts, ends):
        segments.append([float(s) / sr, float(e) / sr])
    return segments


def _save_vad_plot(audio: np.ndarray,
                   mask: np.ndarray,
                   sr: int,
                   out_path: str,
                   title: str):
    """保存单路波形+VAD叠加图"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t = np.arange(len(audio)) / float(sr) if len(audio) else np.array([0.0, 1.0])
    plt.figure(figsize=(10, 3))
    plt.plot(t[:len(audio)], audio, color="#1f77b4", linewidth=0.8, label="waveform")
    if mask is not None and mask.size == len(audio):
        # 将mask放缩到[-1,1]范围内可视化
        m = (mask > 0.5).astype(np.float32)
        m = (m * 2.0) - 1.0  # 0->-1, 1->+1
        plt.fill_between(t[:len(m)], -1.05, m, where=(m > 0), color="orange", alpha=0.25, step="pre", label="VAD active")
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def process_pair(pair: Tuple[str, str, str, str],
                 verifier: Any,
                 save_vad_dir: Optional[str],
                 vad_frame_ms: int,
                 vad_min_speech_ms: int,
                 vad_max_silence_ms: int,
                 array_only: bool) -> Dict:
    src_path, tts_path, vp_id, prompt_id = pair
    
    try:
        # 使用 TEN VAD 仅在VAD active片段上计算相似度（若可用）
        sr_target = 16000
        src_audio, _ = _load_audio_mono(src_path, sr_target)
        tts_audio, _ = _load_audio_mono(tts_path, sr_target)

        # 最小音频长度检查（至少需要 500 样本 ≈ 31ms @ 16kHz，fbank window_size=400）
        min_audio_len = 500
        if src_audio.size < min_audio_len or tts_audio.size < min_audio_len:
            return {
                "voiceprint_id": vp_id,
                "prompt_id": prompt_id,
                "source_path": src_path,
                "tts_path": tts_path,
                "similarity": 0.0,
                "similarity_original": 0.0,
                "similarity_vad": 0.0,
                "success": False,
                "error_message": f"Audio too short: src={src_audio.size} tts={tts_audio.size} samples (min={min_audio_len})",
                "vad": {"used": False},
                "debug_plots": {},
                "durations_sec": {
                    "src_total": float(len(src_audio) / 16000.0),
                    "tts_total": float(len(tts_audio) / 16000.0),
                    "src_used": 0.0,
                    "tts_used": 0.0
                }
            }

        if TEN_VAD_AVAILABLE:
            src_mask = _apply_ten_vad_refined(src_audio, sr_target,
                                              frame_ms=vad_frame_ms,
                                              vad_min_speech_ms=vad_min_speech_ms,
                                              vad_max_silence_ms=vad_max_silence_ms)
            tts_mask = _apply_ten_vad_refined(tts_audio, sr_target,
                                              frame_ms=vad_frame_ms,
                                              vad_min_speech_ms=vad_min_speech_ms,
                                              vad_max_silence_ms=vad_max_silence_ms)
            src_active = src_audio[src_mask > 0.5]
            tts_active = tts_audio[tts_mask > 0.5]
            min_len = int(0.2 * sr_target)  # 至少200ms
            if src_active.size >= min_len and tts_active.size >= min_len:
                src_audio_use = src_active.astype(np.float32)
                tts_audio_use = tts_active.astype(np.float32)
            else:
                src_audio_use = src_audio
                tts_audio_use = tts_audio
        else:
            src_audio_use = src_audio
            tts_audio_use = tts_audio

        # 计算第一个相似度：在原始音频上（不经过VAD）
        src_emb_original = verifier.extract_embedding_array(src_audio, sr_target)  # type: ignore
        tts_emb_original = verifier.extract_embedding_array(tts_audio, sr_target)  # type: ignore
        sim_original = verifier.compute_similarity(src_emb_original, tts_emb_original)

        # VAD 后再次检查长度（防止 VAD 后为空）
        sim_vad = None
        if src_audio_use.size < min_audio_len or tts_audio_use.size < min_audio_len:
            # VAD后音频过短，但原始相似度已计算
            sim_vad = 0.0
            logger.debug(f"VAD后音频过短 [{vp_id}/{prompt_id}]: src={src_audio_use.size} tts={tts_audio_use.size}")
        else:
            # 计算第二个相似度：在VAD处理后的音频上
            src_emb_vad = verifier.extract_embedding_array(src_audio_use, sr_target)  # type: ignore
            tts_emb_vad = verifier.extract_embedding_array(tts_audio_use, sr_target)  # type: ignore
            sim_vad = verifier.compute_similarity(src_emb_vad, tts_emb_vad)

        # 兼容旧接口：默认使用VAD后的相似度（如果可用），否则使用原始相似度
        sim = sim_vad if (sim_vad is not None and TEN_VAD_AVAILABLE) else sim_original
        
        # 保存 VAD 结果（仅占比与参数，不落盘掩码）
        vad_info = {}
        plot_paths: Dict[str, str] = {}
        segments_src: List[List[float]] = []
        segments_tts: List[List[float]] = []
        if TEN_VAD_AVAILABLE:
            src_ratio = float(np.mean((src_mask > 0.5).astype(np.float32))) if src_audio.size > 0 else 0.0
            tts_ratio = float(np.mean((tts_mask > 0.5).astype(np.float32))) if tts_audio.size > 0 else 0.0
            segments_src = _mask_to_segments(src_mask, sr_target)
            segments_tts = _mask_to_segments(tts_mask, sr_target)
            vad_info = {
                "used": True,
                "frame_ms": int(vad_frame_ms),
                "min_speech_ms": int(vad_min_speech_ms),
                "max_silence_ms": int(vad_max_silence_ms),
                "src_active_ratio": src_ratio,
                "tts_active_ratio": tts_ratio,
                "src_segments_sec": segments_src,
                "tts_segments_sec": segments_tts
            }
        else:
            vad_info = {"used": False}

        # 调试模式：保存波形+VAD图
        if save_vad_dir:
            uid = uuid.uuid4().hex[:8]
            base_name = f"{_sanitize_for_filename(prompt_id)}__{_sanitize_for_filename(vp_id)}__{uid}"
            src_fig = os.path.join(save_vad_dir, f"{base_name}_src.png")
            tts_fig = os.path.join(save_vad_dir, f"{base_name}_tts.png")
            _save_vad_plot(src_audio, (src_mask if TEN_VAD_AVAILABLE else np.ones_like(src_audio, dtype=np.float32)),
                           sr_target, src_fig, f"SRC {prompt_id} | {vp_id}")
            _save_vad_plot(tts_audio, (tts_mask if TEN_VAD_AVAILABLE else np.ones_like(tts_audio, dtype=np.float32)),
                           sr_target, tts_fig, f"TTS {prompt_id} | {vp_id}")
            plot_paths = {"src_plot": src_fig, "tts_plot": tts_fig}

        return {
            "voiceprint_id": vp_id,
            "prompt_id": prompt_id,
            "source_path": src_path,
            "tts_path": tts_path,
            "similarity": sim,  # 兼容旧接口：默认相似度
            "similarity_original": sim_original,  # 新增：原始音频相似度
            "similarity_vad": sim_vad,  # 新增：VAD后音频相似度
            "success": True,
            "error_message": "",
            "vad": vad_info,
            "debug_plots": plot_paths,
            "durations_sec": {
                "src_total": float(len(src_audio) / 16000.0),
                "tts_total": float(len(tts_audio) / 16000.0),
                "src_used": float(len(src_audio_use) / 16000.0),
                "tts_used": float(len(tts_audio_use) / 16000.0)
            }
        }
    except Exception as e:
        # 捕获任何异常，返回失败结果
        logger.error(f"处理音频对失败 [{vp_id}/{prompt_id}]: {e}")
        return {
            "voiceprint_id": vp_id,
            "prompt_id": prompt_id,
            "source_path": src_path,
            "tts_path": tts_path,
            "similarity": 0.0,
            "similarity_original": 0.0,
            "similarity_vad": 0.0,
            "success": False,
            "error_message": str(e),
            "vad": {"used": False},
            "debug_plots": {},
            "durations_sec": {
                "src_total": 0.0,
                "tts_total": 0.0,
                "src_used": 0.0,
                "tts_used": 0.0
            }
        }

def _load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """加载为单通道float32，并重采样到 target_sr。"""
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type='soxr_vhq')
        sr = target_sr
    return data.astype(np.float32), sr

def _apply_ten_vad_refined(audio: np.ndarray, sr: int,
                           frame_ms: int = 16,
                           vad_min_speech_ms: int = 80,
                           vad_max_silence_ms: int = 160) -> np.ndarray:
    """
    使用 TEN VAD 生成 refined 掩码（简化版）：
    - 按帧运行 TenVad 得到原始 flags
    - 去除过短语音段（< vad_min_speech_ms）
    - 填补过短静音段（<= vad_max_silence_ms）
    - 展开到样本级，作为 ten_vad_refined
    """
    if not TEN_VAD_AVAILABLE or audio.size == 0:
        return np.ones_like(audio, dtype=np.float32)

    hop = int(frame_ms * sr / 1000)
    if hop <= 0:
        hop = max(1, int(0.016 * sr))
    tv = TenVad(hop, 0.5)

    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_clipped * 32767.0).astype(np.int16)
    num_frames = max(0, len(audio_i16) // hop)
    flags = []
    for i in range(num_frames):
        frame = audio_i16[i * hop:(i + 1) * hop]
        if frame.shape[0] == hop:
            _, flag = tv.process(frame)
            flags.append(int(flag))
        else:
            flags.append(0)
    flags = np.array(flags, dtype=np.int8)

    # 去除过短语音段
    min_speech_frames = max(1, int(vad_min_speech_ms / frame_ms))
    padded = np.concatenate(([0], flags, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < min_speech_frames:
            flags[s0:e0] = 0

    # 填补过短静音段
    max_silence_frames = max(1, int(vad_max_silence_ms / frame_ms))
    padded = np.concatenate(([0], flags, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for i in range(len(ends) - 1):
        silence_len = starts[i + 1] - ends[i]
        if silence_len <= max_silence_frames:
            flags[ends[i]:starts[i + 1]] = 1

    # 展开到样本级 refined 掩码
    refined = np.zeros(len(audio), dtype=np.float32)
    for i, f in enumerate(flags):
        if f:
            s = i * hop
            e = min((i + 1) * hop, len(audio))
            refined[s:e] = 1.0
    return refined

def _split_even(items: List, n: int) -> List[List]:
    """将 items 尽量均匀划分为 n 份"""
    n = max(1, n)
    total = len(items)
    base = total // n
    rem = total % n
    splits = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        end = start + size
        if size > 0:
            splits.append(items[start:end])
        else:
            splits.append([])
        start = end
    return splits

def _process_subset_on_gpu(args_tuple):
    """子进程入口：固定到指定GPU上，初始化模型并处理子集"""
    gpu_id, subset_pairs, model_dir, log_every, save_vad_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms = args_tuple
    # 限定可见GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 在该进程中使用cuda设备
    from multilingual_inference import WeSpeakerVerification  # noqa: E402
    verifier = WeSpeakerVerification(model_dir=model_dir, device="cuda")
    results = []
    total = len(subset_pairs)
    for idx, p in enumerate(subset_pairs, 1):
        results.append(process_pair(p, verifier, save_vad_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms, False))
        if log_every and (idx % log_every == 0 or idx == total):
            logging.info(f"GPU {gpu_id}: 进度 {idx}/{total} ({idx/total:.1%})")
    return results

def _worker_process(gpu_id: int, subset_pairs: List[Tuple[str, str, str, str]], model_dir: str, log_every: int, result_queue: "mp.Queue",
                    save_vad_dir: Optional[str], vad_frame_ms: int, vad_min_speech_ms: int, vad_max_silence_ms: int):
    """mp.spawn/Process 使用的工作进程：将结果通过队列返回"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from multilingual_inference import WeSpeakerVerification  # noqa: E402
    verifier = WeSpeakerVerification(model_dir=model_dir, device="cuda")
    results = []
    total = len(subset_pairs)
    for idx, p in enumerate(subset_pairs, 1):
        results.append(process_pair(p, verifier, save_vad_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms, False))
        if log_every and (idx % log_every == 0 or idx == total):
            logging.info(f"GPU {gpu_id}: 进度 {idx}/{total} ({idx/total:.1%})")
    result_queue.put(results)


def aggregate_and_save(results: List[Dict],
                       threshold: float,
                       output_path: str,
                       meta: Dict):
    sims = [r["similarity"] for r in results if r["success"]]
    sims_original = [r["similarity_original"] for r in results if r["success"] and "similarity_original" in r]
    sims_vad = [r["similarity_vad"] for r in results if r["success"] and "similarity_vad" in r and r["similarity_vad"] is not None]
    
    passed = [r for r in results if r["success"] and r["similarity"] >= threshold]
    failed = [r for r in results if r["success"] and r["similarity"] < threshold]
    errors = [r for r in results if not r["success"]]

    stats = {
        "total_pairs": len(results),
        "processed_pairs": len(sims),
        "failed_pairs": len(errors),
        "passed_pairs": len(passed),
        "filtered_pairs": len(failed),
        "threshold": threshold,
    }
    if sims:
        stats["similarity_stats"] = {
            "mean": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "std": float(np.std(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
        }
    if sims_original:
        stats["similarity_original_stats"] = {
            "mean": float(np.mean(sims_original)),
            "median": float(np.median(sims_original)),
            "std": float(np.std(sims_original)),
            "min": float(np.min(sims_original)),
            "max": float(np.max(sims_original)),
        }
    if sims_vad:
        stats["similarity_vad_stats"] = {
            "mean": float(np.mean(sims_vad)),
            "median": float(np.median(sims_vad)),
            "std": float(np.std(sims_vad)),
            "min": float(np.min(sims_vad)),
            "max": float(np.max(sims_vad)),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "meta": meta,
        "statistics": stats,
        "filter_results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果保存: {output_path}")

    # 写出筛除列表（TTS路径）
    filtered_list_path = output_path.replace(".json", "_filtered_list.txt")
    with open(filtered_list_path, "w", encoding="utf-8") as f:
        for r in failed:
            f.write(f"{r['tts_path']}\n")
    logger.info(f"筛除列表: {filtered_list_path}")


def main():
    parser = argparse.ArgumentParser(description="TTS 复刻音频与原始音频的声纹相似度筛选（Multilingual WeSpeakerVerification）")
    parser.add_argument("--mapping_dir", type=str, required=True, help="映射JSON目录")
    parser.add_argument("--tts_zero_shot", type=str, required=True, help="TTS zero_shot 目录")
    parser.add_argument("--wav_scp", type=str, action="append", required=True, help="Kaldi wav.scp，可多次传入")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度阈值 (默认: 0.90)")
    parser.add_argument("--model_dir", type=str, default="/root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100",
                        help="Multilingual WeSpeaker 模型目录（samresnet100）")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="设备（默认自动）")
    parser.add_argument("--num_workers", type=int, default=8, help="单GPU线程池并发（仅在单GPU时使用）")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的GPU数量（多GPU分片并行）")
    parser.add_argument("--output", type=str, required=True, help="输出JSON路径")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    # 调试选项
    parser.add_argument("--debug", action="store_true", help="调试模式：随机打乱，仅取样若干条，保存波形+VAD图")
    parser.add_argument("--debug_samples", type=int, default=1000, help="调试模式下采样条数")
    parser.add_argument("--debug_dir", type=str, default=None, help="调试输出目录（保存波形+VAD图）")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.mapping_dir):
        logger.error(f"映射目录不存在: {args.mapping_dir}")
        return 1
    if not os.path.isdir(args.tts_zero_shot):
        logger.error(f"TTS zero_shot 不存在: {args.tts_zero_shot}")
        return 1
    for p in args.wav_scp:
        if not os.path.isfile(p):
            logger.error(f"wav.scp 不存在: {p}")
            return 1

    pairs = build_pairs_from_mapping(
        args.mapping_dir,
        args.tts_zero_shot,
        args.wav_scp,
        debug=bool(args.debug),
        debug_limit=int(args.debug_samples) if args.debug else None
    )
    if not pairs:
        logger.error("未构建到有效的 (source, tts) 配对，结束")
        return 1

    start = time.time()

    # 调试模式：随机打乱并裁样
    save_debug_dir: Optional[str] = None
    if args.debug:
        random.seed(2025)
        random.shuffle(pairs)
        if args.debug_samples > 0 and len(pairs) > args.debug_samples:
            pairs = pairs[:args.debug_samples]
        # 调试输出目录
        if args.debug_dir:
            save_debug_dir = args.debug_dir
        else:
            save_debug_dir = os.path.splitext(args.output)[0] + "_debug"
        os.makedirs(save_debug_dir, exist_ok=True)

    total_pairs = len(pairs)
    logger.info(f"开始相似度计算: {total_pairs} 对")

    # 决定使用的GPU数
    if args.device == "cpu":
        usable_gpus = 0
    else:
        usable_gpus = (torch.cuda.device_count() if (torch and torch.cuda.is_available()) else 0)
    num_gpus = min(args.num_gpus, usable_gpus) if usable_gpus > 0 else 0
    # 调试模式：强制单进程路径，避免多GPU spawn 带来的加载/同步开销
    if args.debug:
        logger.info("调试模式：使用单进程以避免多GPU开销")
        num_gpus = 0

    results: List[Dict] = []
    # VAD 参数（不落盘掩码）
    vad_save_dir = save_debug_dir  # 调试模式下用于保存图像
    vad_frame_ms = 16
    vad_min_speech_ms = 80
    vad_max_silence_ms = 160
    if num_gpus and num_gpus > 1:
        # 多GPU分片 + 多进程（mp spawn，不使用 ProcessPoolExecutor）
        logger.info(f"启用多GPU处理: {num_gpus} 卡")
        subsets = _split_even(pairs, num_gpus)
        log_every = max(1, total_pairs // (num_gpus * 20))  # 每子集约5%打印一次，至少为1
        # 使用 spawn 避免 CUDA 在 fork 下的问题
        mp.set_start_method("spawn", force=True)
        result_queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []
        for gpu_id, subset in enumerate(subsets):
            if not subset:
                continue
            p = mp.Process(target=_worker_process, args=(gpu_id, subset, args.model_dir, log_every, result_queue,
                                                         vad_save_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms))
            p.start()
            processes.append(p)
        # 收集结果
        for _ in processes:
            # 超时防止卡住（按数据量动态放宽）
            timeout_s = max(60, total_pairs // 2)
            part = result_queue.get(timeout=timeout_s)
            if part:
                results.extend(part)
        # 回收进程（带超时，必要时强制终止）
        for p in processes:
            p.join(timeout=300)
        for p in processes:
            if p.is_alive():
                logging.warning(f"GPU 子进程仍在运行，强制终止: PID={p.pid}")
                p.terminate()
                p.join(timeout=30)
    else:
        # 单GPU/CPU：单进程 + 线程并发
        # 在调试模式且未显式指定设备时，强制使用CPU，避免CUDA初始化导致的潜在崩溃
        if args.device in ("cuda", "cpu"):
            device_opt = args.device
        elif args.debug:
            logger.info("调试模式：强制使用 CPU 设备以提高稳定性")
            device_opt = "cpu"
        else:
            device_opt = "cuda" if usable_gpus > 0 else "cpu"
        from multilingual_inference import WeSpeakerVerification  # noqa: E402
        verifier = WeSpeakerVerification(model_dir=args.model_dir, device=device_opt)
        # 在 debug 模式下强制仅使用 array 提特征，避免触发外部解码库
        array_only = True if args.debug else False
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futures = [ex.submit(process_pair, p, verifier, vad_save_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms, array_only) for p in pairs]
            completed = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                completed += 1
                if completed % 100 == 0 or completed == total_pairs:
                    logger.info(f"相似度计算进度: {completed}/{total_pairs} ({completed/total_pairs:.1%})")

    duration = time.time() - start
    logger.info(f"处理完成: {len(results)} 对，耗时 {duration:.2f}s")

    meta = {
        "mapping_dir": args.mapping_dir,
        "tts_zero_shot": args.tts_zero_shot,
        "wav_scp": args.wav_scp,
        "model_dir": args.model_dir,
        "num_workers": args.num_workers,
        "duration_sec": duration,
        "debug": {
            "enabled": bool(args.debug),
            "samples": int(args.debug_samples),
            "debug_dir": save_debug_dir
        }
    }
    aggregate_and_save(results, args.threshold, args.output, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())


