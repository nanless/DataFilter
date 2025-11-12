#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt-vs-Clone speaker similarity workflow for datasets organized as:
  /root/group-shared/voiceprint/share/voiceclone_child_YYYYMMDD/
    ├── <subdir_1>/
    ├── <subdir_2>/
    ├── ...
    ├── <subdir_10>/
    └── *.json  (each JSON describes mappings inside a specific subdir)

Each JSON contains entries with at least:
  - prompt_id: ID of the prompt audio
  - clone_id:  ID of the cloned audio (may be missing if not cloned)
  - prompt_text: optional

This script will:
  - scan the ten subdirectories for audio files (wav/mp3/flac/m4a)
  - read JSONs in the root to build (prompt_audio, clone_audio, clone_id, prompt_id) pairs
  - compute speaker similarity using the same WeSpeaker workflow as compute_similarity.py
  - save a rich JSON report and a filtered list of rejected clone paths (by threshold)
"""

import os
import sys

# ========================================
# 必须在导入 torch/torchaudio 之前设置环境变量
# ========================================
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"

import re
import json
import time
import glob
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')

import numpy as np

# Reuse WeSpeaker + processing pipeline from compute_similarity.py
from multilingual_inference import WeSpeakerVerification  # noqa: E402
import compute_similarity as base  # reuse process_pair, aggregate, VAD, workers


logger = logging.getLogger("prompt_vs_clone")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".webm")


def _norm_id(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return s


def _basename_no_ext(p: str) -> str:
    b = os.path.basename(p)
    s, _ = os.path.splitext(b)
    return s


def _scan_subdir_audio_indices(subdir_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Build indices for a subdir:
      - by_stem: filename without extension -> [abs paths]
      - by_basename: filename with extension -> [abs paths]
    """
    by_stem: Dict[str, List[str]] = {}
    by_base: Dict[str, List[str]] = {}
    for root, _, files in os.walk(subdir_path):
        for fn in files:
            if fn.lower().endswith(SUPPORTED_AUDIO_EXTS):
                full = os.path.join(root, fn)
                base = os.path.basename(full)
                stem, _ = os.path.splitext(base)
                by_base.setdefault(base, []).append(full)
                by_stem.setdefault(stem, []).append(full)
    return by_stem, by_base


def _find_audio_by_id(audio_id: str,
                      subdir_stem_idx: Dict[str, List[str]],
                      subdir_base_idx: Dict[str, List[str]],
                      global_stem_idx: Dict[str, List[str]],
                      global_base_idx: Dict[str, List[str]]) -> Optional[str]:
    """
    Heuristics to locate an audio file by ID within the subdir first, then globally:
    1) Exact stem match in subdir
    2) Exact basename match in subdir (adding extensions)
    3) Exact stem match globally
    4) Exact basename match globally (adding extensions)
    
    移除了包含关系匹配，避免O(N*M)复杂度导致的性能问题。
    """
    if not audio_id:
        return None
    # strip ext if provided
    stem_guess = os.path.splitext(str(audio_id))[0]
    # 1) exact stem in subdir
    cand = subdir_stem_idx.get(stem_guess) or subdir_stem_idx.get(audio_id)
    if cand:
        for p in cand:
            if os.path.isfile(p):
                return p
    # 2) exact basename with known extensions in subdir
    for ext in SUPPORTED_AUDIO_EXTS:
        base = f"{stem_guess}{ext}"
        cand = subdir_base_idx.get(base)
        if cand:
            for p in cand:
                if os.path.isfile(p):
                    return p
    # 3) exact stem globally
    cand = global_stem_idx.get(stem_guess) or global_stem_idx.get(audio_id)
    if cand:
        for p in cand:
            if os.path.isfile(p):
                return p
    # 4) exact basename with known extensions globally
    for ext in SUPPORTED_AUDIO_EXTS:
        base = f"{stem_guess}{ext}"
        cand = global_base_idx.get(base)
        if cand:
            for p in cand:
                if os.path.isfile(p):
                    return p
    return None


def _iter_entries_from_json(obj) -> List[Dict[str, Optional[str]]]:
    """
    Normalize various JSON schemas into a list of entries with prompt_id and clone_id.
    Accepts:
      - list of dicts
      - dict[str -> list]
      - dict[str -> dict] variants
    Field name variants supported:
      prompt: 'prompt_id', 'promptId', 'prompt', 'src_prompt_id'
      clone:  'clone_id', 'tts_id', 'voice_id', 'audio_id', 'cloneId', 'ttsId'
    Also support direct file paths:
      prompt_path: 'prompt_path','prompt_wav','prompt_file'
      clone_path:  'clone_path','clone_wav','clone_file'
    """
    out: List[Dict[str, Optional[str]]] = []

    def extract_one(d: dict) -> Optional[Dict[str, Optional[str]]]:
        if not isinstance(d, dict):
            return None
        prompt_keys = [
            "prompt_id", "promptId", "prompt", "src_prompt_id",
            "prompt_audio_id", "promptAudioId", "source_utt", "source_id", "sourceId"
        ]
        clone_keys = [
            "clone_id", "tts_id", "voice_id", "audio_id", "cloneId", "ttsId", "tts_audio_id",
            "voiceprint_id", "vp", "voiceId", "voiceprintId", "ttsAudioId", "tts_audio"
        ]
        prompt_path_keys = ["prompt_path", "prompt_wav", "prompt_file"]
        clone_path_keys = ["clone_path", "clone_wav", "clone_file", "tts_path", "tts_wav", "tts_file"]
        prompt_id = None
        clone_id = None
        prompt_path = None
        clone_path = None
        for k in prompt_keys:
            if k in d:
                prompt_id = _norm_id(d.get(k))
                break
        for k in clone_keys:
            if k in d:
                clone_id = _norm_id(d.get(k))
                break
        for k in prompt_path_keys:
            if k in d:
                prompt_path = _norm_id(d.get(k))
                break
        for k in clone_path_keys:
            if k in d:
                clone_path = _norm_id(d.get(k))
                break
        if prompt_id is None and "id" in d:
            # sometimes prompt id is just 'id'
            prompt_id = _norm_id(d.get("id"))
        if prompt_id or clone_id or prompt_path or clone_path:
            return {
                "prompt_id": prompt_id,
                "clone_id": clone_id,
                "prompt_path": prompt_path,
                "clone_path": clone_path,
            }
        return None

    if isinstance(obj, list):
        for it in obj:
            maybe = extract_one(it) if isinstance(it, dict) else None
            if maybe:
                out.append(maybe)
    elif isinstance(obj, dict):
        # support dict mapping: { "<prompt_id>": "<clone_id or object>" }
        for k, v in obj.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        maybe = extract_one(it)
                        if maybe:
                            if not maybe.get("prompt_id"):
                                maybe["prompt_id"] = _norm_id(k)
                            out.append(maybe)
                    elif isinstance(it, str):
                        # format like: "voiceprint_id\tprompt_text"
                        parts = it.split("\t", 1)
                        clone_id = _norm_id(parts[0]) if parts else None
                        if clone_id:
                            out.append({"prompt_id": _norm_id(k), "clone_id": clone_id})
            elif isinstance(v, dict):
                maybe = extract_one(v)
                if maybe:
                    # if top-level key looks like a prompt id and inner object lacks it, fill it
                    if not maybe.get("prompt_id"):
                        maybe["prompt_id"] = _norm_id(k)
                    # if 'id' present in inner dict and not used as prompt_id, treat as clone_id fallback
                    if not maybe.get("clone_id") and isinstance(v, dict) and "id" in v:
                        maybe["clone_id"] = _norm_id(v.get("id"))
                    out.append(maybe)
            else:
                # value is a string -> treat as clone_id, key as prompt_id
                if isinstance(v, str):
                    out.append({"prompt_id": _norm_id(k), "clone_id": _norm_id(v)})
    return out


def _load_json_safe(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取JSON失败: {path} -> {e}")
        return None


def _pick_best_subdir_for_json(json_stem: str, subdirs: List[str]) -> Optional[str]:
    """
    Pick the most plausible subdir for a json file:
      1) exact name match
      2) json_stem contained in subdir name
      3) subdir name contained in json_stem
      4) fallback: None
    """
    if json_stem in subdirs:
        return json_stem
    contains = [sd for sd in subdirs if json_stem in sd]
    if contains:
        return sorted(contains, key=lambda s: len(s))[0]
    contained = [sd for sd in subdirs if sd in json_stem]
    if contained:
        return sorted(contained, key=lambda s: -len(s))[0]
    return None


def _resolve_path_candidate(path_like: Optional[str], root_dir: str, json_stem: Optional[str]) -> Optional[str]:
    """
    Resolve a path possibly absolute or relative to a subdir guessed by json_stem.
    Try absolute, then root/json_stem/path, then root/path.
    """
    if not path_like:
        return None
    p = os.path.normpath(path_like)
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    # try relative to json_stem subdir
    if json_stem:
        cand = os.path.join(root_dir, json_stem, p)
        if os.path.isfile(cand):
            return cand
    # try relative to root
    cand = os.path.join(root_dir, p)
    if os.path.isfile(cand):
        return cand
    return None


def _read_wav_scp(paths: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    读取 Kaldi wav.scp:
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


def _find_prompt_in_wavscp(prompt_id: str,
                           wav_by_id: Dict[str, str],
                           wav_by_base: Dict[str, List[str]],
                           wav_by_stem: Dict[str, List[str]]) -> Optional[str]:
    if not prompt_id:
        return None
    # exact id
    p = wav_by_id.get(prompt_id)
    if p and os.path.isfile(p):
        return p
    stem = os.path.splitext(prompt_id)[0]
    # exact stem
    for c in wav_by_stem.get(stem, []):
        if os.path.isfile(c):
            return c
    # exact basename
    for ext in SUPPORTED_AUDIO_EXTS:
        for c in wav_by_base.get(f"{stem}{ext}", []):
            if os.path.isfile(c):
                return c
    return None


def build_pairs_from_prompt_root(root_dir: str,
                                 wav_scp_paths: Optional[List[str]] = None) -> List[Tuple[str, str, str, str]]:
    """
    Build (prompt_path, clone_path, clone_id, prompt_id) pairs by:
      - scanning all immediate subdirectories for audio files
      - matching IDs from JSONs located in root_dir
    """
    pairs: List[Tuple[str, str, str, str]] = []
    if not os.path.isdir(root_dir):
        logger.error(f"目录不存在: {root_dir}")
        return pairs

    # Scan immediate subdirectories
    subdir_paths = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]
    subdirs = [os.path.basename(p) for p in subdir_paths]
    # Build per-subdir indices and global indices
    subdir_name_to_idx: Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]]]] = {}
    global_stem_idx: Dict[str, List[str]] = {}
    global_base_idx: Dict[str, List[str]] = {}
    for sd in sorted(subdirs):
        stem_idx, base_idx = _scan_subdir_audio_indices(os.path.join(root_dir, sd))
        subdir_name_to_idx[sd] = (stem_idx, base_idx)
        # merge into global
        for k, v in stem_idx.items():
            global_stem_idx.setdefault(k, []).extend(v)
        for k, v in base_idx.items():
            global_base_idx.setdefault(k, []).extend(v)

    # Read JSON files from root_dir
    json_files = sorted(glob.glob(os.path.join(root_dir, "*.json")))
    if not json_files:
        logger.error(f"未找到JSON文件: {root_dir}")
        return pairs

    logger.info(f"发现子目录 {len(subdirs)} 个，JSON {len(json_files)} 个")
    # wav.scp 索引（可选）
    wav_by_id: Dict[str, str] = {}
    wav_by_base: Dict[str, List[str]] = {}
    wav_by_stem: Dict[str, List[str]] = {}
    if wav_scp_paths:
        wav_by_id, wav_by_base, wav_by_stem = _read_wav_scp(wav_scp_paths)

    total_before = 0
    for jf in json_files:
        data = _load_json_safe(jf)
        if not isinstance(data, (list, dict)):
            continue
        entries = _iter_entries_from_json(data)
        if not entries:
            continue
        # Guess subdir by json filename stem
        json_stem = os.path.splitext(os.path.basename(jf))[0]
        best_sub = _pick_best_subdir_for_json(json_stem, subdirs)
        sub_stem_idx, sub_base_idx = subdir_name_to_idx.get(best_sub or "", ({}, {}))

        # counters for diagnostics
        cnt_total = 0
        cnt_with_prompt = 0
        cnt_with_clone = 0
        cnt_prompt_found = 0
        cnt_clone_found = 0

        before = len(pairs)
        for ent in entries:
            cnt_total += 1
            prompt_id = _norm_id(ent.get("prompt_id"))
            clone_id = _norm_id(ent.get("clone_id"))
            prompt_path_hint = _norm_id(ent.get("prompt_path"))
            clone_path_hint = _norm_id(ent.get("clone_path"))
            if prompt_id or prompt_path_hint:
                cnt_with_prompt += 1
            if clone_id or clone_path_hint:
                cnt_with_clone += 1
            # first: resolve direct paths if provided
            prompt_path = _resolve_path_candidate(prompt_path_hint, root_dir, best_sub or json_stem)
            clone_path = _resolve_path_candidate(clone_path_hint, root_dir, best_sub or json_stem)

            # if no direct path, search by id
            if not prompt_path and prompt_id:
                prompt_path = _find_audio_by_id(prompt_id, sub_stem_idx, sub_base_idx, global_stem_idx, global_base_idx)
            if not prompt_path and prompt_id and wav_by_id:
                prompt_path = _find_prompt_in_wavscp(prompt_id, wav_by_id, wav_by_base, wav_by_stem)
            if prompt_path and os.path.isfile(prompt_path):
                cnt_prompt_found += 1
            else:
                # 不逐条输出，避免大量日志；统计后统一打印
                continue
            if not clone_path and clone_id:
                clone_path = _find_audio_by_id(clone_id, sub_stem_idx, sub_base_idx, global_stem_idx, global_base_idx)
            if clone_path and os.path.isfile(clone_path):
                cnt_clone_found += 1
            else:
                # Many clone_ids may be absent; skip silently
                continue
            pairs.append((prompt_path, clone_path, clone_id, prompt_id))
            # 进度日志（每5000条打印一次）
            if cnt_total % 5000 == 0:
                logger.info(f"{os.path.basename(jf)} 进度: {cnt_total} 条，prompt找到 {cnt_prompt_found}，clone找到 {cnt_clone_found}，已匹配 {len(pairs)-before}")
        added = len(pairs) - before
        logger.info(f"{os.path.basename(jf)}: 条目 {cnt_total}，含prompt {cnt_with_prompt}，含clone {cnt_with_clone}，"
                    f"找到prompt {cnt_prompt_found}，找到clone {cnt_clone_found}，匹配到 {added} 对 (累计: {len(pairs)})")

    logger.info(f"构建配对完成: 总计 {len(pairs)} 对")
    return pairs


def _worker_process_with_env(gpu_id: int, subset_pairs: List[Tuple[str, str, str, str]], 
                             model_dir: str, log_every: int, result_queue: mp.Queue,
                             vad_save_dir: Optional[str], vad_frame_ms: int, 
                             vad_min_speech_ms: int, vad_max_silence_ms: int):
    """
    多进程 worker，在子进程中设置 GPU 和加载模型
    环境变量和 mock 已经在 multilingual_inference.py 模块级别设置
    """
    try:
        # 设置 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 导入 WeSpeaker（环境变量已在 multilingual_inference.py 中设置）
        from multilingual_inference import WeSpeakerVerification
        
        verifier = WeSpeakerVerification(model_dir=model_dir, device="cuda")
        results = []
        total = len(subset_pairs)
        for idx, p in enumerate(subset_pairs, 1):
            results.append(base.process_pair(p, verifier, vad_save_dir, 
                                            vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms, False))
            if log_every and (idx % log_every == 0 or idx == total):
                logging.info(f"GPU {gpu_id}: 进度 {idx}/{total} ({idx/total:.1%})")
        result_queue.put(results)
    except Exception as e:
        logging.error(f"GPU {gpu_id} 子进程失败: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put([])


def main():
    parser = argparse.ArgumentParser(description="Prompt-vs-Clone 声纹相似度计算（按子目录+JSON映射）")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="根目录（包含十个子目录和对应的JSON文件）")
    parser.add_argument("--threshold", type=float, default=0.90, help="相似度阈值 (默认: 0.90)")
    parser.add_argument("--inspect", action="store_true",
                        help="仅检查子目录与JSON对应关系并打印映射，然后退出")
    parser.add_argument("--wav_scp", type=str, action="append", default=[],
                        help="Kaldi wav.scp 文件路径，可多次传入用于定位prompt音频")
    parser.add_argument("--model_dir", type=str, default="/root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100",
                        help="Multilingual WeSpeaker 模型目录（samresnet100）")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="设备（默认自动）")
    parser.add_argument("--num_workers", type=int, default=8, help="单GPU线程池并发（仅在单GPU时使用）")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的GPU数量（多GPU分片并行）")
    parser.add_argument("--output", type=str, required=True, help="输出JSON路径")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    # 调试/可视化选项
    parser.add_argument("--debug", action="store_true", help="调试模式：随机打乱，仅取样若干条，保存波形+VAD图")
    parser.add_argument("--debug_samples", type=int, default=100, help="调试模式下采样条数")
    parser.add_argument("--debug_dir", type=str, default=None, help="调试输出目录（保存波形+VAD图）")
    # VAD参数
    parser.add_argument("--vad_frame_ms", type=int, default=16, help="TEN VAD 帧长(ms)")
    parser.add_argument("--vad_min_speech_ms", type=int, default=80, help="TEN VAD 最短语音段(ms)")
    parser.add_argument("--vad_max_silence_ms", type=int, default=160, help="TEN VAD 最长填补静音(ms)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.root_dir):
        logger.error(f"根目录不存在: {args.root_dir}")
        return 1

    # Inspect mode: print subdir <-> json mapping then exit
    if args.inspect:
        subdir_paths = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                        if os.path.isdir(os.path.join(args.root_dir, d))]
        subdirs = sorted([os.path.basename(p) for p in subdir_paths])
        json_files = sorted(glob.glob(os.path.join(args.root_dir, "*.json")))
        # count audio files per subdir
        subdir_audio_counts: Dict[str, int] = {}
        for sd in subdirs:
            stem_idx, base_idx = _scan_subdir_audio_indices(os.path.join(args.root_dir, sd))
            cnt = sum(len(v) for v in base_idx.values())
            subdir_audio_counts[sd] = cnt
        print("子目录（音频文件数量）:")
        for sd in subdirs:
            print(f"- {sd}: {subdir_audio_counts.get(sd, 0)}")
        print("\nJSON 映射（JSON -> 可能的子目录）:")
        for jf in json_files:
            json_stem = os.path.splitext(os.path.basename(jf))[0]
            best_sub = _pick_best_subdir_for_json(json_stem, subdirs)
            print(f"- {os.path.basename(jf)} -> {best_sub if best_sub else '(未匹配)'}")
        return 0

    pairs = build_pairs_from_prompt_root(args.root_dir, args.wav_scp if args.wav_scp else None)
    if not pairs:
        logger.error("未构建到有效的 (prompt, clone) 配对，结束")
        return 1

    # 调试模式：采样+保存VAD图
    save_debug_dir: Optional[str] = None
    if args.debug:
        import random
        random.seed(2025)
        random.shuffle(pairs)
        if args.debug_samples > 0 and len(pairs) > args.debug_samples:
            pairs = pairs[:args.debug_samples]
        save_debug_dir = args.debug_dir or (os.path.splitext(args.output)[0] + "_debug")
        os.makedirs(save_debug_dir, exist_ok=True)

    total_pairs = len(pairs)
    logger.info(f"开始相似度计算: {total_pairs} 对")

    start = time.time()
    results: List[Dict] = []

    # Decide device(s)
    if args.device == "cpu":
        usable_gpus = 0
    else:
        try:
            import torch
            usable_gpus = (torch.cuda.device_count() if torch.cuda.is_available() else 0)
        except Exception:
            usable_gpus = 0
    num_gpus = min(args.num_gpus, usable_gpus) if usable_gpus > 0 else 0
    if args.debug:
        logger.info("调试模式：使用单进程避免多GPU开销")
        num_gpus = 0

    # VAD params
    vad_frame_ms = int(args.vad_frame_ms)
    vad_min_speech_ms = int(args.vad_min_speech_ms)
    vad_max_silence_ms = int(args.vad_max_silence_ms)
    vad_save_dir = save_debug_dir

    if num_gpus and num_gpus > 1:
        logger.info(f"启用多GPU处理: {num_gpus} 卡")
        subsets = base._split_even(pairs, num_gpus)
        log_every = max(1, total_pairs // (num_gpus * 20))
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        result_queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []
        for gpu_id, subset in enumerate(subsets):
            if not subset:
                continue
            p = mp.Process(target=_worker_process_with_env,
                           args=(gpu_id, subset, args.model_dir, log_every, result_queue,
                                 vad_save_dir, vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms))
            p.start()
            processes.append(p)
        for _ in processes:
            try:
                timeout_s = max(60, total_pairs // 2)
                part = result_queue.get(timeout=timeout_s)
            except Exception:
                logging.error("等待GPU子进程结果超时，可能存在卡顿或异常。")
                part = []
            if part:
                results.extend(part)
        for p in processes:
            p.join(timeout=300)
        for p in processes:
            if p.is_alive():
                logging.warning(f"GPU 子进程仍在运行，强制终止: PID={p.pid}")
                p.terminate()
                p.join(timeout=30)
    else:
        # single GPU/CPU with threads
        if args.device in ("cuda", "cpu"):
            device_opt = args.device
        elif args.debug:
            logger.info("调试模式：强制使用 CPU 设备以提高稳定性")
            device_opt = "cpu"
        else:
            device_opt = "cuda" if num_gpus == 1 else "cpu"
        verifier = WeSpeakerVerification(model_dir=args.model_dir, device=device_opt)
        # 在 debug 模式下强制仅使用 array 提特征，避免触发外部解码库
        array_only = True if args.debug else False
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futures = [ex.submit(base.process_pair, p, verifier, vad_save_dir,
                                 vad_frame_ms, vad_min_speech_ms, vad_max_silence_ms, array_only) for p in pairs]
            completed = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                completed += 1
                if completed % 100 == 0 or completed == total_pairs:
                    logger.info(f"相似度计算进度: {completed}/{total_pairs} ({completed/total_pairs:.1%})")

    duration = time.time() - start
    logger.info(f"处理完成: {len(results)} 对，耗时 {duration:.2f}s")

    meta = {
        "flow": "prompt_vs_clone",
        "root_dir": args.root_dir,
        "model_dir": args.model_dir,
        "num_workers": args.num_workers,
        "num_gpus": num_gpus,
        "duration_sec": duration,
        "debug": {
            "enabled": bool(args.debug),
            "samples": int(args.debug_samples),
            "debug_dir": vad_save_dir
        },
        "vad": {
            "frame_ms": vad_frame_ms,
            "min_speech_ms": vad_min_speech_ms,
            "max_silence_ms": vad_max_silence_ms
        }
    }
    base.aggregate_and_save(results, args.threshold, args.output, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())


