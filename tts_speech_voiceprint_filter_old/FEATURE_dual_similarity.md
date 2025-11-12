# 双重相似度计算功能

## 功能概述

现在系统会对每个音频对计算**两次**相似度：

1. **原始音频相似度** (`similarity_original`): 在未经VAD处理的原始音频上计算
2. **VAD后音频相似度** (`similarity_vad`): 在VAD处理后的音频片段上计算

## 修改内容

### 1. `process_pair` 函数

- **之前**: 只计算一次相似度（在VAD处理后的音频上，如果VAD启用）
- **现在**: 
  - 先在原始音频上计算 `similarity_original`
  - 再在VAD处理后的音频上计算 `similarity_vad`（如果VAD可用且音频长度足够）
  - 保留 `similarity` 字段用于向后兼容，默认为VAD后的相似度

### 2. 返回结果字段

每个处理结果现在包含：

```json
{
  "voiceprint_id": "...",
  "prompt_id": "...",
  "source_path": "...",
  "tts_path": "...",
  "similarity": 0.85,              // 默认相似度（兼容旧版本）
  "similarity_original": 0.82,     // 新增：原始音频相似度
  "similarity_vad": 0.85,          // 新增：VAD后音频相似度
  "success": true,
  "vad": {
    "used": true,
    "frame_ms": 16,
    "min_speech_ms": 80,
    "max_silence_ms": 160,
    "src_active_ratio": 0.75,
    "tts_active_ratio": 0.80
  },
  "durations_sec": {
    "src_total": 2.5,
    "tts_total": 2.3,
    "src_used": 1.9,
    "tts_used": 1.8
  }
}
```

### 3. 统计信息

`aggregate_and_save` 函数现在输出三组统计信息：

- **similarity_stats**: 默认相似度统计（用于阈值判断）
- **similarity_original_stats**: 原始音频相似度统计（均值、中位数、标准差等）
- **similarity_vad_stats**: VAD后音频相似度统计

输出示例：

```json
{
  "statistics": {
    "total_pairs": 1000,
    "processed_pairs": 950,
    "passed_pairs": 800,
    "filtered_pairs": 150,
    "threshold": 0.70,
    "similarity_stats": {
      "mean": 0.85,
      "median": 0.87,
      "std": 0.12,
      "min": 0.45,
      "max": 0.98
    },
    "similarity_original_stats": {
      "mean": 0.82,
      "median": 0.84,
      "std": 0.13,
      "min": 0.40,
      "max": 0.97
    },
    "similarity_vad_stats": {
      "mean": 0.85,
      "median": 0.87,
      "std": 0.12,
      "min": 0.45,
      "max": 0.98
    }
  }
}
```

## 使用场景

这个功能让你可以：

1. **对比分析**: 观察VAD对相似度计算的影响
2. **灵活筛选**: 可根据两种相似度中的任意一个或组合进行后处理筛选
3. **调优参考**: 帮助调整VAD参数（frame_ms, min_speech_ms, max_silence_ms）
4. **问题诊断**: 当VAD后相似度异常低时，可参考原始相似度判断是VAD问题还是音频本身质量问题

## 典型观察

- **正常情况**: `similarity_vad` ≥ `similarity_original` (VAD去除了静音/噪声，提高了纯净度)
- **VAD过度**: `similarity_vad` << `similarity_original` (可能VAD参数过于激进，切掉了有效语音)
- **音频过短**: `similarity_vad` = 0.0, `similarity_original` > 0 (VAD后剩余片段不足)

## 向后兼容

- 保留了 `similarity` 字段，默认值为 `similarity_vad`（如果可用），否则为 `similarity_original`
- 现有的阈值筛选逻辑（`--threshold`）继续使用 `similarity` 字段
- 旧的脚本和工具可无缝继续使用

## 修改范围

- ✅ `tts_speech_voiceprint_filter/compute_similarity.py`
- ✅ `tts_speech_voiceprint_filter_old/compute_similarity.py`
- ✅ 所有使用 `process_pair` 的脚本（包括 `compute_similarity_prompts.py`）会自动继承此功能

---

**更新日期**: 2025-11-12

