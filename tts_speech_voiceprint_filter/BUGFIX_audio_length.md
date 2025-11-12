# Bug修复说明：音频长度检查

## 问题描述

运行声纹相似度计算时可能出现错误：

```
AssertionError: choose a window size 400 that is [2, 0]
```

**根本原因：**
- 某些音频文件经过 VAD 处理后变成空数组（长度为 0）
- WeSpeaker 提取 fbank 特征时需要 window_size=400 样本（约 25ms @ 16kHz）
- 传入长度为 0 的音频导致断言失败

## 修复方案

在 `compute_similarity.py` 的 `process_pair()` 函数中添加了三层防护：

### 1. **原始音频长度检查**（第 286-305 行）
```python
min_audio_len = 500  # 至少 31ms @ 16kHz
if src_audio.size < min_audio_len or tts_audio.size < min_audio_len:
    return {
        "success": False,
        "error_message": f"Audio too short: src={src_audio.size} tts={tts_audio.size}",
        ...
    }
```

### 2. **VAD 后长度检查**（第 329-347 行）
```python
# VAD 后再次检查长度（防止 VAD 后为空）
if src_audio_use.size < min_audio_len or tts_audio_use.size < min_audio_len:
    return {
        "success": False,
        "error_message": f"Audio too short after VAD: src={src_audio_use.size} tts={tts_audio_use.size}",
        ...
    }
```

### 3. **异常捕获**（第 407-426 行）
```python
except Exception as e:
    logger.error(f"处理音频对失败 [{vp_id}/{prompt_id}]: {e}")
    return {
        "success": False,
        "error_message": str(e),
        ...
    }
```

## 修复的文件

- ✅ `tts_speech_voiceprint_filter_old/compute_similarity.py`
- ✅ `tts_speech_voiceprint_filter/compute_similarity.py`

## 影响

1. **兼容性：** 不影响正常音频的处理
2. **健壮性：** 短音频/损坏音频会被优雅地跳过并记录
3. **统计：** 失败的样本会在结果 JSON 中标记 `"success": false`
4. **日志：** 错误信息会记录到日志中便于排查

## 测试建议

重新运行脚本，观察：
1. 脚本能否正常完成而不崩溃
2. 日志中 `error_message` 字段记录的短音频信息
3. 最终统计中 `failed_pairs` 的数量

## 相关问题

如果大量音频被标记为 "too short"，可能需要检查：
- VAD 参数是否过于严格（`vad_min_speech_ms`, `vad_max_silence_ms`）
- 原始音频质量（是否有大量静音或损坏文件）
- 音频格式兼容性


