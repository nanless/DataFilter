# Debug 模式使用说明

## 问题修复

### MagicMock 错误修复

之前运行时出现的 `'<' not supported between instances of 'MagicMock' and 'int'` 错误已经修复。

**修复内容：**
1. 更新 `multilingual_inference.py`，在模块级别设置环境变量和 Mock torio 扩展
2. 移除 `compute_similarity_prompts.py` 中的重复 Mock 代码
3. 简化 `_load_model` 方法，移除不必要的异常处理

**技术细节：**
- 在导入 torch/torchaudio 之前设置环境变量：
  ```python
  os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
  os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
  os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"
  ```
- 在模块级别 Mock torio 扩展，防止段错误
- 使用 WeSpeaker 的 `extract_embedding_from_pcm` 方法直接从数组提取 embedding

## Debug 模式使用

### 1. 使用 Shell 脚本运行

最简单的方式是使用 `run_voiceprint_filter.sh` 脚本：

```bash
# 启用 debug 模式，使用 100 个样本
./run_voiceprint_filter.sh --debug --debug_samples 100

# 指定调试输出目录
./run_voiceprint_filter.sh --debug --debug_samples 100 --debug_dir ./debug_output

# 完整示例
./run_voiceprint_filter.sh \
  --prompt_root /root/group-shared/voiceprint/share/voiceclone_child_20250804 \
  --threshold 0.9 \
  --num_workers 4 \
  --debug \
  --debug_samples 100 \
  --verbose
```

### 2. 直接使用 Python 脚本

也可以直接调用 Python 脚本：

```bash
python3 compute_similarity_prompts.py \
  --root_dir /root/group-shared/voiceprint/share/voiceclone_child_20250804 \
  --threshold 0.9 \
  --model_dir /root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100 \
  --num_workers 4 \
  --output ./results/debug_test.json \
  --debug \
  --debug_samples 100 \
  --debug_dir ./debug_output \
  --verbose
```

### 3. Debug 模式参数说明

- `--debug`: 启用调试模式
- `--debug_samples INT`: 调试模式下采样的数量（默认：100）
- `--debug_dir DIR`: 调试输出目录（保存波形+VAD图等，默认：输出文件名_debug）

### 4. Debug 模式功能

启用 debug 模式后：

1. **随机采样**：从所有配对中随机抽取指定数量的样本
2. **保存 VAD 图**：为每对音频生成波形+VAD叠加图
   - `<prompt_id>__<voiceprint_id>__<uid>_src.png`：源音频波形+VAD
   - `<prompt_id>__<voiceprint_id>__<uid>_tts.png`：TTS音频波形+VAD
3. **单进程模式**：避免多GPU spawn 带来的加载/同步开销
4. **CPU 模式**：默认强制使用 CPU 设备以提高稳定性

### 5. 输出文件

Debug 模式会生成以下文件：

```
debug_output/
├── <prompt_id>__<voiceprint_id>__<uid1>_src.png
├── <prompt_id>__<voiceprint_id>__<uid1>_tts.png
├── <prompt_id>__<voiceprint_id>__<uid2>_src.png
├── <prompt_id>__<voiceprint_id>__<uid2>_tts.png
└── ...

results/
├── debug_test.json                    # 完整结果（含VAD信息、相似度等）
└── debug_test_filtered_list.txt      # 筛除的TTS音频列表
```

### 6. 查看结果

```bash
# 查看统计信息
cat results/debug_test.json | jq '.statistics'

# 查看失败的样本
cat results/debug_test.json | jq '.filter_results[] | select(.success == false)'

# 查看低相似度样本
cat results/debug_test.json | jq '.filter_results[] | select(.similarity < 0.7)'

# 查看 VAD 使用情况
cat results/debug_test.json | jq '.filter_results[] | .vad'
```

## 配置文件

`config.json` 中包含 debug 配置：

```json
{
  "runtime": {
    "debug": {
      "enabled": false,
      "samples": 100,
      "dir": ""
    }
  }
}
```

## 常见问题

### Q: 为什么 debug 模式使用 CPU？
A: CPU 模式更稳定，避免 CUDA 初始化可能导致的问题。如需使用 GPU，可显式指定 `--device cuda`。

### Q: 可以同时使用多 GPU 吗？
A: Debug 模式默认单进程以简化调试。正式运行时可使用 `--num_gpus` 参数启用多 GPU。

### Q: VAD 图保存在哪里？
A: 默认保存在 `<output_file>_debug/` 目录下，或通过 `--debug_dir` 指定。

## 性能对比

| 模式 | 样本数 | 设备 | GPU数 | 耗时（估计） |
|------|--------|------|-------|-------------|
| Debug | 100 | CPU | - | ~10-20秒 |
| Debug | 100 | GPU | 1 | ~5-10秒 |
| 生产 | 全部 | GPU | 8 | ~5-30分钟 |

## 修改日期

2025-11-12

## 参考

- 新版本修复日志：`../tts_speech_voiceprint_filter/FIX_LOG.md`
- Shell 脚本：`run_voiceprint_filter.sh`
- Python 脚本：`compute_similarity_prompts.py`

