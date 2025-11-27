# SenseVoice 模式使用指南

## 快速开始

### 1. 单数据集处理（推荐）

使用专用脚本：
```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

./run_single_tts_filter_sensevoice.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh
```

或使用通用脚本：
```bash
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --use_sensevoice \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh
```

### 2. 处理 voiceclone_20250804 数据集

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 默认使用 SenseVoice（自动检测）
./process_voiceclone_20250804.sh \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh

# 强制使用 SenseVoice
./process_voiceclone_20250804.sh \
    --use_sensevoice \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh

# 处理指定 part
./process_voiceclone_20250804.sh \
    --use_sensevoice \
    --process_parts "1,2,3" \
    --cer_threshold 0.05 \
    --num_gpus 4
```

### 3. 合并 JSON 并筛选

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 默认使用 SenseVoice（自动检测）
./combine_and_filter.sh \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh

# 强制使用 SenseVoice
./combine_and_filter.sh \
    --use_sensevoice \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh
```

## 参数说明

### 通用参数

- `--cer_threshold <float>`: CER 阈值，默认 0.05（5%）
- `--num_gpus <int>`: 使用的 GPU 数量，默认使用所有可用 GPU
- `--language <auto|zh|en>`: 文本语言，默认 auto（自动检测）
- `--sensevoice_model_dir <dir>`: SenseVoice 模型路径或ID，默认 `iic/SenseVoiceSmall`
- `--output <path>`: 输出结果文件路径
- `--verbose`: 输出详细日志
- `--test_mode`: 测试模式，只处理前10个prompt
- `--debug_mode`: 调试模式，限制样本数量
- `--force`: 强制重新处理已有结果

### process_voiceclone_20250804.sh 特有参数

- `--process_parts <parts>`: 指定处理哪些 part，如 "1,2,5" 或 "all"（默认 all）
- `--merge`: 合并所有 JSON 后统一处理（默认分别处理）

## 使用示例

### 示例1：处理单个数据集（中文）

```bash
./run_single_tts_filter_sensevoice.sh \
    /root/group-shared/voiceprint/share/voiceclone_child_20251022/voiceprint_20251022_part1_20251022 \
    /root/group-shared/voiceprint/share/voiceclone_child_20251022/voiceprint_20251022_part1_20251022.json \
    --cer_threshold 0.05 \
    --num_gpus 2 \
    --language zh \
    --output ./results/sensevoice_part1.json
```

### 示例2：处理 voiceclone_20250804（所有 part）

```bash
./process_voiceclone_20250804.sh \
    --use_sensevoice \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh \
    --process_parts all
```

### 示例3：处理指定 part（1, 2, 5）

```bash
./process_voiceclone_20250804.sh \
    --use_sensevoice \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --language zh \
    --process_parts "1,2,5"
```

### 示例4：测试模式（少量数据）

```bash
./run_single_tts_filter_sensevoice.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --test_mode \
    --cer_threshold 0.05 \
    --num_gpus 1 \
    --language zh
```

### 示例5：调试模式

```bash
./run_single_tts_filter_sensevoice.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --debug_mode \
    --debug_samples 100 \
    --cer_threshold 0.05 \
    --num_gpus 1
```

## 模式选择

### 自动检测（默认）

脚本会自动检测可用依赖，按优先级选择：
1. **SenseVoice**（如果 funasr 可用）← **默认优先**
2. Whisper+LLM（如果 LLM 服务可用）
3. Kimi-Audio（回退选项）

### 强制指定模式

```bash
# 强制使用 SenseVoice
--use_sensevoice

# 强制使用 Whisper+LLM
--use_whisper

# 强制使用 Kimi-Audio
--no-use_whisper
```

## 输出结果

结果文件包含：
- `filter_results`: 每个音频的详细结果
- `statistics`: 统计信息（总数、通过数、筛除数、CER统计等）
- `*_filtered_list.txt`: 被筛选掉的音频文件列表

## 常见问题

### Q: 如何检查 funasr 是否安装？

```bash
python3 -c "import funasr; print('funasr已安装')"
```

### Q: 如何检查 NeMo 文本标准化是否安装？

```bash
python3 -c "import nemo_text_processing; print('NeMo文本标准化已安装')"
```

### Q: 模型会自动下载吗？

是的，SenseVoice 模型会在首次使用时自动从 HuggingFace 下载到 `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall`

### Q: 如何查看处理进度？

添加 `--verbose` 参数查看详细日志。

### Q: 如何增量处理？

默认启用增量处理，已处理的音频会自动跳过。使用 `--force` 强制重新处理。

