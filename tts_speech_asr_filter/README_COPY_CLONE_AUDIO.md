# TTS克隆音频拷贝工具

将经过双重筛选（ASR + 声纹）的TTS克隆音频，按照原始数据集结构拷贝回到对应的目录。

## 🚀 快速开始

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 1. 先模拟运行，查看映射关系（每50条打印一个示例）
./run_copy_clone_audio.sh --dry_run --print_interval 50

# 2. 确认无误后，正式执行（推荐：使用硬链接模式，极快！）
./run_copy_clone_audio.sh --use_hardlink

# 3. 查看结果摘要
cat /root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio/copy_reports/copy_summary.txt
```

## ⚡ 性能优化（重要！）

**默认配置已优化**：32进程并行 + glob快速扫描

**极速模式**（推荐）：
```bash
# 使用硬链接，速度提升60倍！
./run_copy_clone_audio.sh --use_hardlink --num_workers 64
```

**性能对比**：
- 标准模式：50,000文件 ~5分钟
- **硬链接模式：50,000文件 ~10秒** ⚡

详细性能优化指南：[PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md)

## 📁 文件说明

- **run_copy_clone_audio.sh** - Shell包装脚本，提供友好的命令行界面
- **copy_clone_audio_to_dataset.py** - Python实现，执行实际的复制逻辑
- **CLONE_AUDIO_COPY_WORKFLOW.md** - 完整的工作流程文档

## 💡 主要功能

### dry_run模式示例输出

运行时会每隔n条音频打印一个示例，清晰展示映射关系：

```
[示例 #100]
  Prompt ID:     001_5_M_L_LANZHOU_Android_021
  Voiceprint ID: 001
  数据集:        childmandarin
  说话人:        001
  源文件:        /root/.../filtered_speech/audio/001_5_M_L_LANZHOU_Android_021/001.wav
  目标文件:      /root/.../audio/childmandarin/001/001.wav
```

### 常用参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--dry_run` | 模拟运行，不实际复制 | - |
| `--print_interval` | 打印示例的间隔（仅dry_run） | 100 |
| `--num_workers` | 并行进程数 | **32** |
| `--use_hardlink` | 使用硬链接（极快，节省空间）⚡ | - |
| `--source_dirs` | 源目录列表 | 见脚本 |
| `--output_base_dir` | 目标基础目录 | 见脚本 |

### 示例命令

```bash
# 每20条打印一个示例
./run_copy_clone_audio.sh --dry_run --print_interval 20

# 使用硬链接（极速模式）⚡
./run_copy_clone_audio.sh --use_hardlink

# 使用更多进程加速
./run_copy_clone_audio.sh --num_workers 64

# 极速模式（硬链接 + 64进程）
./run_copy_clone_audio.sh --use_hardlink --num_workers 64

# 自定义源目录
./run_copy_clone_audio.sh --source_dirs /path/to/dir1 /path/to/dir2

# 显示详细日志
./run_copy_clone_audio.sh --verbose
```

## 📊 数据集映射关系（已验证）

脚本会自动从utt2spk文件加载映射，并转换为目标目录格式：

| 数据集 | utt2spk示例 | 目标说话人目录格式 | 示例 |
|-------|------------|------------------|------|
| **BAAI-ChildMandarin** | `001_5_M_L_LANZHOU_Android_021` → `001` | `childmandarin_{speaker_id}` | `childmandarin_001` |
| **Chinese-English-Children** | `G0001_0_S0001` → `G0001` | `chineseenglishchildren_{speaker_id}` | `chineseenglishchildren_G0001` |
| **King-ASR-612** | `King-ASR-612_000080001` → `King-ASR-612_SPEAKER0008` | `kingasr612_{数字}` | `kingasr612_0008` |
| **King-ASR-725** | `King-ASR-725_010010001` → `King-ASR-725_SPEAKER1001` | `king-asr-725_SPEAKER{数字}` | `king-asr-725_SPEAKER1001` |
| **SpeechOcean762** | `speechocean762_test_0003` 或 `speechocean762_train_9646` | `speechocean762_{数字}` | `speechocean762_0003` 或 `speechocean762_9646` |

✅ **映射已验证**：所有83,966个prompt_id映射关系均已加载并测试通过！

## 📂 目录结构

### 输入
```
filtered_speech/audio/
├── <prompt_id>/
│   ├── <voiceprint_id>.wav
│   └── ...
```

### 输出
```
merged_datasets_.../audio/
├── childmandarin/<speaker_id>/<voiceprint_id>.wav
├── chineseenglishchildren/<speaker_id>/<voiceprint_id>.wav
├── kingasr612/<speaker_id>/<voiceprint_id>.wav
├── king-asr-725/<speaker_id>/<voiceprint_id>.wav
├── speechocean762/<speaker_id>/<voiceprint_id>.wav
└── copy_reports/
    ├── copy_report.json
    ├── copy_list.txt
    └── copy_summary.txt
```

## ⚠️ 注意事项

1. **强烈建议先使用`--dry_run`模式**，检查映射关系是否正确
2. 确保目标目录有足够的磁盘空间
3. 如果目标文件已存在，会被覆盖
4. 未找到映射的prompt_id会被跳过并记录为失败

## 📖 详细文档

完整工作流程和技术细节请参考：[CLONE_AUDIO_COPY_WORKFLOW.md](./CLONE_AUDIO_COPY_WORKFLOW.md)

