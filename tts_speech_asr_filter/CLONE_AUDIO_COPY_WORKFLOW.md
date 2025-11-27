# TTS克隆音频复制到数据集目录 - 完整流程

## 📋 流程概述

这个流程用于将经过双重筛选（ASR + 声纹）的TTS克隆音频，根据prompt音频id对应的原始数据集和说话人，拷贝回到标准的数据集目录结构中。

## 🎯 目标

将以下两个目录中的筛选后音频：
- `/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech`
- `/root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech`

按照原始数据集结构，拷贝到目标目录：
- `/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio`

## 📊 数据集映射关系

| 数据集名称 | 目标子目录 | utt2spk路径 |
|-----------|-----------|-------------|
| BAAI-ChildMandarin | `childmandarin` | `/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/utt2spk` |
| Chinese-English-Children | `chineseenglishchildren` | `/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/utt2spk` |
| King-ASR-612 | `kingasr612` | `/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/utt2spk` |
| King-ASR-725 | `king-asr-725` | 同上 |
| SpeechOcean762 | `speechocean762` | `/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/utt2spk` |

## 🔄 工作流程

### 1. 准备阶段

脚本会自动执行以下步骤：

1. **加载utt2spk映射**
   - 从4个数据集的utt2spk文件加载prompt_id到speaker_id的映射关系
   - utt2spk格式：`<prompt_id> <speaker_id>`

2. **扫描源目录**
   - 扫描两个filtered_speech目录下的`audio/`子目录
   - 收集所有通过筛选的音频文件
   - 提取prompt_id和voiceprint_id信息

### 2. 复制阶段

对每个音频文件：

1. **查找映射关系**
   - 根据prompt_id在utt2spk映射中查找对应的数据集和speaker_id
   - 确定目标子目录（对King-ASR需要根据speaker_id判断是612还是725）

2. **构建目标路径**
   ```
   目标路径 = <output_base_dir>/<dataset_subdir>/<speaker_id>/<voiceprint_id>.wav
   ```

3. **复制文件**
   - 使用多进程并行复制（默认16个进程）
   - 自动创建必要的目录结构
   - 记录复制结果

### 3. 报告阶段

生成三个报告文件（保存在`<output_base_dir>/copy_reports/`）：

1. **copy_report.json** - 详细的复制记录（JSON格式）
2. **copy_list.txt** - 源路径到目标路径的映射列表
3. **copy_summary.txt** - 统计摘要（文本格式）

## 🚀 使用方法

### 基本使用（使用默认配置）

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter
./run_copy_clone_audio.sh
```

### 模拟运行（查看将会如何复制，不实际复制文件）

```bash
./run_copy_clone_audio.sh --dry_run

# 自定义打印间隔（例如每50条打印一个）
./run_copy_clone_audio.sh --dry_run --print_interval 50
```

### 自定义源目录

```bash
./run_copy_clone_audio.sh \
  --source_dirs /path/to/filtered_speech1 /path/to/filtered_speech2
```

### 自定义目标目录和进程数

```bash
./run_copy_clone_audio.sh \
  --output_base_dir /path/to/output \
  --num_workers 32
```

### 查看所有选项

```bash
./run_copy_clone_audio.sh --help
```

## 📂 目录结构示例

### 输入结构（filtered_speech）

```
filtered_speech/
└── audio/
    ├── 001_5_M_L_LANZHOU_Android_021/
    │   ├── 001.wav
    │   ├── 002.wav
    │   └── ...
    ├── G0001_0_S0001/
    │   ├── G0001.wav
    │   └── ...
    └── ...
```

### 输出结构（目标目录）

```
merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio/
├── childmandarin/
│   ├── 001/
│   │   ├── 001.wav
│   │   ├── 002.wav
│   │   └── ...
│   └── 002/
│       └── ...
├── chineseenglishchildren/
│   ├── G0001/
│   │   ├── G0001.wav
│   │   └── ...
│   └── ...
├── king-asr-725/
│   └── ...
├── kingasr612/
│   └── ...
├── speechocean762/
│   └── ...
└── copy_reports/
    ├── copy_report.json
    ├── copy_list.txt
    └── copy_summary.txt
```

## 📊 统计报告示例

生成的`copy_summary.txt`包含：

```
================================================================================
TTS克隆音频复制到数据集目录 - 统计报告
================================================================================

生成时间: 2025-11-17 17:30:00

总体统计:
  总音频数: 50000
  成功复制: 49800
  失败: 200
  成功率: 99.60%

各数据集分布:
  childmandarin: 25000 (50.20%)
  chineseenglishchildren: 15000 (30.12%)
  speechocean762: 5000 (10.04%)
  kingasr612: 3000 (6.02%)
  king-asr-725: 1800 (3.61%)

================================================================================
```

## 🔍 关键特性

1. **多进程并行复制** - 使用ProcessPoolExecutor并行处理，大幅提升速度
2. **自动目录创建** - 自动创建所需的目录结构
3. **完整的错误处理** - 记录失败的文件和原因
4. **详细的统计报告** - 按数据集统计分布情况
5. **模拟运行模式** - 可以先查看将会如何复制，确认无误后再实际执行
6. **示例打印功能** - dry_run模式下每隔n条音频打印一个复制示例，直观查看映射关系

## 📺 示例输出（dry_run模式）

运行`./run_copy_clone_audio.sh --dry_run --print_interval 50`时，你会看到类似输出：

```
================================================================================
   TTS克隆音频复制到数据集目录
================================================================================
源目录:
  - /root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech
  - /root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech
目标目录: /root/group-shared/.../merged_datasets_.../audio
工作进程数: 16
模拟运行: 是（不实际复制文件）
打印间隔: 每 50 条打印一个示例

2025-11-17 18:00:00 - INFO - 加载utt2spk映射...
2025-11-17 18:00:00 - INFO - 加载 childmandarin: .../utt2spk
2025-11-17 18:00:00 - INFO - 总共加载 50000 个prompt_id映射
2025-11-17 18:00:00 - INFO - 扫描源目录...
2025-11-17 18:00:00 - INFO - 总共找到 10000 个音频文件
2025-11-17 18:00:00 - INFO - 准备模拟复制 10000 个音频文件，使用 16 个工作进程
2025-11-17 18:00:00 - INFO - dry_run模式：每隔 50 条音频打印一个复制示例

2025-11-17 18:00:01 - INFO - 
[示例 #50]
  Prompt ID:     001_5_M_L_LANZHOU_Android_021
  Voiceprint ID: 001
  数据集:        childmandarin
  说话人:        001
  源文件:        /root/.../filtered_speech/audio/001_5_M_L_LANZHOU_Android_021/001.wav
  目标文件:      /root/.../audio/childmandarin/001/001.wav

2025-11-17 18:00:02 - INFO - 
[示例 #100]
  Prompt ID:     G0001_0_S0001
  Voiceprint ID: G0001
  数据集:        chineseenglishchildren
  说话人:        G0001
  源文件:        /root/.../filtered_speech/audio/G0001_0_S0001/G0001.wav
  目标文件:      /root/.../audio/chineseenglishchildren/G0001/G0001.wav

...

2025-11-17 18:00:10 - INFO - 模拟复制进度: 10000/10000 (100.0%)
2025-11-17 18:00:10 - INFO - 
模拟复制完成统计:
  成功: 9950
  失败: 50

各数据集分布:
  childmandarin: 5000
  chineseenglishchildren: 3000
  speechocean762: 1500
  kingasr612: 300
  king-asr-725: 150
```

## ⚠️ 注意事项

1. **磁盘空间** - 确保目标目录有足够的磁盘空间（约与源目录相同）
2. **权限** - 确保对源目录有读权限，对目标目录有写权限
3. **文件覆盖** - 如果目标文件已存在，将会被覆盖
4. **映射缺失** - 如果prompt_id在utt2spk中找不到映射，该文件会被跳过并记录为失败

## 🛠️ 技术细节

### utt2spk格式

```
<utterance_id> <speaker_id>
```

示例：
```
001_5_M_L_LANZHOU_Android_021 001
G0001_0_S0001 G0001
King-ASR-612_000080001 King-ASR-612_SPEAKER0008
```

### 文件命名规则

- 源文件：`<prompt_id>/<voiceprint_id>.wav`
- 目标文件：`<dataset_subdir>/<speaker_id>/<voiceprint_id>.wav`

### King-ASR数据集特殊处理

King-ASR包含两个子集（612和725），通过speaker_id前缀区分：
- 如果speaker_id包含`King-ASR-612`，复制到`kingasr612`
- 如果speaker_id包含`King-ASR-725`，复制到`king-asr-725`

## 📞 相关脚本

1. **run_merge_filter.sh** - 执行ASR和声纹双重筛选
2. **merge_filter_results.py** - 合并筛选结果并复制通过的音频
3. **run_copy_clone_audio.sh** - 本流程的Shell包装脚本
4. **copy_clone_audio_to_dataset.py** - 本流程的Python实现

## 🔗 完整工作流

```
原始音频 
  → TTS克隆生成
  → ASR筛选 (run_asr_filter.sh)
  → 声纹筛选 (run_voiceprint_filter.sh)
  → 双重筛选合并 (run_merge_filter.sh)
  → 复制到filtered_speech
  → 复制回数据集目录 (run_copy_clone_audio.sh) ← 当前步骤
```

## ✅ 验证结果

复制完成后，建议：

1. 检查统计报告：`cat <output_dir>/copy_reports/copy_summary.txt`
2. 验证文件数量：
   ```bash
   # 统计各数据集的文件数
   find <output_dir>/childmandarin -name "*.wav" | wc -l
   find <output_dir>/chineseenglishchildren -name "*.wav" | wc -l
   # ...以此类推
   ```
3. 随机抽查几个文件，确认路径和命名正确
4. 检查是否有失败的文件，分析失败原因

## 📝 日志和调试

- 默认日志级别：INFO
- 使用`--verbose`开启DEBUG级别日志
- 使用`--dry_run`模拟运行，查看将会执行的操作但不实际复制

## 🎓 示例运行

```bash
# 1. 先模拟运行，查看将会如何复制
./run_copy_clone_audio.sh --dry_run

# 2. 确认无误后，正式执行
./run_copy_clone_audio.sh

# 3. 查看结果摘要
cat /root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio/copy_reports/copy_summary.txt
```

