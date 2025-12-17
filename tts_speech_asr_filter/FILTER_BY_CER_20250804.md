# TTS音频按CER分类筛选流程（20250804数据集）

## 概述

基于ASR筛选结果，按CER（字符错误率）值对TTS音频进行分类筛选，将符合条件的音频按CER范围分类存储。

## 输入文件

- **ASR结果JSON**: `/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_asr_filter_sensevoice/results/tts_asr_filter_merged_all_parts_20251210_100140.json`
- 包含字段：
  - `filter_results`: 筛选结果列表
  - 每个结果项包含：`audio_path`, `prompt_id`, `voiceprint_id`, `cer`, `groundtruth_text`, `transcription`等

## 输出目录

- **输出目录**: `/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech_sensevoice_cer0.25`

## 目录结构

```
filtered_speech_sensevoice_cer0.25/
├── cer0/                    # CER == 0
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
├── cer0-0.05/               # 0 < CER <= 0.05
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
├── cer0.05-0.1/             # 0.05 < CER <= 0.1
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
├── cer0.1-0.15/             # 0.1 < CER <= 0.15
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
├── cer0.15-0.2/             # 0.15 < CER <= 0.2
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
├── cer0.2-0.25/             # 0.2 < CER <= 0.25
│   └── <prompt_id>/
│       ├── <voiceprint_id>.wav
│       └── <voiceprint_id>.json
└── filter_summary.txt       # 筛选统计摘要
```

## JSON文件内容

每个音频文件旁边会生成对应的JSON文件，包含以下信息：

```json
{
  "voiceprint_id": "voiceprint_xxx",
  "prompt_id": "prompt_xxx",
  "audio_path": "/path/to/original/audio.wav",
  "groundtruth_text": "原始文本",
  "transcription": "ASR识别文本",
  "normalized_groundtruth": "标准化后的原始文本",
  "normalized_transcription": "标准化后的识别文本",
  "cer": 0.05,
  "cer_range": "cer0-0.05",
  "language": "zh",
  "timestamp": "2025-01-XX..."
}
```

## 使用方法

### 1. 使用默认配置

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter
./run_filter_by_cer_20250804.sh
```

### 2. 自定义参数

```bash
# 自定义CER阈值和工作进程数
./run_filter_by_cer_20250804.sh --cer_threshold 0.20 --num_workers 32

# 指定输入和输出路径
./run_filter_by_cer_20250804.sh \
  --asr_result /path/to/asr_result.json \
  --output_dir /path/to/output

# 启用详细日志
./run_filter_by_cer_20250804.sh --verbose
```

### 3. 查看帮助

```bash
./run_filter_by_cer_20250804.sh --help
```

## 筛选规则

- **CER阈值**: 默认0.25，只处理CER <= 0.25的音频
- **CER分类**:
  - `cer0`: CER == 0
  - `cer0-0.05`: 0 < CER <= 0.05
  - `cer0.05-0.1`: 0.05 < CER <= 0.1
  - `cer0.1-0.15`: 0.1 < CER <= 0.15
  - `cer0.15-0.2`: 0.15 < CER <= 0.2
  - `cer0.2-0.25`: 0.2 < CER <= 0.25
  - `cer0.25+`: CER > 0.25（不会复制，因为超过阈值）

## 处理流程

1. **加载ASR结果**: 从JSON文件读取筛选结果
2. **筛选音频**: 只保留CER <= 阈值的音频
3. **分类复制**: 按CER范围分类，复制到对应目录
4. **生成JSON**: 为每个音频生成包含元信息的JSON文件
5. **生成统计**: 生成筛选统计摘要文件

## 统计信息

处理完成后，会生成 `filter_summary.txt` 文件，包含：
- 总音频数
- CER分布统计
- 各CER范围的音频数量
- 处理成功/失败统计
- CER统计（平均值、中位数、标准差等）

## 查看结果

```bash
# 查看统计摘要
cat /root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech_sensevoice_cer0.25/filter_summary.txt

# 查看各CER范围的音频数量
find /root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech_sensevoice_cer0.25 -type d -name "cer*" | while read dir; do
  echo "$(basename $dir): $(find $dir -name "*.wav" | wc -l) 个音频"
done
```

## 注意事项

1. 确保有足够的磁盘空间存储筛选后的音频文件
2. 处理大量文件时，建议使用足够的并行工作进程数（默认16）
3. 如果源音频文件不存在，会跳过该文件并记录错误
4. 处理过程会显示进度信息，包括处理速度和预计剩余时间

## 相关脚本

- `filter_by_cer_ranges.py`: 核心处理脚本
- `run_filter_by_cer_20250804.sh`: 20250804数据集的包装脚本
- `run_filter_by_cer.sh`: 通用包装脚本（20251022数据集）

