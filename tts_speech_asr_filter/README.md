# TTS语音筛选系统

本系统用于筛选TTS合成的音频，支持两种ASR模式：
1. **Whisper+LLM模式**（默认）：使用OpenAI Whisper进行语音识别，LLM服务进行高质量文本标准化
2. **Kimi-Audio模式**：使用Kimi-Audio进行语音识别，WeTextProcessing进行文本标准化

通过比对识别结果与groundtruth文本的CER（字符错误率）来判断TTS合成质量。CER超过阈值的音频将被筛选掉。

## 系统特点

- **多ASR支持**：可选择Kimi-Audio或Whisper进行语音识别
- **多GPU并行处理**：自动检测可用GPU数量，支持多卡并行处理
- **文本标准化**：
  - Kimi模式：使用WeTextProcessing进行中英文文本标准化（TN处理）
  - Whisper模式：使用LLM服务进行更精确的文本标准化
- **语言支持**：支持自动检测语言或手动指定中文/英文
- **高效处理**：
  - Kimi模式：所有GPU都用于语音识别
  - Whisper模式：每个GPU独立运行ASR和LLM服务
- **批量处理**：支持批量处理多个数据集
- **详细统计**：提供详细的处理统计和筛选结果
- **实时日志**：处理过程中输出识别结果、语言设置、CER等详细信息

## 目录结构

```
tts_speech_filter/
├── tts_filter_by_kimi_asr.py     # Kimi-Audio ASR主程序
├── tts_filter_by_whisper_asr.py  # Whisper ASR主程序
├── config.json                    # 配置文件
├── run_single_tts_filter.sh       # 单个数据集处理脚本
├── run_all_tts_filter.sh          # 批量处理脚本
├── start_filter_all.sh            # 快捷启动脚本（自动激活环境）
├── quick_test.sh                  # 快速测试脚本
├── test_whisper_llm.sh            # Whisper+LLM模式测试脚本
├── run_with_kimi_env.sh           # 环境包装脚本
├── auto_start_llm_services.sh     # LLM服务启动脚本
├── stop_multi_llm_services.sh     # LLM服务停止脚本
├── llm_service.py                 # LLM文本标准化服务
├── README.md                      # 本文档
├── QUICK_START.md                 # 快速开始指南
├── results/                       # 结果输出目录
└── logs/                          # 日志文件目录
```

## 模式选择建议

### Whisper+LLM模式（默认推荐）
- **优点**：
  - Whisper模型识别准确率高，支持多种语言
  - LLM提供更智能的文本标准化，能处理复杂的文本变体
  - 处理每条音频时明确输出语言信息
  - 适合高精度要求的筛选场景
- **缺点**：需要启动额外的LLM服务，每个GPU需要运行独立的LLM实例
- **适用场景**：高精度要求的筛选，混合语言文本，复杂文本格式，英文TTS筛选

### Kimi-Audio模式
- **优点**：部署简单，所有GPU都用于ASR，适合大规模批量处理
- **缺点**：文本标准化能力相对有限（使用规则基础的WeTextProcessing）
- **适用场景**：标准的中文TTS筛选，对文本标准化要求不高的场景，或不想启动LLM服务的情况

## 数据格式要求

### 1. 目录结构
```
/path/to/dataset/
├── zero_shot/
│   ├── prompt_id_1/
│   │   ├── voiceprint_id_1.wav
│   │   ├── voiceprint_id_2.wav
│   │   └── ...
│   ├── prompt_id_2/
│   │   └── ...
│   └── ...
```

### 2. JSON文件格式
```json
{
  "prompt_id_1": [
    "voiceprint_id_1\t文本内容1",
    "voiceprint_id_2\t文本内容2",
    ...
  ],
  "prompt_id_2": [
    ...
  ]
}
```

## 安装依赖

1. 确保已安装Kimi-Audio环境：
```bash
conda activate kimi-audio
```

2. 安装WeTextProcessing：
```bash
cd /root/code/github_repos
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
pip install -e .
```

3. 其他依赖：
```bash
pip install jiwer tqdm soundfile numpy
```

## 使用方法

### 0. 环境激活和快速测试

系统需要使用 `kimi-audio` conda 环境，脚本会自动激活。

```bash
# 快速测试（使用1个GPU处理part1）
./quick_test.sh
```

### 1. 处理单个数据集

#### Whisper+LLM模式（默认）

```bash
# 基本用法（需要先启动LLM服务）
./auto_start_llm_services.sh
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json

# 指定参数
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --cer_threshold 0.1 \
    --num_gpus 4 \
    --output /path/to/output.json

# 指定语言（中文）
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --language zh \
    --num_gpus 4

# 指定语言（英文）
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --language en \
    --num_gpus 4
```

#### Kimi-Audio模式

如果需要使用Kimi-Audio模式，需要显式指定：

```bash
# 基本用法（不使用Whisper）
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --no-use_whisper

# 指定语言（中文）
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --no-use_whisper \
    --language zh \
    --num_gpus 4

# 指定语言（英文）
./run_single_tts_filter.sh /path/to/audio/dir /path/to/groundtruth.json \
    --no-use_whisper \
    --language en \
    --num_gpus 4
```

参数说明：
- `base_dir`: 音频文件基础目录（包含zero_shot子目录）
- `json_file`: 包含groundtruth文本的JSON文件
- `--cer_threshold`: CER阈值，默认0.1（10%）
- `--num_gpus`: 使用的GPU数量，默认8
- `--language`: 文本语言，可选值：auto（自动检测）、zh（中文）、en（英文），默认en
- `--output`: 输出结果文件路径
- `--use_whisper`: 使用Whisper+LLM模式（默认开启，需要先启动LLM服务）
- `--no-use_whisper`: 禁用Whisper模式，使用Kimi-Audio模式
- `--whisper_model`: Whisper模型大小，可选：tiny, base, small, medium, large, large-v2, large-v3，默认large-v3

### 2. 批量处理多个数据集

```bash
# 方法1：使用启动脚本（自动激活kimi-audio环境）
./start_filter_all.sh

# 方法2：手动激活环境后运行
conda activate kimi-audio
./run_all_tts_filter.sh

# 指定参数
./start_filter_all.sh \
    --cer_threshold 0.1 \
    --num_gpus 4 \
    --pattern 'voiceprint_20250804_part*_*.json'

# 处理中文数据集（默认使用Whisper+LLM）
# 记得先启动LLM服务：./auto_start_llm_services.sh
./start_filter_all.sh \
    --language zh \
    --num_gpus 8

# 处理英文数据集（默认使用Whisper+LLM）
./start_filter_all.sh \
    --language en \
    --num_gpus 8

# 使用Kimi-Audio模式处理
./start_filter_all.sh \
    --no-use_whisper \
    --language zh \
    --num_gpus 8
```

参数说明：
- `--share_dir`: voiceprint share目录，默认`/root/group-shared/voiceprint/share`
- `--results_dir`: 结果输出目录
- `--cer_threshold`: CER阈值
- `--num_gpus`: 每个任务使用的GPU数量
- `--language`: 文本语言，可选值：auto（自动检测）、zh（中文）、en（英文），默认auto
- `--pattern`: JSON文件名模式

### 3. 直接运行Python脚本

```bash
python tts_filter_by_kimi_asr.py \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --cer_threshold 0.05 \
    --num_gpus 8 \
    --kimi_model_path /root/data/pretrained_models/Kimi-Audio-7B-Instruct
```

## 输出结果

### 1. 主结果文件（JSON格式）

```json
{
  "base_dir": "/path/to/audio/dir",
  "json_path": "/path/to/groundtruth.json",
  "timestamp": "2024-01-01T12:00:00",
  "statistics": {
    "total_files": 1000,
    "processed_files": 990,
    "failed_files": 10,
    "filtered_files": 150,
    "passed_files": 840,
    "cer_stats": {
      "mean": 0.035,
      "median": 0.025,
      "std": 0.045,
      "min": 0.0,
      "max": 0.85
    }
  },
  "filter_results": [
    {
      "audio_path": "/path/to/audio.wav",
      "voiceprint_id": "voiceprint_12345",
      "prompt_id": "prompt_001",
      "groundtruth_text": "原始文本",
      "transcription": "识别文本",
      "cer": 0.08,
      "passed": false,
      "success": true
    },
    ...
  ]
}
```

### 2. 筛选文件列表

自动生成`*_filtered_list.txt`文件，包含所有被筛选掉的音频文件路径。

### 3. 批量处理汇总报告

批量处理完成后生成汇总报告，包含所有数据集的处理统计。

## 性能优化

1. **多GPU并行**：根据可用GPU数量自动分配任务
2. **批处理**：多个prompt并行处理
3. **内存管理**：及时释放不需要的资源

## 常见问题

### 1. CUDA内存不足
- 减少`--num_gpus`参数
- 检查是否有其他进程占用GPU

### 2. 文本标准化失败
- 确保WeTextProcessing正确安装
- 检查文本语言是否正确识别

### 3. 音频文件找不到
- 确保目录结构符合要求
- 检查文件名是否与JSON中的ID匹配

### 4. 多GPU处理错误
- 如果看到 "Attempting to deserialize object on CUDA device X" 错误，说明多进程GPU分配有问题
- 确保使用最新版本的代码（已修复此问题）
- 检查CUDA环境是否正确配置：`nvidia-smi`
- 查看详细日志了解GPU分配情况

## 注意事项

1. 确保音频文件名与JSON中的voiceprint_id一致
2. 音频文件必须是.wav格式
3. 建议在处理大量数据前先用小数据集测试
4. CER阈值可根据实际需求调整，默认5%是较严格的标准

## 联系方式

如有问题或建议，请联系技术支持团队。