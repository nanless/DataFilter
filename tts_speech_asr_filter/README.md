# TTS语音筛选与质量控制系统

一套完整的TTS（文本转语音）音频质量筛选和管理系统，提供多维度的质量评估和数据组织能力。

## 目录

- [系统概述](#系统概述)
- [核心功能模块](#核心功能模块)
- [快速开始](#快速开始)
- [安装依赖](#安装依赖)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [输出格式](#输出格式)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

---

## 系统概述

本系统提供端到端的TTS音频质量控制解决方案，包括：

- **ASR质量筛选**：通过语音识别和CER（字符错误率）评估TTS合成质量
- **声纹相似度筛选**：评估TTS音频与原始音频的音色相似度
- **双重筛选**：结合ASR和声纹两个维度的综合筛选
- **按相似度抽样**：生成不同相似度档次的样本用于人工听辨
- **音频重组**：按说话人重新组织音频数据，便于后续处理

### 核心特性

✅ **多ASR支持**：Whisper+LLM（高精度）和Kimi-Audio（高效率）两种模式  
✅ **多GPU并行**：自动检测可用GPU，支持多卡并行处理  
✅ **智能文本标准化**：LLM或规则引擎处理文本格式差异  
✅ **增量处理**：只处理新增样本，节省时间和资源  
✅ **双重质量保障**：ASR准确度+声纹相似度双重验证  
✅ **灵活配置**：丰富的参数支持各种使用场景  

---

## 核心功能模块

### 1. TTS语音质量筛选（ASR+CER）

通过ASR识别TTS音频，计算识别结果与原文的CER，评估合成质量。

**主要脚本**：
- `tts_filter_by_whisper_asr.py` - Whisper+LLM模式主程序
- `tts_filter_by_kimi_asr.py` - Kimi-Audio模式主程序
- `tts_filter_by_sensevoice_asr.py` - SenseVoice Small+NeMo TN模式主程序
- `llm_service.py` - LLM文本标准化服务
- `run_single_tts_filter.sh` - 单数据集处理（支持三种模式）
- `run_single_tts_filter_sensevoice.sh` - SenseVoice模式专用脚本
- `run_all_tts_filter.sh` - 批量处理
- `auto_start_llm_services.sh` - 启动LLM服务
- `stop_multi_llm_services.sh` - 停止LLM服务

**三种模式对比**：

| 特性 | Whisper+NeMo TN模式 | Kimi-Audio模式 | SenseVoice+NeMo TN模式 |
|------|----------------|----------------|----------------------|
| 识别准确度 | 高 | 中等 | 高（中文优化） |
| 文本标准化 | NeMo TN（英文） | 规则引擎 | NeMo TN（英文）/ 简单标准化（中文） |
| 部署复杂度 | 简单 | 简单 | 简单 |
| 资源占用 | 中等 | 较低 | 中等 |
| 模型大小 | 大（large-v3约10GB） | 大（7B约14GB） | 小（Small约1GB） |
| 处理速度 | 较慢 | 中等 | 快 |
| 适用场景 | 高精度要求、混合语言 | 标准中文、大规模批量 | 中文为主、快速处理 |

### 2. 双重筛选（ASR+声纹）

结合ASR筛选和声纹筛选结果，进行双重质量验证。

**主要脚本**：
- `merge_filter_results.py` - 合并ASR和声纹筛选结果
- `run_merge_filter.sh` - 执行双重筛选

**筛选条件**：
- CER ≤ CER阈值（默认0.2）
- max(similarity_vad, similarity_original) ≥ 相似度阈值（默认0.7）

### 3. 按声纹相似度分档抽样

从筛选结果中按相似度分档抽样，用于人工听辨评估。

**主要脚本**：
- `sample_by_similarity.py` - 按相似度分档抽样
- `run_sample_by_similarity.sh` - 执行抽样

**相似度档次**：

| 档次 | 相似度范围 | 预期质量 |
|------|-----------|---------|
| 0.0-0.5 | 极低 | 音色差异很大 |
| 0.5-0.6 | 低 | 音色明显不同 |
| 0.6-0.7 | 中等偏低 | 音色有差异 |
| 0.7-0.8 | 中等 | 音色较相似 |
| 0.8-0.9 | 高 | 音色很相似 |
| 0.9-1.0 | 极高 | 音色几乎一致 |

### 4. 音频数据重组

将筛选后的音频按说话人重新组织，便于后续处理。

**主要脚本**：
- `reorganize_filtered_audio.py` - 音频重组主程序
- `run_reorganize_audio.sh` - 执行重组

**功能**：
- 读取utt2spk映射关系
- 自动判断数据集类型
- 按说话人组织目录结构
- 多进程并行复制

---

## 快速开始

### 场景1：TTS音频质量筛选（Whisper+LLM模式）

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 1. 启动LLM服务
./auto_start_llm_services.sh

# 2. 处理单个数据集
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --language en \
    --num_gpus 4 \
    --cer_threshold 0.1

# 3. 批量处理多个数据集
./start_filter_all.sh \
    --language en \
    --num_gpus 8 \
    --cer_threshold 0.1
```

### 场景1b：TTS音频质量筛选（SenseVoice Small+NeMo TN模式）

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 1. 处理单个数据集（使用专用脚本）
./run_single_tts_filter_sensevoice.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --language zh \
    --num_gpus 4 \
    --cer_threshold 0.05

# 2. 或使用通用脚本指定模式
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --use_sensevoice \
    --language zh \
    --num_gpus 4 \
    --cer_threshold 0.05
```

### 场景2：双重筛选（ASR+声纹）

```bash
# 合并ASR和声纹筛选结果，进行双重筛选
./run_merge_filter.sh \
    --cer_threshold 0.2 \
    --similarity_threshold 0.7
```

### 场景3：按相似度抽样（人工听辨）

```bash
# 每个档次抽取20个样本
./run_sample_by_similarity.sh --samples_per_bin 20
```

### 场景4：音频数据重组

```bash
# 按说话人重组音频
./run_reorganize_audio.sh \
    --target_dir /path/to/output \
    --num_workers 16
```

---

## 安装依赖

### 硬件要求

- **GPU**：至少1张NVIDIA GPU（推荐8张）
- **显存**：每张GPU至少16GB（运行Whisper large-v3 + LLM）
- **磁盘空间**：根据数据量确定，建议预留足够空间

### 软件环境

#### 1. Conda环境

```bash
conda create -n kimi-audio python=3.10
conda activate kimi-audio
```

#### 2. Python依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jiwer tqdm soundfile numpy requests fastapi uvicorn openai-whisper pydantic
```

#### 3. Whisper模式额外依赖

```bash
# 安装Ollama（用于运行LLM）
curl -fsSL https://ollama.com/install.sh | sh
pip install ollama
```

#### 4. Kimi模式额外依赖

```bash
# 安装NeMo文本标准化（英文TN）
pip install nemo_text_processing
```

#### 5. SenseVoice模式额外依赖

```bash
# 安装funasr（SenseVoice模型）
pip install funasr

# 安装NeMo文本标准化（英文TN）
pip install nemo_text_processing
```

### 模型准备

- **Whisper模型**：会自动下载到配置目录
- **LLM模型**：默认使用`qwen3:32b`，启动服务时自动拉取
- **Kimi-Audio模型**：需手动下载到指定路径
- **SenseVoice模型**：会自动从HuggingFace下载（`iic/SenseVoiceSmall`）

---

## 详细使用指南

### 1. TTS语音质量筛选

#### 数据格式要求

**目录结构**：
```
/path/to/dataset/
└── zero_shot/
    ├── prompt_id_1/
    │   ├── voiceprint_a.wav
    │   ├── voiceprint_b.wav
    │   └── ...
    └── prompt_id_2/
        └── ...
```

**JSON文件格式**：
```json
{
  "prompt_id_1": [
    "voiceprint_a\t这是第一段文本内容",
    "voiceprint_b\t这是第二段文本内容"
  ],
  "prompt_id_2": [
    "voiceprint_c\t这是第三段文本内容"
  ]
}
```

#### Whisper+LLM模式

**启动LLM服务**：
```bash
./auto_start_llm_services.sh

# 检查服务状态
curl http://localhost:8000/health
curl http://localhost:8001/health
```

**处理单个数据集**：
```bash
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --use_whisper \
    --whisper_model large-v3 \
    --language en \
    --num_gpus 4 \
    --cer_threshold 0.1 \
    --output /path/to/output.json
```

**参数说明**：
- `--language`：文本语言（auto/zh/en），默认en
- `--cer_threshold`：CER阈值，默认0.2（20%）
- `--num_gpus`：使用的GPU数量，默认8
- `--whisper_model`：模型大小（tiny/base/small/medium/large/large-v3），默认large-v3
- `--skip_existing`：增量处理模式（默认开启）
- `--force`：强制重新处理所有音频

**批量处理**：
```bash
./start_filter_all.sh \
    --language en \
    --num_gpus 8 \
    --cer_threshold 0.1 \
    --pattern 'voiceprint_*_part*_*.json'
```

**停止LLM服务**：
```bash
./stop_multi_llm_services.sh
```

#### Kimi-Audio模式

无需启动LLM服务，直接运行：

```bash
# 单数据集
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --no-use_whisper \
    --language zh \
    --num_gpus 4 \
    --cer_threshold 0.05

# 批量处理
./start_filter_all.sh \
    --no-use_whisper \
    --language zh \
    --num_gpus 8
```

#### SenseVoice Small+NeMo TN模式

无需启动LLM服务，直接运行：

```bash
# 单数据集（使用专用脚本）
./run_single_tts_filter_sensevoice.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --language zh \
    --num_gpus 4 \
    --cer_threshold 0.05 \
    --sensevoice_model_dir iic/SenseVoiceSmall

# 或使用通用脚本
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --use_sensevoice \
    --language zh \
    --num_gpus 4 \
    --cer_threshold 0.05
```

**参数说明**：
- `--use_sensevoice`：使用SenseVoice模式
- `--sensevoice_model_dir`：SenseVoice模型路径或ID（默认：`iic/SenseVoiceSmall`）
- `--cer_threshold`：CER阈值，默认0.05（5%）
- 其他参数与Whisper/Kimi模式相同

### 2. 双重筛选

```bash
# 使用默认配置
./run_merge_filter.sh

# 自定义参数
./run_merge_filter.sh \
    --asr_result /path/to/asr_results.json \
    --voiceprint_result /path/to/voiceprint_results.json \
    --output_dir /path/to/output \
    --cer_threshold 0.2 \
    --similarity_threshold 0.7

# 只生成报告，不复制音频
./run_merge_filter.sh --no_copy_audio

# 增加并行进程数
./run_merge_filter.sh --num_workers 32
```

**筛选逻辑**：
```
CER ≤ cer_threshold  AND  max(similarity_vad, similarity_original) ≥ similarity_threshold
```

### 3. 按相似度抽样

```bash
# 使用默认配置（每档10个样本）
./run_sample_by_similarity.sh

# 自定义样本数量
./run_sample_by_similarity.sh --samples_per_bin 20

# 自定义输入输出路径
./run_sample_by_similarity.sh \
    --result_json /path/to/voiceprint_results.json \
    --output_dir /path/to/samples
```

**输出目录结构**：
```
similarity_samples/
├── similarity_0.50-0.60/
│   ├── sample_001/
│   │   ├── source.wav       # 原始音频
│   │   ├── tts.wav          # TTS复刻音频
│   │   ├── info.txt         # 详细信息
│   │   └── README.txt       # 简要说明
│   └── ...
├── similarity_0.70-0.80/
└── samples_summary.txt      # 抽样统计摘要
```

**听辨流程**：
1. 查看统计摘要
2. 从低相似度档次开始听
3. 对比每个样本的source.wav和tts.wav
4. 查看info.txt了解详细信息
5. 记录不同档次的音质特点
6. 确定合适的相似度阈值

### 4. 音频数据重组

```bash
# 使用默认配置
./run_reorganize_audio.sh

# 自定义参数
./run_reorganize_audio.sh \
    --target_dir /path/to/output \
    --global_prefix "custom-label" \
    --num_workers 32

# 详细日志
./run_reorganize_audio.sh --verbose
```

**目标目录结构**：
```
output_dir/
├── {global_prefix}_BAAI_001/
│   ├── {global_prefix}_001_5_M_L_LANZHOU_Android_002.wav
│   └── ...
├── {global_prefix}_King-ASR_King-ASR-612_SPEAKER0008/
│   └── ...
└── {global_prefix}_Ocean_speechocean762_test_0003/
    └── ...
```

**数据集标签映射**：

| Prompt ID前缀 | 数据集全名 | 数据集标签 |
|--------------|-----------|----------|
| `001_`, `002_`, ... | BAAI-ChildMandarin41.25H | `BAAI` |
| `Chinese_English_` | Chinese_English_Scripted_Speech_Corpus_Children | `CESSC` |
| `King-ASR-` | King-ASR-EN-Kid | `King-ASR` |
| `speechocean762_` | speechocean762 | `Ocean` |

---

## 配置说明

### config.json

```json
{
  "global_config": {
    "kimi_model_path": "/root/data/pretrained_models/Kimi-Audio-7B-Instruct",
    "kimi_audio_dir": "/root/code/github_repos/Kimi-Audio",
    "whisper_model_dir": "/root/data/pretrained_models/whisper_modes",
    "cer_threshold": 0.2,
    "num_gpus": 8,
    "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]
  },
  "output_settings": {
    "output_dir": "/root/group-shared/voiceprint/share/tts_filter_results",
    "save_filtered_list": true,
    "save_detailed_results": true
  },
  "processing_settings": {
    "batch_size": 32,
    "max_workers_per_gpu": 1,
    "timeout_seconds": 300
  }
}
```

**配置项说明**：

- `kimi_model_path`：Kimi-Audio模型路径
- `whisper_model_dir`：Whisper模型存储目录
- `cer_threshold`：默认CER阈值
- `num_gpus`：默认使用的GPU数量
- `output_dir`：默认输出目录

**注意**：命令行参数优先级高于配置文件

---

## 输出格式

### ASR筛选结果JSON

```json
{
  "base_dir": "/path/to/audio/dir",
  "json_path": "/path/to/groundtruth.json",
  "timestamp": "2025-01-15T12:00:00",
  "statistics": {
    "total_files": 1000,
    "processed_files": 980,
    "failed_files": 20,
    "filtered_files": 150,
    "passed_files": 830,
    "skipped_files": 50,
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
      "voiceprint_id": "voiceprint_123",
      "prompt_id": "prompt_001",
      "groundtruth_text": "原始文本",
      "transcription": "识别文本",
      "normalized_groundtruth": "标准化后文本1",
      "normalized_transcription": "标准化后文本2",
      "cer": 0.08,
      "passed": false,
      "success": true,
      "error_message": "",
      "language": "en"
    }
  ]
}
```

### 双重筛选结果JSON

```json
{
  "timestamp": "2025-01-15T12:00:00",
  "asr_result_path": "/path/to/asr_results.json",
  "voiceprint_result_path": "/path/to/voiceprint_results.json",
  "cer_threshold": 0.2,
  "similarity_threshold": 0.7,
  "statistics": {
    "total_audios": 10000,
    "passed_count": 7500,
    "filtered_count": 2500,
    "pass_rate": 0.75
  },
  "merged_results": [
    {
      "prompt_id": "prompt_001",
      "voiceprint_id": "voiceprint_123",
      "audio_path": "/path/to/audio.wav",
      "asr": {
        "cer": 0.08,
        "groundtruth_text": "原始文本",
        "transcription": "识别文本"
      },
      "voiceprint": {
        "similarity_vad": 0.7006,
        "similarity_original": 0.7109,
        "similarity": 0.7109,
        "sim_ok": true
      },
      "passed": true,
      "filter_reason": "通过"
    }
  ]
}
```

### 筛除列表TXT

```
/path/to/audio1.wav
/path/to/audio2.wav
...
```

### 统计摘要TXT

```
================================================================================
TTS音频双重筛选结果统计
================================================================================
总音频数:     10000
通过筛选:     7500 (75.00%)
被筛除:       2500 (25.00%)

失败原因分布:
  通过: 7500 (75.00%)
  CER超标 (threshold=0.2): 1200 (12.00%)
  相似度不足 (threshold=0.7): 800 (8.00%)
  ASR处理失败: 300 (3.00%)
  声纹处理失败: 200 (2.00%)
================================================================================
```

---

## 常见问题

### 1. Whisper模式LLM服务不可用

**现象**：
```
错误: 没有可用的LLM服务
✗ LLM服务不可用 (端口 8000)
```

**解决方法**：
1. 启动LLM服务：
   ```bash
   ./auto_start_llm_services.sh
   ```

2. 检查服务状态：
   ```bash
   curl http://localhost:8000/health
   ```

3. 查看日志：
   ```bash
   tail -f logs/llm_service_0.log
   tail -f logs/ollama_0.log
   ```

4. 检查代理设置：
   ```bash
   env | grep -i proxy
   # 如果有代理，清除：
   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
   ```

### 2. CUDA内存不足

**现象**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
1. 减少GPU数量：`--num_gpus 4` → `--num_gpus 2`
2. 使用更小的Whisper模型：`--whisper_model base`
3. 检查GPU占用：`nvidia-smi`
4. 使用Kimi模式：`--no-use_whisper`

### 3. 音频文件找不到

**现象**：
```
错误: 音频文件不存在: /path/to/audio.wav
```

**检查清单**：
1. 确认目录结构包含`zero_shot`子目录
2. 确认文件名与JSON中的voiceprint_id一致
3. 确认文件权限：`ls -l /path/to/audio.wav`

### 4. JSON格式错误

**现象**：
```
JSONDecodeError: Expecting property name enclosed in double quotes
```

**解决方法**：
1. 验证JSON格式：`python3 -m json.tool /path/to/file.json`
2. 检查编码：`file /path/to/file.json`（应为UTF-8）
3. 确认格式要求：每个元素为`"voiceprint_id\t文本内容"`

### 5. 多GPU处理错误

**现象**：
```
Attempting to deserialize object on CUDA device X
```

**解决方法**：
- 确保使用最新版本代码
- 检查CUDA环境：`nvidia-smi`

### 6. 增量处理未生效

**现象**：所有文件都被重新处理

**检查**：
1. 确认使用了增量模式（没有`--force`）
2. 确认输出文件路径正确：`ls -la /path/to/output.json`
3. 检查日志：`grep "跳过已处理" logs/tts_filter_*.log`

---

## 性能优化

### 1. GPU选择建议

| 数据集规模 | 推荐GPU数量 |
|----------|-----------|
| < 1000样本 | 1-2张 |
| 1000-10000样本 | 4-8张 |
| > 10000样本 | 全部可用GPU |

### 2. Whisper模型选择

| 模型 | 显存需求 | 处理速度 | 准确率 | 适用场景 |
|------|---------|---------|--------|---------|
| tiny | ~1GB | 最快 | 较低 | 快速测试 |
| base | ~2GB | 快 | 中等 | 日常处理 |
| small | ~3GB | 较快 | 良好 | 平衡选择 |
| medium | ~5GB | 中等 | 优秀 | 高质量要求 |
| large-v3 | ~10GB | 较慢 | 最佳 | 最高精度 |

### 3. CER阈值建议

**英文**：
- 严格：0.05 (5%)
- 标准：0.10 (10%)
- 宽松：0.15 (15%)

**中文**：
- 严格：0.03 (3%)
- 标准：0.05 (5%)
- 宽松：0.10 (10%)

### 4. 相似度阈值建议

- 宽松：0.65
- 标准：0.70
- 严格：0.75
- 非常严格：0.85

### 5. 批处理策略

**连续处理多个数据集**：
```bash
./start_filter_all.sh --num_gpus 8
```

**并行处理独立任务**：
```bash
# 终端1
./run_single_tts_filter.sh /data1 /text1.json --num_gpus 4 &

# 终端2
./run_single_tts_filter.sh /data2 /text2.json --num_gpus 4 &
```

### 6. 增量处理最佳实践

**数据持续增长场景**：
```bash
# 第一天：处理初始数据
./run_single_tts_filter.sh /data /text.json --output result.json

# 第二天：自动增量处理新增数据
./run_single_tts_filter.sh /data /text.json --output result.json
```

### 7. 系统资源监控

```bash
# GPU监控
watch -n 1 nvidia-smi

# 进程监控
watch -n 1 'ps aux | grep -E "python|llm_service|ollama"'

# 磁盘I/O监控
iostat -x 1
```

---

## 工作流程示例

### 完整质量控制流程

```bash
# 步骤1：启动LLM服务
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter
./auto_start_llm_services.sh

# 步骤2：ASR筛选（假设已有声纹筛选结果）
./run_single_tts_filter.sh \
    /path/to/audio/dir \
    /path/to/groundtruth.json \
    --language en \
    --num_gpus 8 \
    --cer_threshold 0.1 \
    --output /path/to/asr_results.json

# 步骤3：双重筛选（合并ASR和声纹结果）
./run_merge_filter.sh \
    --asr_result /path/to/asr_results.json \
    --voiceprint_result /path/to/voiceprint_results.json \
    --cer_threshold 0.2 \
    --similarity_threshold 0.7 \
    --output_dir /path/to/filtered_speech

# 步骤4：按相似度抽样（人工听辨评估）
./run_sample_by_similarity.sh \
    --result_json /path/to/filtered_speech/merged_filter_results.json \
    --samples_per_bin 20 \
    --output_dir /path/to/samples

# 步骤5：音频数据重组（按说话人组织）
./run_reorganize_audio.sh \
    --source_dir /path/to/filtered_speech \
    --target_dir /path/to/organized_audio \
    --num_workers 16

# 步骤6：停止LLM服务
./stop_multi_llm_services.sh
```

---

## 目录结构

```
tts_speech_asr_filter/
├── README.md                          # 本文档
├── config.json                        # 配置文件
│
├── tts_filter_by_whisper_asr.py      # Whisper+LLM主程序
├── tts_filter_by_kimi_asr.py         # Kimi-Audio主程序
├── llm_service.py                    # LLM文本标准化服务
├── merge_filter_results.py           # 双重筛选主程序
├── sample_by_similarity.py           # 按相似度抽样主程序
├── reorganize_filtered_audio.py      # 音频重组主程序
│
├── run_single_tts_filter.sh          # 单数据集ASR筛选
├── run_all_tts_filter.sh             # 批量ASR筛选
├── start_filter_all.sh               # 快捷启动脚本
├── auto_start_llm_services.sh        # 启动LLM服务
├── stop_multi_llm_services.sh        # 停止LLM服务
├── run_merge_filter.sh               # 双重筛选
├── run_sample_by_similarity.sh       # 按相似度抽样
├── run_reorganize_audio.sh           # 音频重组
├── process_voiceclone_20250804.sh    # 特定项目：处理voiceclone_20250804
├── combine_and_filter.sh             # 特定项目：合并JSON并筛选
│
├── results/                          # 结果输出目录
│   ├── tts_filter_results_*.json    # ASR筛选结果
│   ├── merged_filter_results.json   # 双重筛选结果
│   ├── *_filtered_list.txt          # 筛除列表
│   └── filter_summary.txt           # 统计摘要
│
└── logs/                            # 日志目录
    ├── tts_filter_*.log             # 处理日志
    ├── llm_service_*.log            # LLM服务日志
    └── ollama_*.log                 # Ollama服务日志
```

---

## LLM服务详解

### 架构说明

系统采用多实例LLM服务架构：
- 每张GPU运行独立的Ollama实例（端口11434 + GPU_ID）
- 每张GPU运行独立的LLM HTTP服务（端口8000 + GPU_ID）
- 各GPU独立处理，互不干扰

### 启动服务

```bash
# 使用默认模型（qwen3:32b）
./auto_start_llm_services.sh

# 使用自定义模型
./auto_start_llm_services.sh --model-name qwen2.5:14b --model-type qwen2.5
```

### 服务端点

**健康检查**：
```bash
curl http://localhost:8000/health
```

**模型信息**：
```bash
curl http://localhost:8000/model_info
```

**文本标准化**：
```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world!", "text2": "hello world"}'
```

### 文本标准化规则

**英语**：
- 转小写：`"Hello"` → `"hello"`
- 去标点：`"Hello!"` → `"hello"`
- 展开缩写：`"don't"` → `"do not"`
- 数字转文字：`"3"` → `"three"`
- 字母拼读统一：`"abc"` 和 `"a b c"` → `"a b c"`

**中文**：
- 去标点：`"你好！"` → `"你好"`
- 去空格：`"你 好"` → `"你好"`
- 繁转简：`"這個"` → `"这个"`
- 数字转换：`"3个"` → `"三个"`
- 保持同音词原样

---

## 特定项目脚本

以下脚本针对特定项目路径设计，用于处理特定数据集：

### process_voiceclone_20250804.sh

处理 `voiceclone_child_20250804` 目录下的所有数据。

**功能**：
- 检查指定目录下的所有JSON文件
- 对每个JSON对应的音频目录执行ASR识别和CER计算
- 自动选择Whisper+LLM模式（如果LLM不可用则回退为Kimi模式）

**使用**：
```bash
# 基本用法
./process_voiceclone_20250804.sh

# 指定参数
./process_voiceclone_20250804.sh \
    --cer_threshold 0.1 \
    --num_gpus 8 \
    --language auto \
    --process_parts "1,2,5"

# 合并所有JSON后统一处理
./process_voiceclone_20250804.sh --merge

# 分别处理每个part
./process_voiceclone_20250804.sh --no-merge
```

**主要选项**：
- `--cer_threshold <float>`：CER阈值（默认0.1）
- `--num_gpus <int>`：使用的GPU数量
- `--language auto|zh|en`：语言（默认auto）
- `--process_parts <parts>`：指定处理哪些part，如"1,2,5"或"all"（默认all）
- `--merge`：合并所有JSON后统一处理（默认）
- `--no-merge`：分别处理每个part
- `--test_mode`：测试模式（只处理少量数据）
- `--force`：强制重新处理已有结果

### combine_and_filter.sh

合并指定目录下的JSON文件并对zero_shot音频执行筛选。

**功能**：
1. 合并`/root/group-shared/voiceprint/share/voiceclone_child_20251022`下所有JSON为一个总JSON
2. 对指定zero_shot目录下的音频执行筛选
3. 自动选择Whisper+LLM模式（如果LLM不可用则回退为Kimi模式）

**使用**：
```bash
# 基本用法
./combine_and_filter.sh

# 指定参数
./combine_and_filter.sh \
    --cer_threshold 0.1 \
    --num_gpus 8 \
    --language en

# 强制使用Kimi模式
./combine_and_filter.sh --no-use_whisper
```

**主要选项**：
- `--cer_threshold <float>`：CER阈值
- `--num_gpus <int>`：使用的GPU数量
- `--language auto|zh|en`：语言
- `--use_whisper` / `--no-use_whisper`：选择ASR模式
- `--output </path/to/output.json>`：指定输出路径

**说明**：
- 默认尝试Whisper+LLM（检测http://localhost:8000/health）
- 失败后自动回退到Kimi模式
- 若明确指定`--no-use_whisper`，则强制使用Kimi模式
- 可传入`run_single_tts_filter.sh`支持的其他参数

---

## 依赖项

- Python 3.6+
- torch, torchvision, torchaudio
- jiwer, tqdm, soundfile, numpy
- requests, fastapi, uvicorn
- openai-whisper, pydantic
- ollama（Whisper模式）
- nemo_text_processing（文本标准化，可选）

---

## 贡献与支持

如有问题或建议，请联系技术支持团队。

**系统版本**：v2.0  
**最后更新**：2024-11-17
