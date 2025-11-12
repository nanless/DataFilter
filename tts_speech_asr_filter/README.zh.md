# TTS 语音质量筛选系统 - 详细中文文档

## 目录

- [系统概述](#系统概述)
- [工作原理](#工作原理)
- [目录结构](#目录结构)
- [前置依赖](#前置依赖)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [核心脚本详解](#核心脚本详解)
- [Python程序详解](#python程序详解)
- [配置文件说明](#配置文件说明)
- [输出格式](#输出格式)
- [增量处理机制](#增量处理机制)
- [常见问题](#常见问题)
- [性能优化建议](#性能优化建议)

---

## 系统概述

本系统是一套基于 ASR（自动语音识别）和文本标准化的 TTS（文本转语音）音频质量筛选流水线。通过计算 ASR 识别文本与 groundtruth 文本之间的字符错误率（CER）来判断音频质量，自动筛除质量不达标的样本。

### 核心功能

1. **双模式 ASR 支持**：
   - **Whisper + LLM 模式**（推荐）：使用 OpenAI Whisper 进行识别，通过 LLM 服务进行智能文本标准化
   - **Kimi-Audio 模式**：使用 Kimi-Audio 模型识别，通过 WeTextProcessing 进行规则型文本标准化

2. **多 GPU 并行处理**：
   - 按样本级别均衡分配任务到多张 GPU
   - 每张 GPU 独立运行 ASR + LLM 服务
   - 自动检测可用 GPU 数量并优化资源分配

3. **增量处理机制**：
   - 默认跳过已处理的音频文件
   - 只处理新增或未完成的样本
   - 自动合并历史结果与新结果

4. **批量处理支持**：
   - 支持单个数据集处理
   - 支持批量遍历处理多个数据集
   - 自动生成汇总报告

---

## 工作原理

### 整体流程图

```
输入数据 (音频 + JSON文本)
    ↓
1. 数据加载与解析
    ↓
2. ASR 语音识别
    ├─ Whisper 模式: whisper 模型识别
    └─ Kimi 模式: Kimi-Audio 模型识别
    ↓
3. 文本标准化
    ├─ Whisper 模式: LLM 服务标准化 (HTTP API)
    └─ Kimi 模式: WeTextProcessing 规则标准化
    ↓
4. CER 计算与判定
    ├─ 计算标准化后文本的 CER
    └─ 与阈值比较，判定通过/筛除
    ↓
5. 结果汇总与输出
    ├─ 生成结果 JSON
    ├─ 生成筛除列表 TXT
    └─ 输出统计信息
```

### 详细处理步骤

1. **数据加载**：
   - 读取 groundtruth JSON 文件，解析出 `prompt_id` 和 `voiceprint_id` 对应的文本
   - 在 `base_dir/zero_shot/<prompt_id>/` 目录下查找对应的 `.wav` 音频文件

2. **多 GPU 任务分配**：
   - 将所有样本展开为 `(audio_path, voiceprint_id, text, prompt_id)` 列表
   - 按样本数量均匀分配到 N 张 GPU（N = 使用的 GPU 数量）
   - 每张 GPU 启动独立的子进程进行处理

3. **语音识别（ASR）**：
   - **Whisper 模式**：加载指定大小的 Whisper 模型（如 `large-v3`），支持语言自动检测或手动指定
   - **Kimi 模式**：加载 Kimi-Audio 模型，对音频进行转录

4. **文本标准化（TN）**：
   - **Whisper 模式**：
     - 通过 HTTP POST 请求调用 LLM 服务的 `/normalize` 接口
     - 每张 GPU 使用独立的端口（8000 + GPU_ID）
     - LLM 根据提示词对识别文本和 groundtruth 进行统一格式标准化
   - **Kimi 模式**：
     - 使用 WeTextProcessing 库进行中英文文本标准化
     - 包括去标点、大小写转换、数字转文字等

5. **CER 计算**：
   - 使用 `jiwer` 库计算标准化后两段文本的字符错误率
   - 公式：`CER = (替换 + 删除 + 插入) / 参考文本长度`

6. **质量判定**：
   - 如果 `CER <= 阈值`，标记为 `passed = True`（通过）
   - 如果 `CER > 阈值`，标记为 `passed = False`（筛除）

7. **结果输出**：
   - 生成包含所有样本详细信息的 JSON 文件
   - 生成被筛除音频文件路径列表的 TXT 文件
   - 输出统计摘要（总数、通过数、筛除数、CER 统计等）

---

## 目录结构

```
tts_speech_asr_filter/
├── README.zh.md                      # 本文档（详细中文说明）
├── README.md                         # 英文简要说明
├── config.json                       # 配置文件
│
├── tts_filter_by_whisper_asr.py     # Whisper+LLM 主程序
├── tts_filter_by_kimi_asr.py        # Kimi-Audio 主程序
├── llm_service.py                   # LLM 文本标准化 HTTP 服务
│
├── run_single_tts_filter.sh         # 单数据集处理脚本
├── run_all_tts_filter.sh            # 批量处理脚本
├── start_filter_all.sh              # 快捷启动脚本
├── combine_and_filter.sh            # JSON 合并与筛选脚本
│
├── auto_start_llm_services.sh       # 启动多实例 LLM 服务
├── stop_multi_llm_services.sh       # 停止 LLM 服务
│
├── test_bypass.sh                   # 增量处理示例脚本
│
├── results/                          # 结果输出目录
│   ├── tts_filter_results_*.json    # 筛选结果文件
│   ├── *_filtered_list.txt          # 筛除文件列表
│   └── batch_summary_*.txt          # 批量处理汇总
│
└── logs/                            # 日志目录
    ├── tts_filter_*.log             # 处理日志
    ├── llm_service_*.log            # LLM 服务日志
    ├── llm_service_*.pid            # LLM 服务进程 ID
    ├── ollama_*.log                 # Ollama 服务日志
    └── ollama_*.pid                 # Ollama 服务进程 ID
```

---

## 前置依赖

### 硬件要求

- **GPU**：至少 1 张 NVIDIA GPU
- **显存**：建议每张 GPU 至少 16GB（运行 Whisper large-v3 + LLM）
- **工具**：`nvidia-smi` 可用

### 软件环境

1. **Conda 环境**：`kimi-audio`
   ```bash
   conda create -n kimi-audio python=3.10
   conda activate kimi-audio
   ```

2. **Python 依赖包**：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install jiwer tqdm soundfile numpy requests fastapi uvicorn openai-whisper pydantic
   ```

3. **Whisper 模式额外依赖**：
   - **Ollama**：用于运行 LLM 模型
     ```bash
     # 安装 Ollama（参考官方文档）
     curl -fsSL https://ollama.com/install.sh | sh
     
     # 安装 Python 客户端
     pip install ollama
     ```
   - **LLM 模型**：默认使用 `qwen3:32b`，脚本会自动拉取

4. **Kimi 模式额外依赖**：
   - **WeTextProcessing**（可选，用于文本标准化）：
     ```bash
     cd /root/code/github_repos
     git clone https://github.com/wenet-e2e/WeTextProcessing.git
     cd WeTextProcessing
     pip install -e .
     ```
   - **Kimi-Audio 模型**：
     - 模型路径：`/root/data/pretrained_models/Kimi-Audio-7B-Instruct`
     - 代码路径：`/root/code/github_repos/Kimi-Audio`

### Whisper 模型准备

Whisper 模型会自动下载到指定目录，也可手动下载：

```bash
# 模型目录（在 config.json 中配置）
mkdir -p /root/data/pretrained_models/whisper_modes

# 手动下载（可选）
# 访问：https://github.com/openai/whisper
```

---

## 数据准备

### 目录结构要求

```
/path/to/dataset/
└── zero_shot/
    ├── prompt_id_1/
    │   ├── voiceprint_a.wav
    │   ├── voiceprint_b.wav
    │   └── ...
    ├── prompt_id_2/
    │   ├── voiceprint_c.wav
    │   └── ...
    └── ...
```

- `base_dir`：数据集根目录（必须包含 `zero_shot` 子目录）
- `prompt_id`：提示词 ID，对应一个子目录
- `voiceprint_id.wav`：音频文件，文件名为 voiceprint ID

### JSON 文件格式

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

- **键**：`prompt_id`，字符串类型
- **值**：数组，每个元素格式为 `"voiceprint_id\t文本内容"`
- **分隔符**：制表符 `\t` 分隔 ID 和文本

---

## 快速开始

### 方式一：Whisper + LLM 模式（推荐）

#### 1. 启动 LLM 服务

```bash
cd /root/code/github_repos/DataFilter/tts_speech_asr_filter

# 启动多实例 LLM 服务（每张 GPU 一个实例）
./auto_start_llm_services.sh

# 检查服务状态
curl http://localhost:8000/health   # GPU 0
curl http://localhost:8001/health   # GPU 1
# ...
```

#### 2. 处理单个数据集

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

#### 3. 批量处理多个数据集

```bash
./start_filter_all.sh \
    --language en \
    --num_gpus 8 \
    --cer_threshold 0.1 \
    --pattern 'voiceprint_*_part*_*.json'
```

### 方式二：Kimi-Audio 模式

无需启动 LLM 服务，直接运行：

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

---

## 核心脚本详解

### 1. run_single_tts_filter.sh - 单数据集处理脚本

**功能**：处理单个数据集的 TTS 音频筛选任务

**用法**：
```bash
./run_single_tts_filter.sh <base_dir> <json_file> [选项]
```

**必需参数**：
- `base_dir`：音频文件基础目录（包含 `zero_shot` 子目录）
- `json_file`：包含 groundtruth 文本的 JSON 文件路径

**主要选项**：
- `--output <path>`：指定输出结果文件路径
- `--cer_threshold <float>`：CER 阈值，默认 `0.2`（20%）
- `--num_gpus <int>`：使用的 GPU 数量，默认 `8`
- `--language <auto|zh|en>`：文本语言，默认 `en`
  - `auto`：自动检测语言
  - `zh`：中文
  - `en`：英文
- `--use_whisper`：启用 Whisper+LLM 模式（默认已开启）
- `--no-use_whisper`：禁用 Whisper，使用 Kimi-Audio 模式
- `--whisper_model <size>`：Whisper 模型大小，默认 `large-v3`
  - 可选：`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- `--test_mode`：测试模式，只处理前 10 个 prompt
- `--verbose`：输出详细日志
- `--skip_existing`：增量处理模式（默认开启）
- `--no-skip_existing` / `--force`：强制重新处理所有音频
- `--debug_mode`：调试模式，强制使用 8 卡并限制样本数
- `--debug_samples <int>`：调试模式下的样本数上限，默认 `1000`

**工作流程**：
1. 检查并激活 `kimi-audio` conda 环境
2. 验证输入路径和文件是否存在
3. 检查 GPU 可用性
4. 如果是 Whisper 模式，检查 LLM 服务是否可用
5. 构建并执行 Python 命令
6. 输出处理结果和统计信息

**示例**：
```bash
# 基本用法
./run_single_tts_filter.sh /data/audio /data/text.json

# 高精度英文筛选
./run_single_tts_filter.sh /data/audio /data/text.json \
    --language en \
    --cer_threshold 0.05 \
    --num_gpus 8

# 中文数据处理
./run_single_tts_filter.sh /data/audio /data/text.json \
    --language zh \
    --num_gpus 4

# 使用 Kimi 模式
./run_single_tts_filter.sh /data/audio /data/text.json \
    --no-use_whisper \
    --language zh
```

---

### 2. run_all_tts_filter.sh - 批量处理脚本

**功能**：遍历指定目录下所有符合模式的 JSON 文件，批量处理多个数据集

**用法**：
```bash
./run_all_tts_filter.sh [选项]
```

**主要选项**：
- `--share_dir <path>`：voiceprint share 目录，默认 `/root/group-shared/voiceprint/share`
- `--results_dir <path>`：结果输出目录，默认 `./results`
- `--cer_threshold <float>`：CER 阈值
- `--num_gpus <int>`：每个任务使用的 GPU 数量
- `--language <auto|zh|en>`：文本语言
- `--use_whisper` / `--no-use_whisper`：选择 ASR 模式
- `--whisper_model <size>`：Whisper 模型大小
- `--pattern <pattern>`：JSON 文件名匹配模式，默认 `voiceprint_*_part*_*.json`
- `--skip_existing`：增量处理（默认开启）
- `--no-skip_existing` / `--force`：强制重新处理

**工作流程**：
1. 在 `share_dir` 目录下查找所有符合 `pattern` 的 JSON 文件
2. 对每个 JSON 文件：
   - 确定对应的音频目录：`share_dir/<json_basename>`
   - 设置输出路径：`results_dir/tts_filter_results_<basename>.json`
   - 调用 `run_single_tts_filter.sh` 进行处理
   - 记录日志到 `logs/tts_filter_<basename>.log`
3. 生成批量处理汇总报告

**输出**：
- 每个数据集的结果 JSON：`results/tts_filter_results_*.json`
- 每个数据集的筛除列表：`results/*_filtered_list.txt`
- 处理日志：`logs/tts_filter_*.log`
- 汇总报告：`results/batch_summary_<timestamp>.txt`

**示例**：
```bash
# 默认配置批量处理
./run_all_tts_filter.sh

# 自定义配置
./run_all_tts_filter.sh \
    --cer_threshold 0.1 \
    --num_gpus 4 \
    --language en

# 处理特定模式的文件
./run_all_tts_filter.sh \
    --pattern 'voiceprint_20250804_*.json' \
    --num_gpus 8

# 强制重新处理所有文件
./run_all_tts_filter.sh --force
```

---

### 3. start_filter_all.sh - 快捷启动脚本

**功能**：自动激活 conda 环境后调用 `run_all_tts_filter.sh`

**用法**：
```bash
./start_filter_all.sh [选项]
```

**说明**：
- 自动激活 `kimi-audio` 环境
- 传递所有参数给 `run_all_tts_filter.sh`
- 适合作为快捷入口使用

**示例**：
```bash
./start_filter_all.sh --language en --num_gpus 8
```

---

### 4. auto_start_llm_services.sh - 启动 LLM 服务

**功能**：为每张 GPU 启动独立的 Ollama 和 LLM HTTP 服务实例

**用法**：
```bash
./auto_start_llm_services.sh [选项]
```

**选项**：
- `--model-name <name>`：LLM 模型名称，默认 `qwen3:32b`
- `--model-type <type>`：模型类型，默认 `qwen3`（影响提示词策略）

**工作原理**：

1. **检测 GPU 数量**：使用 `nvidia-smi` 检测可用 GPU 数量 N

2. **启动 Ollama 服务**：
   - 为每张 GPU 启动独立的 Ollama 实例
   - GPU 0：端口 `11434`（默认端口）
   - GPU i：端口 `11434 + i`
   - 设置 `CUDA_VISIBLE_DEVICES=$i` 绑定 GPU
   - 清除代理环境变量，确保本地连接

3. **拉取模型**：
   - 等待 Ollama 服务启动（15 秒）
   - 为每个实例拉取指定的 LLM 模型

4. **启动 LLM HTTP 服务**：
   - 为每张 GPU 启动 `llm_service.py` HTTP 服务
   - 端口：`8000 + GPU_ID`
   - 连接对应的 Ollama 实例

5. **健康检查**：
   - 等待服务完全启动（30 秒）
   - 轮询检查每个端口的 `/health` 接口

**输出文件**：
- `logs/ollama_<GPU_ID>.log`：Ollama 服务日志
- `logs/ollama_<GPU_ID>.pid`：Ollama 进程 ID
- `logs/llm_service_<GPU_ID>.log`：LLM HTTP 服务日志
- `logs/llm_service_<GPU_ID>.pid`：LLM 服务进程 ID

**示例**：
```bash
# 使用默认模型
./auto_start_llm_services.sh

# 使用自定义模型
./auto_start_llm_services.sh --model-name qwen2.5:14b --model-type qwen2.5

# 检查服务状态
curl http://localhost:8000/health
curl http://localhost:8001/health
```

---

### 5. stop_multi_llm_services.sh - 停止 LLM 服务

**功能**：停止所有 LLM 和 Ollama 服务实例，释放端口

**用法**：
```bash
./stop_multi_llm_services.sh
```

**工作流程**：

1. **停止 LLM 服务**：
   - 读取 `logs/llm_service_*.pid` 文件
   - 向进程发送 SIGTERM 信号，优雅关闭
   - 等待 10 秒，如果进程仍存在，发送 SIGKILL 强制终止
   - 如果没有 PID 文件，搜索运行中的 `llm_service.py` 进程

2. **停止 Ollama 服务**（可选）：
   - 读取 `logs/ollama_*.pid` 文件或搜索 `ollama serve` 进程
   - 交互模式下询问用户是否停止
   - 非交互模式下自动停止

3. **检查端口占用**：
   - LLM 服务端口：`8000-8010`
   - Ollama 服务端口：`11434-11444`
   - 如果端口仍被占用，提示用户或自动释放

**示例**：
```bash
# 停止所有服务
./stop_multi_llm_services.sh

# 非交互模式（自动释放端口）
./stop_multi_llm_services.sh < /dev/null
```

---

### 6. combine_and_filter.sh - JSON 合并与筛选脚本

**功能**：合并多个 JSON 文件为一个，并对指定目录的音频执行筛选

**用法**：
```bash
./combine_and_filter.sh [选项]
```

**工作流程**：

1. **合并 JSON**：
   - 扫描指定目录下所有 `.json` 文件
   - 合并为一个总 JSON（去重）
   - 输出到 `results/combined_*.json`

2. **自动选择 ASR 模式**：
   - 检测 LLM 服务（`http://localhost:8000/health`）
   - 如果可用，使用 Whisper+LLM 模式
   - 如果不可用，自动回退到 Kimi-Audio 模式

3. **执行筛选**：
   - 调用 `run_single_tts_filter.sh` 处理合并后的 JSON

**配置路径**（脚本内修改）：
- `JSON_SRC_DIR`：JSON 源目录
- `AUDIO_ZERO_SHOT_DIR`：音频 zero_shot 目录
- `RESULTS_DIR`：结果输出目录

**示例**：
```bash
# 使用默认配置
./combine_and_filter.sh

# 传递额外参数
./combine_and_filter.sh --cer_threshold 0.1 --num_gpus 8

# 强制使用 Kimi 模式
./combine_and_filter.sh --no-use_whisper
```

---

## Python 程序详解

### 1. tts_filter_by_whisper_asr.py - Whisper+LLM 主程序

**核心类**：

#### WhisperProcessor - Whisper 语音识别处理器

**功能**：封装 Whisper 模型的加载和转录功能

**初始化参数**：
- `model_size`：模型大小（`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`）
- `device`：CUDA 设备（如 `cuda:0`）
- `gpu_id`：GPU ID（用于日志显示）
- `language`：语言设置（`auto`, `zh`, `en`）
- `model_dir`：模型存储目录

**主要方法**：
- `load_model()`：加载 Whisper 模型到指定 GPU
- `transcribe_audio(audio_path)`：对音频进行转录，返回文本

**特点**：
- 支持语言自动检测
- 支持手动指定语言提高准确率
- 详细的日志输出

#### HTTPTextNormalizer - HTTP 文本标准化器

**功能**：通过 HTTP API 调用 LLM 服务进行文本标准化

**初始化参数**：
- `service_url`：LLM 服务地址（如 `http://localhost:8000`）
- `timeout`：请求超时时间（秒）
- `max_retries`：最大重试次数
- `gpu_id`：GPU ID（用于日志）

**主要方法**：
- `check_service()`：检查 LLM 服务是否可用
- `normalize_text_pair(text1, text2, language)`：标准化文本对

**工作原理**：
- 向 `http://localhost:<port>/normalize` 发送 POST 请求
- 请求体：`{"text1": "识别文本", "text2": "groundtruth"}`
- 返回：`{"normalized_text1": "...", "normalized_text2": "...", "success": true}`
- 支持重试和指数退避

#### TTSFilterProcessor - TTS 筛选处理器

**功能**：协调 ASR、TN、CER 计算的完整流程

**主要方法**：

1. `process_single_audio(audio_path, groundtruth_text, voiceprint_id, prompt_id)`：
   - 处理单个音频文件
   - 返回包含 CER、通过状态等信息的字典

2. `process_sample_tasks(sample_tasks, subset_id)`：
   - 处理分配给当前 GPU 的样本子集
   - 支持增量处理（跳过已处理样本）
   - 返回结果列表和统计信息

**输出格式**：
```python
{
    'audio_path': '/path/to/audio.wav',
    'voiceprint_id': 'voiceprint_123',
    'prompt_id': 'prompt_001',
    'groundtruth_text': '原始文本',
    'transcription': 'ASR识别文本',
    'normalized_groundtruth': '标准化后的原始文本',
    'normalized_transcription': '标准化后的识别文本',
    'cer': 0.08,
    'passed': False,  # CER > 阈值
    'success': True,
    'error_message': '',
    'language': 'en'
}
```

**命令行参数**：
```bash
python3 tts_filter_by_whisper_asr.py \
    <base_dir> \
    <json_file> \
    --output <output.json> \
    --cer_threshold 0.1 \
    --num_gpus 4 \
    --whisper_model_size large-v3 \
    --language en \
    --test_mode \
    --verbose \
    --skip_existing \
    --debug_mode \
    --debug_samples 1000
```

---

### 2. llm_service.py - LLM 文本标准化 HTTP 服务

**功能**：基于 FastAPI 的 HTTP 服务，提供文本标准化接口

**核心类**：

#### OllamaLLMService - Ollama LLM 服务类

**初始化参数**：
- `model_name`：Ollama 模型名称（如 `qwen3:32b`）
- `model_type`：模型类型（`qwen2.5` 或 `qwen3`，影响提示词策略）
- `ollama_host`：Ollama 服务地址（如 `localhost:11434`）

**主要方法**：

1. `check_ollama_service()`：检查 Ollama 服务是否运行

2. `load_model()`：
   - 检查模型是否存在
   - 如果不存在，自动拉取模型
   - 测试模型是否可用

3. `normalize_text_pair(text1, text2)`：
   - 构建详细的标准化提示词
   - 调用 LLM 进行文本标准化
   - 解析响应并提取标准化后的文本

**文本标准化规则**：

**英语标准化**：
- 转小写：`"Hello"` → `"hello"`
- 去标点：`"Hello!"` → `"hello"`
- 展开缩写：`"don't"` → `"do not"`, `"it's"` → `"it is"`
- 数字转文字：`"3"` → `"three"`, `"2023"` → `"two thousand twenty three"`
- 字母拼读统一：`"abc"` 和 `"a b c"` → 都变成 `"a b c"`

**中文标准化**：
- 去标点：`"你好！"` → `"你好"`
- 去空格：`"你 好"` → `"你好"`
- 繁转简：`"這個"` → `"这个"`
- 数字转换：`"3个"` → `"三个"`, `"2023年"` → `"二零二三年"`
- **保持同音词原样**：`"在家"` 和 `"再家"` 不纠错

**中英混合**：
- 英文部分按英语规则
- 中文部分按中文规则

**REST API 接口**：

1. **GET /health** - 健康检查
   ```bash
   curl http://localhost:8000/health
   ```
   响应：
   ```json
   {"status": "healthy", "message": "LLM服务运行正常"}
   ```

2. **GET /model_info** - 模型信息
   ```bash
   curl http://localhost:8000/model_info
   ```
   响应：
   ```json
   {
     "model_name": "qwen3:32b",
     "model_type": "qwen3",
     "ollama_host": "localhost:11434",
     "status": "running"
   }
   ```

3. **POST /normalize** - 文本标准化
   ```bash
   curl -X POST http://localhost:8000/normalize \
     -H "Content-Type: application/json" \
     -d '{"text1": "Hello world!", "text2": "hello world"}'
   ```
   响应：
   ```json
   {
     "normalized_text1": "hello world",
     "normalized_text2": "hello world",
     "success": true,
     "error_message": null
   }
   ```

**启动方式**：
```bash
python3 llm_service.py \
    --model_name qwen3:32b \
    --model_type qwen3 \
    --ollama_host localhost:11434 \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
```

---

### 3. tts_filter_by_kimi_asr.py - Kimi-Audio 主程序

**功能**：使用 Kimi-Audio 模型进行语音识别和筛选

**与 Whisper 版本的区别**：
1. ASR 模型：Kimi-Audio 替代 Whisper
2. 文本标准化：WeTextProcessing 规则库替代 LLM
3. 无需启动额外的 HTTP 服务
4. 默认 CER 阈值更低（0.05）

**主要参数**：
```bash
python3 tts_filter_by_kimi_asr.py \
    <base_dir> \
    <json_file> \
    --output <output.json> \
    --cer_threshold 0.05 \
    --num_gpus 4 \
    --kimi_model_path /path/to/Kimi-Audio-7B-Instruct \
    --kimi_audio_dir /path/to/Kimi-Audio \
    --language auto
```

---

## 配置文件说明

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

#### global_config
- `kimi_model_path`：Kimi-Audio 模型路径
- `kimi_audio_dir`：Kimi-Audio 代码目录
- `whisper_model_dir`：Whisper 模型存储目录
- `cer_threshold`：默认 CER 阈值（命令行参数优先）
- `num_gpus`：默认使用的 GPU 数量
- `gpu_ids`：GPU ID 列表

#### output_settings
- `output_dir`：默认输出目录
- `save_filtered_list`：是否保存筛除文件列表
- `save_detailed_results`：是否保存详细结果

#### processing_settings
- `batch_size`：批处理大小（预留）
- `max_workers_per_gpu`：每个 GPU 的工作进程数
- `timeout_seconds`：超时时间

**注意**：
- `tts_filter_by_whisper_asr.py` 会读取 `whisper_model_dir`
- Shell 脚本不直接读取 `config.json`，使用脚本内默认值或命令行参数
- 建议以命令行参数为准，配置文件作为备用

---

## 输出格式

### 结果 JSON

**文件名**：`tts_filter_results_<basename>_<timestamp>.json`

**结构**：
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
      "audio_path": "/path/to/zero_shot/prompt_x/voiceprint_a.wav",
      "voiceprint_id": "voiceprint_a",
      "prompt_id": "prompt_x",
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

**字段说明**：

#### statistics
- `total_files`：总音频文件数（新处理的）
- `processed_files`：成功处理的文件数
- `failed_files`：处理失败的文件数
- `filtered_files`：被筛除的文件数（CER > 阈值）
- `passed_files`：通过筛选的文件数（CER <= 阈值）
- `skipped_files`：跳过的文件数（增量处理模式）
- `cer_stats`：CER 统计信息
  - `mean`：平均值
  - `median`：中位数
  - `std`：标准差
  - `min`：最小值
  - `max`：最大值

#### filter_results
每个元素代表一个音频样本的处理结果：
- `audio_path`：音频文件绝对路径
- `voiceprint_id`：声纹 ID
- `prompt_id`：提示词 ID
- `groundtruth_text`：原始文本
- `transcription`：ASR 识别文本
- `normalized_groundtruth`：标准化后的原始文本
- `normalized_transcription`：标准化后的识别文本
- `cer`：字符错误率
- `passed`：是否通过筛选
- `success`：是否处理成功
- `error_message`：错误信息（如有）
- `language`：语言设置

### 筛除列表 TXT

**文件名**：`tts_filter_results_<basename>_filtered_list.txt`

**格式**：每行一个被筛除的音频文件绝对路径

```
/path/to/zero_shot/prompt_x/voiceprint_a.wav
/path/to/zero_shot/prompt_y/voiceprint_b.wav
...
```

**用途**：
- 可用于批量删除或移动不合格音频
- 可作为后续处理的输入列表

### 批量处理汇总报告

**文件名**：`batch_summary_<timestamp>.txt`

**内容**：
```
批量TTS音频筛选汇总报告
========================
处理时间: 2025-01-15 12:00:00
总文件数: 10
成功处理: 9
跳过文件: 0
处理失败: 1
总耗时: 2小时 30分钟 15秒

处理详情:
----------
voiceprint_part1.json: 总数=1000, 筛选=150 (15.0%)
voiceprint_part2.json: 总数=950, 筛选=120 (12.6%)
...
```

---

## 增量处理机制

### 原理

增量处理通过记录已处理的音频文件路径，实现"只处理新增样本"的功能：

1. **首次运行**：
   - 处理所有样本
   - 生成结果 JSON

2. **再次运行**（增量模式）：
   - 加载已有结果 JSON
   - 提取 `processed_audio_paths` 集合
   - 在主进程中过滤掉已处理样本
   - 只处理新增样本
   - 合并新旧结果并保存

3. **统计合并**：
   - 累加 `total_files`, `processed_files` 等
   - 合并 `cer_values` 列表
   - 重新计算 CER 统计量

### 使用方式

**默认行为**（增量处理）：
```bash
# 第一次运行
./run_single_tts_filter.sh /data/audio /data/text.json --output result.json
# 处理 1000 个样本

# 第二次运行（新增了 200 个样本）
./run_single_tts_filter.sh /data/audio /data/text.json --output result.json
# 只处理新增的 200 个样本，跳过已有的 1000 个
```

**强制重新处理**：
```bash
# 方式一：使用 --force
./run_single_tts_filter.sh /data/audio /data/text.json --output result.json --force

# 方式二：使用 --no-skip_existing
./run_single_tts_filter.sh /data/audio /data/text.json --output result.json --no-skip_existing
```

### 优势

1. **节省时间**：
   - 大数据集场景下，新增少量样本时无需重新处理全部数据
   - 处理中断后可继续，不丢失已有结果

2. **资源利用**：
   - 减少 GPU 计算量
   - 减少存储 I/O

3. **灵活性**：
   - 支持分批次添加数据
   - 支持增量式数据清洗流程

### 注意事项

- 增量处理基于音频文件的**绝对路径**判断
- 如果音频文件路径改变，会被视为新样本
- 如果需要重新评估旧样本（如更改阈值），使用 `--force`

---

## 常见问题

### 1. Whisper 模式 LLM 服务不可用

**现象**：
```
错误: 没有可用的LLM服务
✗ LLM服务不可用 (端口 8000)
```

**解决方法**：
1. 启动 LLM 服务：
   ```bash
   cd /root/code/github_repos/DataFilter/tts_speech_asr_filter
   ./auto_start_llm_services.sh
   ```

2. 检查服务状态：
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8001/health
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

### 2. CUDA 内存不足

**现象**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
1. 减少使用的 GPU 数量：
   ```bash
   --num_gpus 4  # 改为 2 或 1
   ```

2. 使用更小的 Whisper 模型：
   ```bash
   --whisper_model base  # 或 small, medium
   ```

3. 检查 GPU 占用：
   ```bash
   nvidia-smi
   # 如果有其他进程占用，先停止
   ```

4. 使用 Kimi 模式（显存需求更低）：
   ```bash
   --no-use_whisper
   ```

### 3. 音频文件找不到

**现象**：
```
错误: 音频文件不存在: /path/to/audio.wav
```

**检查清单**：
1. 确认目录结构：
   ```bash
   ls -la /path/to/base_dir/zero_shot/
   ls -la /path/to/base_dir/zero_shot/<prompt_id>/
   ```

2. 确认文件名匹配：
   - JSON 中的 `voiceprint_id` 必须与 `.wav` 文件名一致
   - 示例：JSON 中为 `voiceprint_123`，文件名应为 `voiceprint_123.wav`

3. 检查文件权限：
   ```bash
   ls -l /path/to/audio.wav
   ```

### 4. JSON 格式错误

**现象**：
```
JSONDecodeError: Expecting property name enclosed in double quotes
```

**解决方法**：
1. 验证 JSON 格式：
   ```bash
   python3 -m json.tool /path/to/file.json
   ```

2. 确认格式要求：
   ```json
   {
     "prompt_id": [
       "voiceprint_id\t文本内容"
     ]
   }
   ```

3. 检查编码：
   ```bash
   file /path/to/file.json
   # 应为 UTF-8 编码
   ```

### 5. 多 GPU 处理错误

**现象**：
```
Attempting to deserialize object on CUDA device X
```

**解决方法**：
- 确保使用最新版本的代码（已修复此问题）
- 检查 CUDA 环境：
  ```bash
  nvidia-smi
  echo $CUDA_VISIBLE_DEVICES
  ```

### 6. Ollama 模型拉取失败

**现象**：
```
Error: failed to pull model
```

**解决方法**：
1. 检查网络连接
2. 检查磁盘空间：
   ```bash
   df -h
   ```
3. 手动拉取模型：
   ```bash
   ollama pull qwen3:32b
   ```
4. 查看 Ollama 日志：
   ```bash
   tail -f logs/ollama_0.log
   ```

### 7. 增量处理未生效

**现象**：
所有文件都被重新处理，没有跳过已处理样本

**检查**：
1. 确认使用了增量模式：
   ```bash
   # 应该有 --skip_existing 或没有 --force
   ```

2. 确认输出文件路径正确：
   ```bash
   ls -la /path/to/output.json
   ```

3. 检查日志：
   ```bash
   grep "增量处理" logs/tts_filter_*.log
   grep "跳过已处理" logs/tts_filter_*.log
   ```

---

## 性能优化建议

### 1. GPU 选择

**单数据集处理**：
- **小数据集**（< 1000 样本）：使用 1-2 张 GPU
- **中等数据集**（1000-10000 样本）：使用 4-8 张 GPU
- **大数据集**（> 10000 样本）：使用全部可用 GPU

**批量处理**：
- 如果同时有多个任务，考虑每个任务使用较少 GPU
- 例如：8 张 GPU，2 个任务，每个任务 4 张 GPU

### 2. Whisper 模型选择

**模型大小与性能对比**：

| 模型 | 显存需求 | 处理速度 | 准确率 | 适用场景 |
|------|---------|---------|--------|---------|
| tiny | ~1GB | 最快 | 较低 | 快速测试 |
| base | ~2GB | 快 | 中等 | 日常处理 |
| small | ~3GB | 较快 | 良好 | 平衡选择 |
| medium | ~5GB | 中等 | 优秀 | 高质量要求 |
| large-v3 | ~10GB | 较慢 | 最佳 | 最高精度 |

**建议**：
- **开发测试**：使用 `base` 或 `small` 快速迭代
- **生产环境**：使用 `large-v3` 保证质量
- **英文为主**：`large-v3` 表现最佳
- **中文为主**：考虑 `medium` 或 Kimi 模式

### 3. CER 阈值调优

**推荐阈值范围**：

- **英文**：
  - 严格：0.05 (5%)
  - 标准：0.10 (10%)
  - 宽松：0.15 (15%)

- **中文**：
  - 严格：0.03 (3%)
  - 标准：0.05 (5%)
  - 宽松：0.10 (10%)

**调优建议**：
1. 先用较高阈值（如 0.2）处理，查看 CER 分布
2. 根据分布和需求调整阈值
3. 检查被筛除样本，确认是否合理

### 4. 批处理策略

**场景一：连续处理多个数据集**
```bash
# 使用批量脚本
./start_filter_all.sh --num_gpus 8
```

**场景二：并行处理多个独立任务**
```bash
# 终端 1
./run_single_tts_filter.sh /data1 /text1.json --num_gpus 4 &

# 终端 2
./run_single_tts_filter.sh /data2 /text2.json --num_gpus 4 &
```

### 5. 增量处理最佳实践

**场景一：数据持续增长**
```bash
# 第一天：处理初始数据
./run_single_tts_filter.sh /data /text.json --output result.json

# 第二天：处理新增数据（自动增量）
./run_single_tts_filter.sh /data /text.json --output result.json
```

**场景二：分批次处理大数据集**
```bash
# 处理前 10 个 prompt（测试）
./run_single_tts_filter.sh /data /text.json --test_mode --output result.json

# 处理全部（跳过已处理的前 10 个）
./run_single_tts_filter.sh /data /text.json --output result.json
```

### 6. 系统资源监控

**监控脚本**：
```bash
# GPU 监控
watch -n 1 nvidia-smi

# 进程监控
watch -n 1 'ps aux | grep -E "python|llm_service|ollama"'

# 磁盘 I/O 监控
iostat -x 1

# 网络监控（LLM 服务请求）
netstat -an | grep 800[0-9]
```

---

## 总结

本系统提供了完整的 TTS 音频质量筛选解决方案，支持：

- ✅ **双模式 ASR**：Whisper+LLM（高精度）和 Kimi-Audio（高效率）
- ✅ **多 GPU 并行**：自动负载均衡，充分利用硬件资源
- ✅ **智能文本标准化**：LLM 或规则引擎，适应多种场景
- ✅ **增量处理**：节省时间和资源，支持大规模数据
- ✅ **灵活配置**：丰富的命令行参数，满足不同需求
- ✅ **详细日志**：完整的处理记录，便于问题排查

通过本文档，您可以：
1. 快速上手使用系统进行音频筛选
2. 深入理解每个脚本和程序的工作原理
3. 根据实际需求调整配置和优化性能
4. 遇到问题时快速定位和解决

如有其他问题或建议，请联系技术支持团队。
