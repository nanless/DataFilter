# 语音筛选Pipeline - 多GPU并行处理系统

一个基于多AI模型的高性能语音筛选工具，专为大规模音频数据处理而设计。支持多GPU并行处理、实时结果保存、多语言音频处理，能够从大量音频文件中筛选出高质量的语音数据。

## 🌟 项目特色

- **多GPU并行处理**：支持4张GPU同时处理，处理效率提升4倍以上
- **多模型音质评估**：集成DistilMOS、DNSMOS、DNSMOSPro三种音质评估模型
- **实时结果保存**：每条音频的处理结果实时保存在音频文件旁边，避免数据丢失
- **多语言支持**：支持中文、英语、日语等多种语言的音频处理
- **灵活配置系统**：支持YAML配置文件和命令行参数配置
- **专业音频处理**：VAD检测、Whisper语音识别、音质评估完整流程
- **详细日志系统**：每个GPU独立日志，便于调试和监控

## 🏗️ 系统架构

```
语音筛选Pipeline
├── 输入音频文件
│   ├── 音频格式检查
│   └── 文件预处理
├── 多GPU并行处理
│   ├── GPU0: 音频块1
│   ├── GPU1: 音频块2
│   ├── GPU2: 音频块3
│   └── GPU3: 音频块4
├── 三阶段处理流程
│   ├── 1. VAD检测 (TEN VAD)
│   ├── 2. 语音识别 (Whisper)
│   └── 3. 音质评估 (DistilMOS/DNSMOS/DNSMOSPro)
├── 筛选决策
│   ├── 通过：复制音频 + 保存JSON结果
│   └── 未通过：仅保存JSON结果
└── 结果汇总
    ├── 多GPU统计合并
    ├── 详细结果索引
    └── 可视化报告
```

## 📋 核心功能

### 1. VAD检测 (Voice Activity Detection)
- 使用TEN VAD模型检测语音活动
- 可配置阈值和时长过滤
- 支持静音段去除和语音段提取

### 2. 语音识别 (Automatic Speech Recognition)
- 基于OpenAI Whisper模型
- 支持多种语言自动检测
- 可配置最少词数要求

### 3. 音质评估 (Audio Quality Assessment)
- **DistilMOS**：基于知识蒸馏的音质评估
- **DNSMOS**：Microsoft官方音质评估模型
- **DNSMOSPro**：增强版DNSMOS模型
- 支持单独或组合使用

### 4. 多GPU并行处理
- 自动文件分片到各GPU
- 独立进程处理，避免GPU间干扰
- 实时进度监控和结果收集
- 每个GPU独立日志记录

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (可选，用于GPU加速)
- 16GB+ 内存 (推荐)
- 4张NVIDIA GPU (可选，用于多GPU处理)

### 1. 依赖安装

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果安装失败，可手动安装关键依赖
pip install torch torchaudio transformers
pip install openai-whisper librosa soundfile
pip install ten-vad PyYAML distillmos
pip install onnxruntime pandas numpy scipy
```

### 2. 模型下载

```bash
# 下载所有模型（推荐）
python download_models.py --all --cache-dir /root/data/pretrained_models

# 或分别下载
python download_models.py --model large-v3 --cache-dir /root/data/pretrained_models
python download_models.py --dnsmos --cache-dir /root/data/pretrained_models
python download_models.py --dnsmospro --cache-dir /root/data/pretrained_models
```

### 3. 基本使用

```bash
# 单GPU处理
python main_multi_gpu.py /path/to/audio -o /path/to/output

# 多GPU处理
python main_multi_gpu.py /path/to/audio -o /path/to/output --num-gpus 4

# 使用配置文件
python main_multi_gpu.py /path/to/audio -o /path/to/output --config config.yaml
```

## 🔧 配置系统

### 配置文件格式 (config.yaml)

```yaml
# VAD检测配置
vad:
  threshold: 0.5                    # TEN VAD阈值 (0.0-1.0)
  hop_size: 256                     # 跳跃大小（样本数）
  min_speech_duration: 0.5          # 最短语音时长（秒）
  max_speech_duration: 30.0         # 最长语音时长（秒）

# 语音识别配置
asr:
  model_name: "large-v3"            # Whisper模型大小
  language: null                    # 目标语言（null表示自动检测）
  model_cache_dir: "/root/data/pretrained_models"

# 音质评估配置
audio_quality:
  distil_mos_threshold: 3.0         # DistilMOS阈值 (1.0-5.0)
  dnsmos_threshold: 3.0             # DNSMOS阈值 (1.0-5.0)
  dnsmospro_threshold: 3.0          # DNSMOSPro阈值 (1.0-5.0)
  use_distil_mos: true              # 是否使用DistilMOS
  use_dnsmos: true                  # 是否使用DNSMOS
  use_dnsmospro: true               # 是否使用DNSMOSPro

# 处理配置
processing:
  supported_formats: [".wav", ".mp3", ".flac", ".m4a"]
  sample_rate: 16000                # 重采样率

# 语言特定配置
language_configs:
  chinese:
    asr:
      language: "zh"
    audio_quality:
      distil_mos_threshold: 3.0
  japanese:
    asr:
      language: "ja"
    audio_quality:
      distil_mos_threshold: 3.2
  english:
    asr:
      language: "en"
    audio_quality:
      distil_mos_threshold: 3.5
```

### 命令行参数

```bash
# 基本参数
python main_multi_gpu.py input_dir [OPTIONS]

# 多GPU配置
--num-gpus 4                      # 使用的GPU数量

# 配置文件
--config config.yaml              # 配置文件路径
--language-preset japanese        # 语言预设配置
--save-config my_config.yaml      # 保存当前配置

# VAD参数
--vad-threshold 0.5               # TEN VAD阈值
--min-speech-duration 0.5         # 最短语音时长
--max-speech-duration 30.0        # 最长语音时长

# Whisper参数
--whisper-model large-v3          # Whisper模型大小
--language zh                     # 目标语言
--model-cache-dir /path/to/models # 模型缓存目录

# 音质评估参数
--distilmos-threshold 3.0         # DistilMOS阈值
--dnsmos-threshold 3.0            # DNSMOS阈值
--dnsmospro-threshold 3.0         # DNSMOSPro阈值
--disable-distilmos               # 禁用DistilMOS
--disable-dnsmos                  # 禁用DNSMOS
--disable-dnsmospro               # 禁用DNSMOSPro

# 输出控制
--export-transcriptions           # 导出转录文本
--export-quality-report           # 导出音质报告
--generate-html-report            # 生成HTML报告
--detailed-results                # 实时保存详细结果
--quiet                           # 静默模式
```

## 🎯 专用处理脚本

### StarRail音频处理脚本

专为StarRail 3.3多语言音频数据设计的处理脚本：

```bash
# 基本使用
./process_starrail_audio.sh

# 环境检查
./process_starrail_audio.sh --check-only

# 预览命令
./process_starrail_audio.sh --dry-run

# 处理特定语言
./process_starrail_audio.sh --language chinese
./process_starrail_audio.sh --language japanese
./process_starrail_audio.sh --language english

# 自定义GPU数量
./process_starrail_audio.sh --num-gpus 2
```

## 📊 输出文件结构

### 主要输出文件

#### 1. 筛选后的音频文件
```
output_dir/
├── folder1/
│   ├── audio1.wav              # 通过筛选的音频
│   ├── audio1.wav.json         # 对应的详细结果
│   └── audio2.wav.json         # 未通过筛选的音频只有JSON文件
├── folder2/
│   ├── audio3.wav
│   ├── audio3.wav.json
│   └── audio4.wav.json
└── logs/                       # 日志文件目录
    ├── gpu_0_processing.log    # GPU0处理日志
    ├── gpu_1_processing.log    # GPU1处理日志
    ├── gpu_2_processing.log    # GPU2处理日志
    ├── gpu_3_processing.log    # GPU3处理日志
    └── processing.log          # 主日志文件
```

#### 2. 个人音频详细结果 (*.json)
每条音频对应一个JSON文件，与音频文件在同一目录：
```json
{
  "file_path": "/input/audio1.wav",
  "relative_path": "folder/audio1.wav",
  "passed": true,
  "vad_segments": [[0.5, 3.2], [4.1, 7.8]],
  "transcription": {
    "text": "这是一段测试语音",
    "language": "zh",
    "word_count": 6,
    "success": true
  },
  "quality_scores": {
    "scores": {
      "distilmos": 4.2,
      "dnsmos": 4.1,
      "dnsmospro": 4.0,
      "overall": 4.1
    },
    "success": true
  },
  "processing_time": 2.3,
  "gpu_id": 0,
  "timestamp": "2024-01-01 12:00:00"
}
```

#### 3. 多GPU处理统计 (multi_gpu_stats.json)
```json
{
  "total_files": 10000,
  "processed_files": 10000,
  "passed_files": 6500,
  "failed_files": 3500,
  "total_processing_time": 3600.0,
  "pass_rate": 65.0,
  "gpu_stats": {
    "0": {"processed": 2500, "passed": 1625, "failed": 875},
    "1": {"processed": 2500, "passed": 1625, "failed": 875},
    "2": {"processed": 2500, "passed": 1625, "failed": 875},
    "3": {"processed": 2500, "passed": 1625, "failed": 875}
  }
}
```

#### 4. 详细结果索引 (detailed_results_index.json)
```json
{
  "total_json_files": 10000,
  "creation_time": "2024-01-01 12:00:00",
  "gpu_count": 4,
  "description": "每条音频的详细处理结果JSON文件已保存在与音频文件相同的目录中",
  "note": "JSON文件包含VAD、识别和音质评估信息，与对应的音频文件在同一目录",
  "processed_files": ["audio1.wav.json", "audio2.wav.json", ...]
}
```

#### 5. 转录文本汇总 (multi_gpu_transcriptions.json)
```json
[
  {
    "file_path": "audio1.wav",
    "text": "这是一段测试语音",
    "language": "zh",
    "word_count": 6
  }
]
```

#### 6. 音质评估报告 (multi_gpu_quality_report.json)
```json
[
  {
    "file_path": "audio1.wav",
    "passed": true,
    "distilmos": 4.2,
    "dnsmos": 4.1,
    "dnsmospro": 4.0,
    "overall": 4.1
  }
]
```

## 🛠️ 技术实现

### 多GPU并行处理机制

1. **文件分片**：自动将音频文件平均分配到各GPU
2. **进程隔离**：每个GPU运行独立进程，避免CUDA上下文冲突
3. **设备映射**：通过CUDA_VISIBLE_DEVICES实现GPU设备映射
4. **结果收集**：使用ProcessPoolExecutor收集各GPU结果
5. **独立日志**：每个GPU独立的日志记录，便于调试

### 音质评估模型

#### DistilMOS
- 基于知识蒸馏的轻量级音质评估模型
- 评分范围：1.0-5.0
- 适合实时处理

#### DNSMOS
- Microsoft官方音质评估模型
- 包含SIG、BAK、OVRL、P808四个维度
- 基于ONNX Runtime推理

#### DNSMOSPro
- 增强版DNSMOS模型
- 更高的评估准确性
- 基于PyTorch JIT推理

### VAD检测

- 使用TEN VAD模型进行语音活动检测
- 支持可配置的阈值和时长过滤
- 实时处理，低延迟

### 语音识别

- 基于OpenAI Whisper模型
- 支持多种语言自动检测
- 内置错误重试机制
- 多进程安全处理

## 📈 性能优化

### 硬件建议
- **GPU**: 4张NVIDIA RTX 4090或A100
- **内存**: 64GB+ 系统内存
- **存储**: NVMe SSD，推荐RAID 0
- **CPU**: 16核心以上

### 参数调优
```bash
# 高性能配置
python main_multi_gpu.py input -o output \
    --num-gpus 4 \
    --whisper-model large-v3 \
    --distilmos-threshold 3.5 \
    --dnsmos-threshold 3.5 \
    --detailed-results

# 快速处理配置
python main_multi_gpu.py input -o output \
    --num-gpus 4 \
    --whisper-model medium \
    --disable-dnsmos \
    --disable-dnsmospro \
    --vad-threshold 0.4
```

### 性能基准
- **4张RTX 4090**: ~1000个文件/小时
- **4张A100**: ~1500个文件/小时
- **单GPU模式**: ~250个文件/小时
- **内存占用**: ~8GB/GPU

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：降低GPU数量或使用更小的模型
   python main_multi_gpu.py input -o output --num-gpus 2
   ```

2. **模型加载失败**
   ```bash
   # 解决方案：重新下载模型
   python download_models.py --all --cache-dir /root/data/pretrained_models
   ```

3. **JSON序列化错误**
   - 项目已集成数据类型转换，自动处理numpy类型

4. **多进程冲突**
   - 使用进程池隔离，避免CUDA上下文冲突

### 日志分析
```bash
# 实时查看主处理日志
tail -f /path/to/output/logs/processing.log

# 查看特定GPU日志
tail -f /path/to/output/logs/gpu_0_processing.log

# 查看GPU统计
cat /path/to/output/multi_gpu_stats.json

# 检查详细结果索引
cat /path/to/output/detailed_results_index.json
```

## 🧪 测试和验证

### 项目检查
```bash
# 检查环境依赖
python -c "import torch; print(torch.cuda.is_available())"

# 检查模型文件
ls -la /root/data/pretrained_models/

# 检查GPU状态
nvidia-smi
```

### 功能测试
```bash
# 小规模测试
python main_multi_gpu.py test_audio/ -o test_output/ --num-gpus 1

# 性能测试
time python main_multi_gpu.py large_dataset/ -o output/ --num-gpus 4 --quiet
```

## 📚 开发指南

### 项目结构
```
speech_filter/
├── __init__.py                    # 模块初始化
├── main_multi_gpu.py              # 多GPU主程序
├── config.py                      # 配置管理
├── config.yaml                    # 默认配置
├── pipeline.py                    # 单线程处理流程
├── multi_gpu_pipeline.py          # 多GPU处理流程
├── vad_detector.py                # VAD检测模块
├── speech_recognizer.py           # 语音识别模块
├── audio_quality_assessor.py      # 音质评估模块
├── utils.py                       # 工具函数
├── download_models.py             # 模型下载脚本
├── dnsmospro_utils.py             # DNSMOSPro工具函数
├── process_starrail_audio.sh      # StarRail专用处理脚本
├── requirements.txt               # 依赖列表
└── README.md                      # 文档
```

### 扩展开发
1. **添加新的音质评估模型**：在`audio_quality_assessor.py`中扩展
2. **支持新的音频格式**：在`config.yaml`中添加格式支持
3. **自定义VAD模型**：在`vad_detector.py`中实现新的VAD检测器
4. **新的语言支持**：在`config.yaml`中添加语言配置

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 技术支持邮箱：support@speechfilter.com
- 项目主页：https://github.com/yourusername/speech-filter

---

**版本**: 2.0.0  
**最后更新**: 2024年1月  
**维护者**: Speech Filter Team