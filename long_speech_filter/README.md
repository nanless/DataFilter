# DataFilter 长音频处理系统

一个用于处理长音频文件的完整流程系统，集成了说话人分离、音频分割、质量筛选和多GPU并行处理功能。

## 🎯 系统概述

### 主要功能
- **说话人分离**: 使用 PyAnnote-audio + TEN-VAD 进行精确的说话人聚类
- **音频分割**: 基于说话人信息自动分割音频，包含0.3秒静音填充
- **质量筛选**: 集成 Whisper + DNSMOS + DNSMOSPro + DistilMOS 多维度质量评估
- **多GPU并行**: 支持多GPU并行处理，显著提升处理速度
- **结构化存储**: 按照 `长音频ID/说话人ID/片段ID` 的层次结构存储
- **完整元数据**: 为每个音频片段保存详细的处理信息和质量分数

### 工作流程
```
长音频输入 → VAD检测 → 说话人分离 → 音频分割(+0.3s填充) → 质量评估 → 筛选保存 → 结构化输出
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda环境
conda create -n DataFilter python=3.8
conda activate DataFilter

# 安装基础依赖
pip install -r requirements.txt

# 安装PyTorch (根据您的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装可选的质量评估模块
pip install distillmos  # 可选，用于DistilMOS评分
```

### 2. 模型准备

系统支持本地模型和在线模型两种方式：

#### 本地模型（推荐）
```bash
# 在项目根目录创建模型目录
mkdir -p pyannote

# 下载预训练模型到 pyannote/ 目录
# 包括：speaker-diarization-3.1, segmentation-3.0, wespeaker-voxceleb-resnet34-LM
```

#### 在线模型
需要设置 Hugging Face Token：
```bash
export HF_TOKEN="your_huggingface_token"
```

### 3. 基本使用

#### 使用启动脚本（推荐）
```bash
# 赋予执行权限
chmod +x start_processing.sh

# 多GPU处理（自动使用全部GPU）
./start_processing.sh --input /path/to/input --output /path/to/output

# 指定GPU数量和并发数
./start_processing.sh \
    --input /path/to/input \
    --output /path/to/output \
    --num-gpus 2 \
    --max-concurrent 4

# 单GPU模式
./start_processing.sh \
    --input /path/to/input \
    --output /path/to/output \
    --single-gpu
```

#### 直接运行Python脚本
```bash
# 多GPU并行处理
python run_multi_gpu.py \
    --input /path/to/input \
    --output /path/to/output \
    --num-gpus 4 \
    --max-concurrent 8

# 单GPU处理
python run_processing.py \
    --input /path/to/input \
    --output /path/to/output
```

## 📋 配置参数

### 核心配置
```python
# VAD检测配置
vad_threshold: 0.5          # VAD检测阈值
min_speech_duration: 0.5    # 最短语音时长(秒)
max_speech_duration: 30.0   # 最长语音时长(秒)
padding_duration: 0.3       # 音频填充时长(秒)

# 说话人分离配置
min_speakers: 1             # 最少说话人数
max_speakers: 10            # 最多说话人数
min_segment_duration: 1.0   # 最短片段时长(秒)

# 质量筛选阈值
distil_mos_threshold: 3.0   # DistilMOS阈值
dnsmos_threshold: 3.0       # DNSMOS阈值
dnsmospro_threshold: 3.0    # DNSMOSPro阈值
min_words: 1                # 最少词数

# Whisper配置
model_name: "large-v3"      # Whisper模型
language: null              # 语言(null=自动检测)
device: "cuda"              # 设备

# 多GPU配置
num_gpus: -1                # GPU数量(-1=全部)
max_concurrent_files: 8     # 最大并发文件数
gpu_memory_fraction: 0.9    # GPU显存使用比例
```

### 自定义配置
```python
from config import LongAudioProcessingConfig

# 创建配置
config = LongAudioProcessingConfig()

# 修改路径
config.input_dir = "/your/input/path"
config.output_dir = "/your/output/path"

# 调整VAD参数
config.vad.threshold = 0.6
config.vad.min_speech_duration = 1.0

# 调整质量阈值
config.quality_filter.distil_mos_threshold = 3.5
config.quality_filter.min_words = 2

# 调整Whisper设置
config.whisper.model_name = "large-v3"
config.whisper.language = "zh"  # 强制中文

# 使用配置
from long_audio_processor import LongAudioProcessor
processor = LongAudioProcessor(config)
```

## 📊 输出结构

### 目录结构
```
output_dir/
├── 音频ID1/
│   ├── SPEAKER_00/
│   │   ├── segment_1673612345678_0_001.wav
│   │   ├── segment_1673612345678_0_001.json
│   │   ├── segment_1673612345790_0_002.wav
│   │   └── segment_1673612345790_0_002.json
│   ├── SPEAKER_01/
│   │   ├── segment_1673612346123_1_001.wav
│   │   └── segment_1673612346123_1_001.json
│   └── processing_summary.json
├── 音频ID2/
│   └── ...
└── final_report.json
```

### 元数据格式
每个音频片段的JSON文件包含：
```json
{
  "segment_id": "1673612345678_0_001",
  "audio_id": "音频文件名",
  "speaker_id": "SPEAKER_00",
  "original_metadata": {
    "start_time": 65.40,           // 原始VAD时间
    "end_time": 66.67,             // 原始VAD时间
    "duration": 1.27,              // 原始时长
    "extended_start_time": 65.10,  // 扩展后开始时间
    "extended_end_time": 66.97,    // 扩展后结束时间
    "extended_duration": 1.87,     // 扩展后时长(含0.3s×2填充)
    "padding_duration": 0.3        // 填充时长
  },
  "transcription": {
    "text": "识别的文本内容",
    "language": "zh",
    "word_count": 4,
    "segments": [...]              // 详细分段信息
  },
  "quality_scores": {
    "distilmos": 4.43,             // DistilMOS分数
    "dnsmos_ovrl": 3.22,           // DNSMOS总分
    "dnsmos_sig": 3.57,            // DNSMOS语音质量
    "dnsmos_bak": 3.97,            // DNSMOS背景噪音
    "dnsmos_p808": 3.15            // DNSMOS P.808分数
  },
  "evaluation_passed": true,        // 是否通过质量筛选
  "processing_timestamp": "2024-01-01T12:00:00"  // 处理时间戳
}
```

## 🔧 高级使用

### 批量处理多个目录
```python
from config import LongAudioProcessingConfig
from long_audio_processor import LongAudioProcessor

# 批量处理配置
input_dirs = [
    "/path/to/batch1",
    "/path/to/batch2", 
    "/path/to/batch3"
]

base_output_dir = "/path/to/output"

for i, input_dir in enumerate(input_dirs, 1):
    config = LongAudioProcessingConfig()
    config.input_dir = input_dir
    config.output_dir = f"{base_output_dir}/batch_{i}"
    
    processor = LongAudioProcessor(config)
    stats = processor.process_directory()
    
    print(f"批次{i}完成: {stats['successful_files']}/{stats['total_files']}")
```

### 单文件处理
```python
from config import LongAudioProcessingConfig
from long_audio_processor import LongAudioProcessor

config = LongAudioProcessingConfig()
processor = LongAudioProcessor(config)

# 处理单个文件
result = processor.process_single_audio("/path/to/audio.wav")

print(f"处理结果: {result.success}")
print(f"检测说话人: {result.speaker_count}")
print(f"总片段: {result.total_segments}")
print(f"通过筛选: {result.passed_segments}")
```

### 配置序列化
```python
import json
from config import LongAudioProcessingConfig

# 保存配置
config = LongAudioProcessingConfig()
config.vad.threshold = 0.7
config.quality_filter.distil_mos_threshold = 3.8

with open('my_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)

# 加载配置
with open('my_config.json', 'r') as f:
    config_dict = json.load(f)
    
# 从字典创建配置对象
config = LongAudioProcessingConfig.from_dict(config_dict)
```

## ⚡ 多GPU并行处理

### 系统架构
```
音频文件列表 → 进程池分配 → GPU资源管理 → 独立处理进程
     ↓              ↓              ↓              ↓
  文件队列    →   进程调度    →   GPU分配    →   [GPU0][GPU1][GPU2][GPU3]
     ↓              ↓              ↓              ↓
  结果收集    ←   状态监控    ←   资源释放    ←   处理完成
```

### 性能参考
以100个10分钟音频文件为例：
- **单GPU**: ~8小时
- **2GPU**: ~4小时 (接近线性加速)
- **4GPU**: ~2小时 (接近线性加速)

### GPU资源管理
- **动态分配**: 基于文件锁的GPU资源分配机制
- **负载均衡**: 确保所有GPU得到充分利用
- **状态监控**: 实时监控处理进度和GPU使用状态
- **自动恢复**: 处理失败时自动重试和资源释放

## 🐛 故障排除

### 常见问题

**Q: JSON文件截断或损坏？**
A: 系统已修复质量分数中的NaN/Inf值问题，确保JSON完整性

**Q: 音频质量差，缺少上下文？**
A: 系统自动在VAD边界前后各添加0.3秒静音，保证语音完整性

**Q: GPU利用率不均衡？**
A: 使用多进程真并行，确保每个GPU独立处理：
```bash
# 监控GPU使用
nvidia-smi -l 1

# 检查进程分配
ps aux | grep python
```

**Q: CUDA内存不足？**
A: 调整并发参数：
```bash
./start_processing.sh \
    --input /path/to/input \
    --output /path/to/output \
    --max-concurrent 4  # 减少并发数
```

**Q: 模型加载失败？**
A: 检查模型路径和权限：
```bash
# 检查本地模型
ls -la pyannote/

# 检查HF token（如使用在线模型）
echo $HF_TOKEN

# 重新下载模型
rm -rf pyannote/
# 重新下载模型文件
```

**Q: 处理速度慢？**
A: 优化建议：
1. 使用多GPU并行：`--num-gpus 4`
2. 调整批处理大小：`--max-concurrent 8` 
3. 使用本地模型避免网络下载
4. 确保SSD存储提高I/O速度

### 调试模式
```bash
# 启用详细日志
./start_processing.sh \
    --input /path/to/input \
    --output /path/to/output \
    --log-level DEBUG

# 监控日志
tail -f logs/processing_YYYYMMDD_HHMMSS.log

# 检查系统状态
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
[print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}') for i in range(torch.cuda.device_count())]
"
```

## 📦 系统要求

### 硬件要求
- **CPU**: 8核以上推荐
- **内存**: 32GB以上推荐  
- **GPU**: NVIDIA GPU with CUDA 11.8+，8GB+ VRAM推荐
- **存储**: SSD存储推荐，确保足够空间存储输出

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+推荐)
- **Python**: 3.8+
- **CUDA**: 11.8+
- **Conda**: 最新版本

### 依赖模块
详见 `requirements.txt`：
- PyTorch 2.0+
- torchaudio
- transformers  
- librosa
- soundfile
- pyannote.audio
- whisper
- numpy
- 其他详见requirements.txt

## 🔄 更新日志

### v1.2.0 (当前版本)
- ✅ 修复JSON文件截断问题（NaN/Inf值处理）
- ✅ 添加音频分割0.3秒静音填充
- ✅ 优化多GPU负载均衡
- ✅ 简化代码结构，移除测试文件
- ✅ 添加启动脚本和完整文档

### v1.1.0
- 🔧 修复GPU负载不均衡问题  
- 🔧 修复文件存储重复命名问题
- 🚀 改用真正的多进程并行
- 📊 改进GPU使用统计和监控

### v1.0.0
- 🎉 初始版本发布
- 集成说话人分离、质量筛选、多GPU支持

## 📄 许可证

本项目使用 MIT 许可证。

## 🤝 贡献

欢迎提交问题报告和改进建议！

---

**DataFilter Team** | 高效的长音频处理解决方案 