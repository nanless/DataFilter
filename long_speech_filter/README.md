# 长音频处理器 (Long Audio Processor)

一个基于深度学习的长音频处理系统，专门用于处理长音频文件的完整流程，包含说话人分离、音频分割和质量筛选功能。

## 🎯 功能特点

### 核心功能
- **说话人分离**: 使用 ten-vad + pyannote-audio 进行说话人聚类和分离
- **音频分割**: 基于说话人信息智能分割音频片段
- **质量筛选**: 集成 Whisper + DNSMOS + DNSMOSPro + DistilMOS 多维度质量评估
- **结构化存储**: 按照 `长音频id/说话人id/句子id` 的层次结构存储
- **多GPU并行**: 支持多GPU并行处理，大幅提升处理效率

### 改进特性
- ✅ **严格的GPU资源管理**: 每GPU限制一个进程，避免显存竞争
- ✅ **主动显存管理**: 多层次显存清理，防止内存泄漏
- ✅ **智能重试机制**: 模型加载失败时自动重试，支持CPU后备模式
- ✅ **实时监控**: 详细的进度和显存使用监控
- ✅ **批量处理优化**: 分批处理减少显存压力

## 📋 系统架构

```
长音频文件 (.wav/.mp3/.flac/.m4a)
    ↓
┌─────────────────────────────────────┐
│ 1. 说话人分离 (Speaker Diarization) │
│   • ten-vad 语音活动检测            │
│   • pyannote-audio 说话人聚类       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 音频分割 (Audio Segmentation)    │
│   • 按说话人片段分割                │
│   • 生成独立音频文件                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 质量筛选 (Quality Assessment)    │
│   • Whisper 语音识别               │
│   • DNSMOS/DNSMOSPro 质量评估      │
│   • DistilMOS 感知质量评估         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 结构化存储                       │
│   输出目录/音频ID/说话人ID/片段.wav │
│   + 完整的元数据信息 (.json)        │
└─────────────────────────────────────┘
```

## 🛠️ 安装配置

### 环境要求
- Python 3.8+
- CUDA 11.0+ (GPU加速)
- 至少 8GB GPU显存 (推荐16GB+)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 依赖说明
```txt
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.21.0
gin-config>=0.5.0
pyannote.audio>=2.1.1
soundfile>=0.10.3
librosa>=0.9.2
numpy>=1.21.0
```

### 模型准备
1. **Whisper模型**: 自动下载到 `/root/data/pretrained_models`
2. **Pyannote模型**: 需要从Hugging Face获取访问令牌
3. **MOS评估模型**: 自动下载相关评估模型

## 📁 目录结构

```
long_speech_filter/
├── __init__.py                    # 模块初始化
├── config.py                      # 配置管理
├── long_audio_processor.py        # 单进程主处理器
├── multi_gpu_processor.py         # 多GPU并行处理器
├── speaker_diarization.py         # 说话人分离模块
├── quality_filter.py             # 质量筛选模块
├── run_improved_multi_gpu.py      # 启动脚本 (推荐)
├── requirements.txt               # 依赖列表
└── README.md                      # 本文档
```

## 🚀 快速开始

### 1. 基本使用
```bash
# 使用默认配置处理 (自动跳过已处理文件)
python run_improved_multi_gpu.py

# 自定义输入输出目录
python run_improved_multi_gpu.py \
    --input /path/to/input \
    --output /path/to/output
```

### 2. 显存受限环境
```bash
# 严格限制显存使用
python run_improved_multi_gpu.py \
    --memory-fraction 0.5 \
    --processes-per-gpu 1 \
    --max-concurrent 2
```

### 3. 处理模式选择
```bash
# 默认模式：跳过已处理文件 (推荐)
python run_improved_multi_gpu.py

# 强制重新处理所有文件
python run_improved_multi_gpu.py --force-reprocess

# 处理所有文件但不强制重新处理已存在的结果
python run_improved_multi_gpu.py --no-skip-processed
```

### 4. 测试验证
```bash
# 干运行检查配置
python run_improved_multi_gpu.py --dry-run

# 测试模式处理少量文件
python run_improved_multi_gpu.py --test-mode
```

## ⚙️ 配置参数

### GPU配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-gpus` | -1 | 使用GPU数量 (-1表示全部) |
| `--processes-per-gpu` | 1 | 每GPU最大进程数 |
| `--memory-fraction` | 0.6 | GPU显存使用比例 |
| `--max-concurrent` | 4 | 最大并发文件数 |

### 模型配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--whisper-model` | large-v3 | Whisper模型名称 |
| `--model-cache-dir` | /root/data/pretrained_models | 模型缓存目录 |

### 质量筛选阈值
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min-words` | 1 | 最少词数要求 |
| `--distilmos-threshold` | 3.0 | DistilMOS阈值 |
| `--dnsmos-threshold` | 3.0 | DNSMOS阈值 |
| `--dnsmospro-threshold` | 3.0 | DNSMOSPro阈值 |

### 处理选项
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip-processed` | True | 跳过已处理的文件 (默认启用) |
| `--no-skip-processed` | False | 处理所有文件，不跳过已处理的 |
| `--force-reprocess` | False | 强制重新处理所有文件，即使已存在结果 |

## 🔧 详细处理流程

### 第一步: 说话人分离

#### VAD (语音活动检测)
```python
# 使用 ten-vad 检测语音片段
vad_config = LongAudioVADConfig(
    threshold=0.5,              # 检测阈值
    min_speech_duration=0.5,    # 最短语音时长
    max_speech_duration=30.0,   # 最长语音时长
    min_silence_duration=0.1    # 最短静音时长
)
```

#### 说话人聚类
```python
# 使用 pyannote-audio 进行聚类
diarization_config = SpeakerDiarizationConfig(
    min_speakers=1,         # 最少说话人数
    max_speakers=10,        # 最多说话人数
    min_segment_duration=1.0, # 最短片段时长
    use_local_models=True   # 使用本地模型
)
```

#### 输出结果
- 检测到的说话人数量
- 每个说话人的时间片段
- 语音活动总时长和比例

### 第二步: 音频分割

#### 片段提取
- 基于说话人时间戳分割音频
- 保持原始采样率和音质
- 生成独立的音频片段

#### 元数据记录
```json
{
    "segment_id": "1234567890_001",
    "speaker_id": "SPEAKER_00", 
    "start_time": 10.5,
    "end_time": 15.8,
    "duration": 5.3,
    "original_file": "audio.wav"
}
```

### 第三步: 质量筛选

#### Whisper语音识别
```python
# 配置参数
whisper_config = WhisperConfig(
    model_name="large-v3",      # 模型大小
    language=None,              # 自动检测语言
    device="cuda",              # 使用GPU
    batch_size=16               # 批处理大小
)
```

**检查项目:**
- 是否识别到文字内容
- 词数是否满足最低要求
- 识别置信度评估

#### MOS质量评估
使用三种互补的质量评估方法:

1. **DNSMOS**: 深度噪声抑制质量评估
2. **DNSMOSPro**: 增强版DNSMOS，支持更多场景
3. **DistilMOS**: 基于知识蒸馏的轻量级质量评估

**评估维度:**
- 语音清晰度 (Speech Quality)
- 背景噪声 (Background Noise)
- 整体感知质量 (Overall Quality)

#### 筛选规则
```python
# 质量阈值配置
quality_config = QualityFilterConfig(
    min_words=1,                    # 最少词数
    distil_mos_threshold=3.0,       # DistilMOS >= 3.0
    dnsmos_threshold=3.0,           # DNSMOS >= 3.0  
    dnsmospro_threshold=3.0,        # DNSMOSPro >= 3.0
    use_distil_mos=True,            # 启用DistilMOS
    use_dnsmos=True,                # 启用DNSMOS
    use_dnsmospro=True              # 启用DNSMOSPro
)
```

### 第四步: 结构化存储

#### 目录结构
```
输出目录/
└── 音频ID_001/
    ├── SPEAKER_00/
    │   ├── segment_1234567890_001.wav
    │   ├── segment_1234567890_001.json
    │   ├── segment_1234567890_002.wav
    │   └── segment_1234567890_002.json
    ├── SPEAKER_01/
    │   ├── segment_1234567890_003.wav
    │   └── segment_1234567890_003.json
    └── processing_summary.json
```

#### 元数据信息
每个音频片段对应一个JSON文件，包含:
```json
{
    "segment_id": "1234567890_001",
    "segment_counter": 1,
    "process_id": 12345,
    "timestamp": 1640995200000,
    "saved_path": "/path/to/segment.wav",
    "audio_id": "audio_001",
    "speaker_id": "SPEAKER_00",
    "original_metadata": {
        "start_time": 10.5,
        "end_time": 15.8,
        "duration": 5.3
    },
    "transcription": {
        "text": "这是识别的文字内容",
        "language": "zh",
        "word_count": 8,
        "confidence": 0.95
    },
    "quality_scores": {
        "distilmos": 3.8,
        "dnsmos_ovrl": 3.5,
        "dnsmos_sig": 3.7,
        "dnsmos_bak": 4.1,
        "dnsmospro": 3.6
    },
    "evaluation_passed": true,
    "processing_timestamp": "2023-12-31T23:59:59"
}
```

## 🔍 显存管理优化

### 问题背景
原系统存在严重的显存管理问题:
- 多个进程竞争同一GPU，总显存占用超过单卡容量
- 模型加载后显存未释放，导致显存泄漏
- 缺少显存监控，无法及时发现问题

### 解决方案

#### 1. 严格GPU进程管理
```python
class SimpleGPUManager:
    """改进的GPU资源管理器"""
    
    def acquire_gpu(self, process_id: int) -> Optional[int]:
        """严格限制每GPU进程数"""
        if data['process_count'] < data['max_processes']:
            data['process_count'] += 1
            data['current_processes'].append({
                'process_id': process_id,
                'start_time': time.time()
            })
            return gpu_id
```

#### 2. 多层次显存清理
```python
def cleanup_gpu_memory():
    """基础显存清理"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

def _aggressive_memory_cleanup(self):
    """激进显存清理"""
    for i in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### 3. 实时显存监控
```python
def get_gpu_memory_usage(gpu_id: int) -> float:
    """获取GPU显存使用率"""
    allocated = torch.cuda.memory_allocated(gpu_id)
    total = torch.cuda.get_device_properties(gpu_id).total_memory
    return allocated / total
```

#### 4. 智能重试机制
- CUDA OOM时自动清理显存并重试
- 多次失败后启用CPU后备模式
- 模型初始化失败时的渐进式降级

#### 5. 批量处理优化
- 小批量处理 (5个文件/批)
- 批次内定期清理 (每3个文件)
- 批次间休息，防止显存积累

## 📊 监控和日志

### 实时监控
系统每15秒输出进度信息:
```log
处理进度: 45/100 (45.0%) - 已用时: 12.3分钟, 预计剩余: 15.2分钟
GPU状态: 4/8 个GPU活跃, 总进程数: 4
  GPU 0: 1 进程, 显存: 67.2%
  GPU 1: 1 进程, 显存: 63.8%
  GPU 2: 1 进程, 显存: 71.5%
  GPU 3: 1 进程, 显存: 59.3%
```

### 详细日志
```log
2025-01-31 17:08:22 - 进程 12345 获取到 GPU 0 (进程数: 1/1)
2025-01-31 17:08:23 - GPU 0 初始显存使用率: 15.2%
2025-01-31 17:08:45 - GPU 0 模型加载后显存使用率: 45.8%
2025-01-31 17:09:12 - GPU 0 处理完成后显存使用率: 47.1%
2025-01-31 17:09:13 - GPU 0 清理后显存使用率: 15.5%
2025-01-31 17:09:13 - 进程 12345 释放 GPU 0 (剩余进程数: 0)
```

### 最终统计报告
```log
=== 处理完成 ===
📊 总文件数: 100
✅ 成功处理: 95
❌ 失败文件: 5
📈 成功率: 95.0%
⏱️ 总处理时间: 45.6分钟
⚡ 平均处理时间: 27.4秒/文件

🖥️ GPU使用统计:
   GPU 0: 处理了 24 个文件, 最终显存: 16.2%
   GPU 1: 处理了 25 个文件, 最终显存: 14.8%
   GPU 2: 处理了 23 个文件, 最终显存: 18.1%
   GPU 3: 处理了 23 个文件, 最终显存: 15.9%

📊 显存使用统计:
   GPU 0: 峰值 72.3%, 平均 58.7%
   GPU 1: 峰值 69.8%, 平均 56.2%
   GPU 2: 峰值 75.1%, 平均 61.4%
   GPU 3: 峰值 68.9%, 平均 57.8%
```

## 🔧 故障排除

### 1. CUDA内存不足
**症状**: `CUDA out of memory` 错误
**解决方案**:
```bash
# 降低显存使用比例
python run_improved_multi_gpu.py --memory-fraction 0.4

# 减少并发进程数
python run_improved_multi_gpu.py --processes-per-gpu 1 --max-concurrent 2

# 使用更小的Whisper模型
python run_improved_multi_gpu.py --whisper-model medium
```

### 2. 处理速度过慢
**症状**: GPU利用率低，处理速度慢
**解决方案**:
```bash
# 检查GPU状态
python run_improved_multi_gpu.py --dry-run

# 适当增加并发（确保显存充足）
python run_improved_multi_gpu.py --max-concurrent 6
```

### 3. 进程挂起
**症状**: 处理进度停滞，进程无响应
**解决方案**:
```bash
# 检查GPU锁文件
ls gpu_work/session_*/gpu_*.lock

# 清理异常锁文件
rm -rf gpu_work/

# 重新启动处理
python run_improved_multi_gpu.py
```

### 4. 质量筛选通过率过低
**症状**: 大部分音频片段被筛选掉
**解决方案**:
```bash
# 降低质量阈值
python run_improved_multi_gpu.py \
    --distilmos-threshold 2.5 \
    --dnsmos-threshold 2.5 \
    --dnsmospro-threshold 2.5

# 减少词数要求
python run_improved_multi_gpu.py --min-words 1
```

### 5. 跳过已处理文件相关问题
**症状**: 文件被错误跳过或重复处理
**解决方案**:
```bash
# 查看处理状态
ls output_dir/audio_id/processing_summary.json

# 强制重新处理特定文件
python run_improved_multi_gpu.py --force-reprocess

# 删除部分处理结果重新处理
rm -rf output_dir/audio_id && python run_improved_multi_gpu.py
```

## 📈 性能优化建议

### 硬件配置
- **GPU**: 推荐RTX 3090/4090或V100，至少16GB显存
- **CPU**: 推荐16核心以上，支持高并发I/O
- **内存**: 推荐64GB以上，支持大量音频文件缓存
- **存储**: 推荐NVMe SSD，提高音频读写速度

### 参数调优
| 场景 | memory-fraction | processes-per-gpu | max-concurrent |
|------|----------------|-------------------|----------------|
| 显存充足 (24GB+) | 0.7 | 1 | 8 |
| 显存一般 (16GB) | 0.6 | 1 | 4 |
| 显存受限 (8GB) | 0.5 | 1 | 2 |
| 极限情况 (4GB) | 0.4 | 1 | 1 |

### 批处理策略
- 按音频文件大小分组处理
- 优先处理短音频，避免长音频阻塞
- 合理设置批次大小，平衡效率和稳定性
- 利用跳过已处理文件功能实现断点续传

## 💡 跳过已处理文件功能详解

### 工作原理
系统通过检查输出目录中的 `processing_summary.json` 文件来判断文件是否已被处理：

1. **检查路径**: `output_dir/audio_id/processing_summary.json`
2. **判断标准**: 
   - 文件存在且可读取
   - `success` 字段为 `true`
   - `processing_results.passed_segments` > 0

### 使用场景

#### 场景1: 正常处理（默认）
```bash
# 自动跳过已处理文件，只处理新文件
python run_improved_multi_gpu.py
```

#### 场景2: 中断后恢复
```bash
# 处理过程中断后重新运行，会自动跳过已完成的文件
python run_improved_multi_gpu.py
```

#### 场景3: 强制重新处理
```bash
# 重新处理所有文件，包括已处理的
python run_improved_multi_gpu.py --force-reprocess
```

#### 场景4: 质量要求变更
```bash
# 修改质量阈值后，强制重新处理
python run_improved_multi_gpu.py \
    --force-reprocess \
    --distilmos-threshold 3.5
```

### 优势
- ✅ **断点续传**: 中断后可继续处理未完成的文件
- ✅ **避免重复**: 节省大量计算时间和资源
- ✅ **灵活控制**: 支持多种处理模式
- ✅ **状态透明**: 清晰显示跳过和处理的文件数量

## 🤝 开发者指南

### 模块扩展
可以通过继承基类来扩展功能:
```python
from long_audio_processor import LongAudioProcessor

class CustomAudioProcessor(LongAudioProcessor):
    def custom_preprocessing(self, audio_path):
        # 自定义预处理逻辑
        pass
    
    def custom_postprocessing(self, result):
        # 自定义后处理逻辑
        pass
```

### 新增质量评估器
```python
from quality_filter import LongAudioQualityFilter

class CustomQualityFilter(LongAudioQualityFilter):
    def custom_quality_assessment(self, audio_path):
        # 实现自定义质量评估
        pass
```

### 配置自定义
所有配置都可以通过修改 `config.py` 来定制:
```python
@dataclass
class CustomConfig(LongAudioProcessingConfig):
    # 添加自定义配置项
    custom_param: str = "default_value"
```

## 📄 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - 说话人分离
- [DNSMOS](https://github.com/microsoft/DNS-Challenge) - 语音质量评估
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 支持

如有问题或建议，请:
1. 查看本README的故障排除部分
2. 检查日志文件获取详细错误信息
3. 使用 `--dry-run` 模式验证配置
4. 尝试 `--test-mode` 进行小规模测试

---

**注**: 本系统针对长音频处理进行了深度优化，特别是在显存管理和多GPU并行处理方面。建议在正式使用前先进行小规模测试，确认配置参数适合您的硬件环境。