# Step-Audio-2 人声检测脚本

本脚本使用 Step-Audio-2 模型实现音频人声检测。

Step-Audio-2 是阶跃星辰开发的音频语言模型，具有强大的音频理解能力，可以直接判断音频中是否包含人声。

## 项目链接

- GitHub: https://github.com/step-ai/Step-Audio2
- 参考示例: `/root/code/github_repos/Step-Audio2/examples.py`

## 功能特性

- ✅ 使用 Step-Audio-2-mini 直接判断音频中是否有人声
- ✅ 支持多进程并行处理（多 GPU）
- ✅ 支持 HuggingFace 模型 ID 或本地模型路径
- ✅ 支持 WAV、FLAC、MP3、OGG、M4A 格式
- ✅ 进度条显示处理状态
- ✅ 详细的统计信息输出

## 模型信息

### Step-Audio-2-mini
- **用途**: 音频理解和指令跟随
- **特性**: 支持音频理解、ASR、S2TT、S2ST、多轮对话等
- **HuggingFace**: 可通过模型 ID `Step-Audio-2-mini` 或本地路径加载

## 安装依赖

```bash
# 激活 stepaudio2 conda 环境
conda activate stepaudio2

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python step_audio2_human_voice_detector.py \
    --original_folder /path/to/audio/folder \
    --target_folder /path/to/output/folder
```

### 完整参数说明

```bash
python step_audio2_human_voice_detector.py \
    --original_folder /path/to/audio/folder \        # 原始音频文件夹（必需）
    --target_folder /path/to/output/folder \         # 输出文件夹（必需）
    --model_path Step-Audio-2-mini \                 # 模型路径或模型 ID
    --num_processes 2 \                              # 并行进程数
    --devices cuda:0,cuda:1 \                        # 使用的 GPU 设备
```

### 多 GPU 并行处理

如果你有多个 GPU，可以使用多进程加速：

```bash
python step_audio2_human_voice_detector.py \
    --original_folder /data/audio \
    --target_folder /data/filtered_audio \
    --num_processes 2 \
    --devices cuda:0,cuda:1
```

### 使用本地模型

如果模型已经下载到本地：

```bash
python step_audio2_human_voice_detector.py \
    --original_folder /data/audio \
    --target_folder /data/filtered_audio \
    --model_path /path/to/Step-Audio-2-mini
```

### 使用快速启动脚本

```bash
# 交互式启动
./start.sh

# 直接运行批量处理
./run_detection.sh

# 测试单个音频文件
./test_single_audio.sh /path/to/audio.wav
```

## 工作流程

1. **扫描文件**: 递归扫描输入文件夹中的所有音频文件
2. **加载模型**: 加载 Step-Audio-2 模型
3. **人声判断**: 使用 Step-Audio-2 直接判断是否有人声
4. **筛选文件**: 将没有人声的音频文件复制到目标文件夹

## 输出结果

脚本会：
- 保持原始的文件夹结构
- 只复制**没有人声**的音频文件到目标文件夹
- 在控制台显示详细的处理信息和统计数据

### 输出示例

```
音频文件: /data/audio/noise/traffic.wav
人声判断: no
✓ 无人声，已复制到: /data/filtered_audio/noise/

音频文件: /data/audio/speech/conversation.wav
人声判断: yes
✗ 检测到人声，跳过

进程 12345 处理完成:
  总处理: 100 个文件
  有人声: 35 个
  无人声: 65 个
  出错: 0 个
```

## 系统要求

### 最低配置
- Python 3.8+
- CUDA 12.0+ （推荐）
- GPU 显存 >= 8GB（推荐 16GB 以上）
- 磁盘空间 >= 10GB（存储模型）

### 推荐配置
- Python 3.10+
- CUDA 12.0+
- GPU: A100 40GB / H100 / V100 32GB
- 多 GPU 并行处理

## 与其他方案对比

| 特性 | Step-Audio-2 | MiMo-Audio | Qwen3-Omni |
|------|-------------|-----------|------------|
| 模型大小 | mini 版本 | 7B Instruct | 30B Captioner + 30B Instruct |
| 推理步骤 | 1步（直接判断） | 1步（直接判断） | 2步（先描述，再判断） |
| 显存需求 | ~8GB | ~16GB | ~48GB |
| 推理速度 | 快 | 快 | 慢 |
| 准确性 | 高（直接音频理解） | 高（直接音频理解） | 高（基于描述判断） |

## 故障排除

### 显存不足

如果遇到显存不足（CUDA OOM）错误：
- 减少 `--num_processes` 参数
- 使用显存更大的 GPU
- 确保没有其他程序占用 GPU

### 模型加载失败

如果模型加载失败：
- 检查模型路径是否正确
- 如果使用 HuggingFace 模型 ID，确保网络连接正常
- 检查 stepaudio2 conda 环境是否正确激活
- 确保 Step-Audio2 源代码在 `/root/code/github_repos/Step-Audio2`

### 处理速度慢

- 使用多 GPU 并行处理：`--num_processes 2 --devices cuda:0,cuda:1`
- 检查 GPU 性能和驱动版本
- 确保使用 GPU 而不是 CPU

### 导入错误

如果遇到 `ModuleNotFoundError: No module named 'stepaudio2'` 错误：

```bash
# 确保激活了正确的 conda 环境
conda activate stepaudio2

# 检查 Step-Audio2 源代码路径
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
```

## 注意事项

1. **模型大小**: Step-Audio-2-mini 模型相对较小，适合快速推理
2. **处理时间**: 处理速度取决于模型大小和 GPU 性能
3. **结果准确性**: 基于模型直接判断，准确性取决于模型能力
4. **文件格式**: 支持常见音频格式，模型会自动处理
5. **模型加载**: 首次运行可能需要下载模型，请确保网络畅通

## 运行脚本示例

创建一个运行脚本 `run_detection.sh`：

```bash
#!/bin/bash

# Step-Audio-2 人声检测脚本

python step_audio2_human_voice_detector.py \
    --original_folder /data/audio_dataset \
    --target_folder /data/filtered_audio_no_voice \
    --model_path Step-Audio-2-mini \
    --num_processes 1 \
    --devices cuda:0
```

## 高级用法

### 使用配置文件

创建 `config.yaml`：

```yaml
# Step-Audio-2 人声检测配置
model:
  model_path: "Step-Audio-2-mini"

processing:
  num_processes: 2
  devices: "cuda:0,cuda:1"
```

然后创建脚本读取配置文件（需要自行实现）。

## 许可证

本脚本遵循项目根目录的许可证。Step-Audio-2 模型使用请遵循相应的许可协议。

## 联系方式

如有问题或建议，请联系：
- Step-Audio-2 团队: 参考 GitHub Issues
- 本项目 Issues: [GitHub Issues](https://github.com/step-ai/Step-Audio2/issues)

