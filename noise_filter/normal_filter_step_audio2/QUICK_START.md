# Step-Audio-2 人声检测 - 快速开始

## 前置条件

1. **Conda 环境**: 确保已安装并激活 `stepaudio2` conda 环境
2. **模型**: Step-Audio-2-mini 模型已下载（在 `Step-Audio-2-mini` 目录下）
3. **Step-Audio2 源代码**: 位于 `/root/code/github_repos/Step-Audio2`

## 快速测试（单文件）

测试单个音频文件是否包含人声：

```bash
# 方法 1: 使用测试脚本
./test_single_audio.sh /path/to/audio.wav

# 方法 2: 直接使用 Python
conda activate stepaudio2
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
python example_usage.py /path/to/audio.wav
```

## 批量处理

### 方法 1: 使用快速启动脚本（推荐）

```bash
./start.sh
```

然后选择：
- 选项 1: 测试单个音频文件
- 选项 2: 批量处理所有音频

### 方法 2: 直接运行批量处理脚本

```bash
# 编辑 run_detection.sh 中的路径配置
vim run_detection.sh

# 运行
./run_detection.sh
```

### 方法 3: 直接使用 Python

```bash
conda activate stepaudio2
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"

python step_audio2_human_voice_detector.py \
    --original_folder /path/to/audio/folder \
    --target_folder /path/to/output/folder \
    --model_path Step-Audio-2-mini
```

## 配置说明

### 修改 run_detection.sh

编辑 `run_detection.sh` 文件，修改以下变量：

```bash
# 原始音频文件夹
ORIGINAL_FOLDER="/path/to/your/audio/folder"

# 输出文件夹（存放无人声的音频）
TARGET_FOLDER="/path/to/output/folder"

# 模型路径（可以是 HuggingFace 模型 ID 或本地路径）
MODEL_PATH="Step-Audio-2-mini"

# GPU 设备
DEVICES="cuda:0"

# 并行进程数
NUM_PROCESSES=1
```

## 多 GPU 并行处理

如果有多个 GPU，可以并行处理：

```bash
python step_audio2_human_voice_detector.py \
    --original_folder /path/to/audio/folder \
    --target_folder /path/to/output/folder \
    --model_path Step-Audio-2-mini \
    --num_processes 2 \
    --devices cuda:0,cuda:1
```

## 输出说明

- **无人声的音频**: 会被复制到 `TARGET_FOLDER`，保持原始目录结构
- **有人声的音频**: 会被跳过，不会复制
- **处理统计**: 在控制台显示处理进度和统计信息

## 常见问题

### 1. 找不到 stepaudio2 模块

```bash
# 确保激活了正确的 conda 环境
conda activate stepaudio2

# 确保设置了 PYTHONPATH
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
```

### 2. 模型加载失败

- 检查模型路径是否正确
- 如果使用 HuggingFace 模型 ID，确保网络连接正常
- 检查模型是否已下载到本地

### 3. CUDA 错误

- 检查 GPU 是否可用: `nvidia-smi`
- 确保 CUDA 版本兼容
- 如果显存不足，减少 `--num_processes` 参数

## 下一步

- 查看 [README.md](README.md) 了解更多详细信息
- 查看 [config.yaml](config.yaml) 了解配置选项
- 查看 `example_usage.py` 了解如何在自己的代码中使用

