#!/bin/bash

# Step-Audio-2 单音频文件测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "Step-Audio-2 单音频文件测试"
echo -e "======================================${NC}"

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}用法: $0 <音频文件路径>${NC}"
    echo -e "${YELLOW}示例: $0 /path/to/audio.wav${NC}"
    exit 1
fi

AUDIO_FILE="$1"

# 检查文件是否存在
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}错误: 文件不存在: $AUDIO_FILE${NC}"
    exit 1
fi

# 激活 conda 环境
echo -e "${YELLOW}[1/3] 激活 stepaudio2 conda 环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate stepaudio2
echo -e "${GREEN}✓ 环境激活成功${NC}"

# 设置 Python 路径
echo -e "${YELLOW}[2/3] 设置 Python 路径...${NC}"
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
echo -e "${GREEN}✓ PYTHONPATH 已设置${NC}"

# 进入工作目录
echo -e "${YELLOW}[3/3] 进入工作目录...${NC}"
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"

echo ""
echo -e "${GREEN}======================================"
echo "开始检测..."
echo -e "======================================${NC}"
echo ""

# 运行检测
python example_usage.py "$AUDIO_FILE" --model_path "/root/code/github_repos/Step-Audio2/Step-Audio-2-mini"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "检测完成！"
    echo -e "======================================${NC}"
    echo -e "${GREEN}结果: 无人声${NC}"
else
    echo ""
    echo -e "${YELLOW}======================================"
    echo "检测完成！"
    echo -e "======================================${NC}"
    echo -e "${YELLOW}结果: 检测到人声${NC}"
fi

