#!/bin/bash

# 使用 stepaudio2 环境在 GPU 上运行 Step-Audio-2 人声检测
# 快速启动脚本

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "Step-Audio-2 人声检测 - 快速启动"
echo -e "======================================${NC}"

# 1. 激活 conda 环境
echo -e "${YELLOW}[1/5] 激活 stepaudio2 conda 环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate stepaudio2
echo -e "${GREEN}✓ 环境激活成功${NC}"

# 2. 设置 CUDA 设备
echo -e "${YELLOW}[2/5] 设置 CUDA 设备为 GPU 0...${NC}"
export CUDA_VISIBLE_DEVICES=0
echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=0${NC}"

# 3. 设置 Python 路径（包含 Step-Audio2 源代码）
echo -e "${YELLOW}[3/5] 设置 Python 路径...${NC}"
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
echo -e "${GREEN}✓ PYTHONPATH 已设置${NC}"

# 4. 进入工作目录
echo -e "${YELLOW}[4/5] 进入工作目录...${NC}"
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"

# 5. 显示配置信息
echo -e "${YELLOW}[5/5] 检查配置...${NC}"
echo "  - 原始音频: /root/group-shared/voiceprint/data/noise/chime4noise/segments_10s"
echo "  - 输出目录: ./test_output"
echo "  - 模型: /root/code/github_repos/Step-Audio2/Step-Audio-2-mini"
echo "  - GPU: 卡 0"
echo ""

# 询问用户选择
echo -e "${GREEN}请选择运行模式:${NC}"
echo "  1) 测试单个音频文件 (快速验证)"
echo "  2) 批量处理所有音频"
echo "  3) 退出"
echo ""
read -p "请输入选项 [1-3]: " choice

case $choice in
    1)
        echo -e "${GREEN}======================================"
        echo "开始测试单个音频文件..."
        echo -e "======================================${NC}"
        ./test_single_audio.sh
        ;;
    2)
        echo -e "${GREEN}======================================"
        echo "开始批量处理..."
        echo -e "======================================${NC}"
        ./run_detection.sh
        ;;
    3)
        echo -e "${YELLOW}退出${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项！${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}======================================"
echo "完成！"
echo -e "======================================${NC}"

