#!/bin/bash

# 使用 Step-Audio-2 批量处理 JSON 文件中的音频列表
# 读取 merged_noise.json，筛选出没有人声的音频，保存到 merged_noise_nohuman.json

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "Step-Audio-2 JSON 批量处理"
echo -e "======================================${NC}"

# 1. 激活 conda 环境
echo -e "${YELLOW}[1/4] 激活 stepaudio2 conda 环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate stepaudio2
echo -e "${GREEN}✓ 环境激活成功${NC}"

# 2. 设置 Python 路径
echo -e "${YELLOW}[2/4] 设置 Python 路径...${NC}"
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"
echo -e "${GREEN}✓ PYTHONPATH 已设置${NC}"

# 3. 进入工作目录
echo -e "${YELLOW}[3/4] 进入工作目录...${NC}"
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"

# 4. 显示配置信息
echo -e "${YELLOW}[4/4] 检查配置...${NC}"
INPUT_JSON="/root/data/lists/noise/merged_dataset_20251127/merged_noise.json"
OUTPUT_JSON="/root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman_stepaudio2.json"
MODEL_PATH="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini"

echo "  - 输入文件: $INPUT_JSON"
echo "  - 输出文件: $OUTPUT_JSON"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 使用所有可用 GPU"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_JSON" ]; then
    echo -e "${RED}错误: 输入文件不存在: $INPUT_JSON${NC}"
    exit 1
fi

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: 未找到 nvidia-smi，无法检测 GPU${NC}"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}检测到 $NUM_GPUS 个 GPU${NC}"
echo ""
echo -e "${GREEN}======================================"
echo "开始处理，使用所有 $NUM_GPUS 个 GPU 并行处理..."
echo -e "======================================${NC}"
echo ""

# 运行处理脚本
python process_json_batch.py \
    --input_json "$INPUT_JSON" \
    --output_json "$OUTPUT_JSON" \
    --model_path "$MODEL_PATH"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "处理完成！"
    echo -e "======================================${NC}"
    echo -e "${GREEN}结果已保存到: $OUTPUT_JSON${NC}"
else
    echo ""
    echo -e "${RED}======================================"
    echo "处理失败，请检查错误信息"
    echo -e "======================================${NC}"
    exit 1
fi

