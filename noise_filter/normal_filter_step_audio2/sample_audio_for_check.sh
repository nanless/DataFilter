#!/bin/bash

# 从原JSON和nohuman JSON中随机采样音频文件用于检查
# - 有人声的音频：在原JSON中但不在nohuman JSON中的
# - 无人声的音频：在nohuman JSON中的
# - 各随机取100条，复制到目标目录

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "音频采样脚本 - 用于检查"
echo -e "======================================${NC}"

# 1. 进入工作目录
echo -e "${YELLOW}[1/3] 进入工作目录...${NC}"
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"

# 2. 设置参数
echo -e "${YELLOW}[2/3] 设置参数...${NC}"
ORIGINAL_JSON="/root/data/lists/noise/merged_dataset_20251127/merged_noise.json"
NOHUMAN_JSON="/root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman.json"
OUTPUT_DIR="/root/data/lists/noise/merged_dataset_20251127/audio_samples_for_check"
SAMPLE_SIZE=100

echo "  - 原始JSON: $ORIGINAL_JSON"
echo "  - 无人声JSON: $NOHUMAN_JSON"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 采样数量: $SAMPLE_SIZE (每种类型)"
echo ""

# 检查输入文件是否存在
if [ ! -f "$ORIGINAL_JSON" ]; then
    echo -e "${RED}错误: 原始JSON文件不存在: $ORIGINAL_JSON${NC}"
    exit 1
fi

if [ ! -f "$NOHUMAN_JSON" ]; then
    echo -e "${RED}错误: 无人声JSON文件不存在: $NOHUMAN_JSON${NC}"
    exit 1
fi

# 3. 运行采样脚本
echo -e "${YELLOW}[3/3] 运行采样脚本...${NC}"
echo -e "${GREEN}======================================"
echo "开始采样..."
echo -e "======================================${NC}"
echo ""

python sample_audio_for_check.py \
    --original_json "$ORIGINAL_JSON" \
    --nohuman_json "$NOHUMAN_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size "$SAMPLE_SIZE"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "采样完成！"
    echo -e "======================================${NC}"
    echo -e "${GREEN}结果已保存到: $OUTPUT_DIR${NC}"
    echo ""
    echo "目录结构:"
    echo "  $OUTPUT_DIR/"
    echo "    ├── has_voice/     (有人声的音频样本)"
    echo "    ├── no_voice/      (无人声的音频样本)"
    echo "    └── sample_results.json  (采样结果JSON)"
else
    echo ""
    echo -e "${RED}======================================"
    echo "采样失败，请检查错误信息"
    echo -e "======================================${NC}"
    exit 1
fi

