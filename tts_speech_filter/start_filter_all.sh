#!/bin/bash

# 启动TTS音频筛选 - 处理所有voiceprint数据
# 支持跳过已处理文件的功能：
#   默认行为：跳过已存在结果的文件
#   使用 --force 或 --no-skip_existing 强制重新处理所有文件

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}    TTS音频筛选 - 批量处理${NC}"
echo -e "${CYAN}========================================${NC}"

# 激活kimi-audio环境
echo -e "${YELLOW}激活kimi-audio conda环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

echo -e "${GREEN}环境已就绪${NC}"
echo -e "${YELLOW}Python版本: $(python --version)${NC}"
echo ""

# 显示增量处理提示信息
echo -e "${CYAN}提示: 默认启用增量处理，只处理新增的音频文件${NC}"
echo -e "${CYAN}      使用 --force 强制重新处理所有音频${NC}"
echo -e "${CYAN}      使用 --help 查看所有选项${NC}"
echo ""

# 执行批量处理，传递所有参数包括bypass选项
echo -e "${YELLOW}开始批量处理所有voiceprint数据...${NC}"
./run_all_tts_filter.sh "$@"