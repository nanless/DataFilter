#!/bin/bash
# 测试 debug 模式是否正常工作

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   测试 Debug 模式${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 激活环境
if [[ "$CONDA_DEFAULT_ENV" != "SpeakerIdentify" ]]; then
  echo -e "${YELLOW}激活 SpeakerIdentify 环境...${NC}"
  source /root/miniforge3/etc/profile.d/conda.sh
  conda activate SpeakerIdentify
fi

# 设置测试参数
PROMPT_ROOT="/root/group-shared/voiceprint/share/voiceclone_child_20250804"
OUTPUT_DIR="$SCRIPT_DIR/test_output"
DEBUG_DIR="$OUTPUT_DIR/debug"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/debug_test_${TIMESTAMP}.json"

# 创建输出目录
mkdir -p "$OUTPUT_DIR" "$DEBUG_DIR"

echo -e "${YELLOW}测试参数：${NC}"
echo -e "  根目录: $PROMPT_ROOT"
echo -e "  输出文件: $OUTPUT_FILE"
echo -e "  调试目录: $DEBUG_DIR"
echo -e "  样本数: 10"
echo ""

# 检查根目录是否存在
if [ ! -d "$PROMPT_ROOT" ]; then
  echo -e "${RED}错误：根目录不存在: $PROMPT_ROOT${NC}"
  echo -e "${YELLOW}请修改脚本中的 PROMPT_ROOT 变量为有效路径${NC}"
  exit 1
fi

echo -e "${BLUE}开始测试...${NC}"
echo ""

# 运行测试
CMD="./run_voiceprint_filter.sh"
CMD="$CMD --prompt_root \"$PROMPT_ROOT\""
CMD="$CMD --output \"$OUTPUT_FILE\""
CMD="$CMD --debug"
CMD="$CMD --debug_samples 10"
CMD="$CMD --debug_dir \"$DEBUG_DIR\""
CMD="$CMD --threshold 0.7"
CMD="$CMD --num_workers 2"
CMD="$CMD --num_gpus 2"
CMD="$CMD --verbose"

echo -e "${CYAN}执行命令：${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

# 执行并捕获结果
if eval "$CMD"; then
  echo ""
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}   测试成功！${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo ""
  
  # 显示结果摘要
  if [ -f "$OUTPUT_FILE" ]; then
    echo -e "${YELLOW}结果摘要：${NC}"
    
    if command -v jq &> /dev/null; then
      echo ""
      echo -e "${CYAN}统计信息：${NC}"
      jq '.statistics' "$OUTPUT_FILE"
      
      echo ""
      echo -e "${CYAN}处理的样本数：${NC}"
      jq '.filter_results | length' "$OUTPUT_FILE"
      
      echo ""
      echo -e "${CYAN}失败的样本：${NC}"
      FAILED_COUNT=$(jq '[.filter_results[] | select(.success == false)] | length' "$OUTPUT_FILE")
      echo "$FAILED_COUNT"
      
      if [ "$FAILED_COUNT" -gt 0 ]; then
        echo ""
        echo -e "${RED}失败样本详情：${NC}"
        jq '.filter_results[] | select(.success == false) | {voiceprint_id, prompt_id, error_message}' "$OUTPUT_FILE"
      fi
    else
      echo "  (安装 jq 以查看详细统计: apt-get install jq)"
      echo "  输出文件: $OUTPUT_FILE"
    fi
    
    echo ""
    echo -e "${CYAN}生成的 VAD 图：${NC}"
    PNG_COUNT=$(find "$DEBUG_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo "  共 $PNG_COUNT 个文件"
    if [ "$PNG_COUNT" -gt 0 ]; then
      echo "  位置: $DEBUG_DIR"
      find "$DEBUG_DIR" -name "*.png" | head -5
      if [ "$PNG_COUNT" -gt 5 ]; then
        echo "  ..."
      fi
    fi
  fi
  
  echo ""
  echo -e "${GREEN}✓ Debug 模式工作正常${NC}"
  echo -e "${GREEN}✓ MagicMock 错误已修复${NC}"
  exit 0
else
  echo ""
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   测试失败${NC}"
  echo -e "${RED}========================================${NC}"
  echo ""
  echo -e "${YELLOW}请检查错误信息并参考 DEBUG_MODE_README.md${NC}"
  exit 1
fi

