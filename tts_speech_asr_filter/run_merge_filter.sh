#!/bin/bash

# TTS音频双重筛选（ASR + 声纹）Shell包装脚本
# 根据ASR筛选和声纹筛选结果，对TTS音频进行双重筛选

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 默认配置
# ASR_RESULT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_asr_filter/results/tts_asr_filter_results_combined_voiceclone_child.json"
ASR_RESULT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_asr_filter/results/tts_asr_filter_combined_voiceclone_20250804_20251116_001542.json"
# VP_RESULT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_voiceprint_filter_dualsim/results/tts_voiceprint_filter_results_20251114_120918.json"
VP_RESULT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_prompt_clone_similarity_dualsim/results/tts_prompt_clone_similarity_20251114_120934.json"
OUTPUT_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech"
CER_THRESHOLD_DEFAULT="0.2"
SIM_THRESHOLD_DEFAULT="0.7"
NUM_WORKERS_DEFAULT="16"

ASR_RESULT="$ASR_RESULT_DEFAULT"
VP_RESULT="$VP_RESULT_DEFAULT"
OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
CER_THRESHOLD="$CER_THRESHOLD_DEFAULT"
SIM_THRESHOLD="$SIM_THRESHOLD_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
NO_COPY_AUDIO=false
FLAT_STRUCTURE=false
VERBOSE=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --asr_result PATH         ASR筛选结果JSON文件"
  echo "                            (默认: $ASR_RESULT_DEFAULT)"
  echo "  --voiceprint_result PATH  声纹筛选结果JSON文件"
  echo "                            (默认: $VP_RESULT_DEFAULT)"
  echo "  --output_dir DIR          输出目录"
  echo "                            (默认: $OUTPUT_DIR_DEFAULT)"
  echo "  --cer_threshold FLOAT     CER阈值，CER <= 此值通过 (默认: $CER_THRESHOLD_DEFAULT)"
  echo "  --similarity_threshold FLOAT  声纹相似度阈值，相似度 >= 此值通过"
  echo "                            (默认: $SIM_THRESHOLD_DEFAULT)"
  echo "  --num_workers INT         复制音频的并行工作进程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --no_copy_audio           不复制音频文件，只生成结果报告"
  echo "  --flat_structure          使用扁平目录结构（默认按prompt组织）"
  echo "  --verbose                 详细日志"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "说明:"
  echo "  - 双重筛选逻辑: CER <= cer_threshold AND similarity >= similarity_threshold"
  echo "  - 使用多进程并行复制音频，大幅提升速度"
  echo "  - 输出目录结构:"
  echo "    output_dir/"
  echo "      ├── audio/                    # 通过筛选的音频文件"
  echo "      │   ├── <prompt_id>/"
  echo "      │   │   └── <voiceprint_id>.wav"
  echo "      ├── merged_filter_results.json  # 完整结果JSON"
  echo "      ├── passed_list.txt             # 通过音频列表"
  echo "      ├── filtered_list.txt           # 筛除音频列表"
  echo "      └── filter_summary.txt          # 统计摘要"
  echo ""
  echo "示例:"
  echo "  # 使用默认配置"
  echo "  $0"
  echo ""
  echo "  # 自定义阈值和进程数"
  echo "  $0 --cer_threshold 0.10 --similarity_threshold 0.70 --num_workers 32"
  echo ""
  echo "  # 只生成报告，不复制音频"
  echo "  $0 --no_copy_audio"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr_result)         ASR_RESULT="$2"; shift 2;;
    --voiceprint_result)  VP_RESULT="$2"; shift 2;;
    --output_dir)         OUTPUT_DIR="$2"; shift 2;;
    --cer_threshold)      CER_THRESHOLD="$2"; shift 2;;
    --similarity_threshold) SIM_THRESHOLD="$2"; shift 2;;
    --num_workers)        NUM_WORKERS="$2"; shift 2;;
    --no_copy_audio)      NO_COPY_AUDIO=true; shift;;
    --flat_structure)     FLAT_STRUCTURE=true; shift;;
    --verbose)            VERBOSE=true; shift;;
    -h|--help)            show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# 检查输入文件
if [ ! -f "$ASR_RESULT" ]; then
  echo -e "${RED}错误: ASR结果文件不存在: $ASR_RESULT${NC}"
  exit 1
fi

if [ ! -f "$VP_RESULT" ]; then
  echo -e "${RED}错误: 声纹结果文件不存在: $VP_RESULT${NC}"
  exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TTS音频双重筛选（ASR + 声纹）${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}ASR结果: $ASR_RESULT${NC}"
echo -e "${YELLOW}声纹结果: $VP_RESULT${NC}"
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}CER阈值: $CER_THRESHOLD${NC}"
echo -e "${YELLOW}相似度阈值: $SIM_THRESHOLD${NC}"
echo -e "${YELLOW}工作进程数: $NUM_WORKERS${NC}"
if [ "$NO_COPY_AUDIO" = true ]; then
  echo -e "${YELLOW}音频复制: 禁用${NC}"
else
  echo -e "${YELLOW}音频复制: 启用${NC}"
fi
if [ "$FLAT_STRUCTURE" = true ]; then
  echo -e "${YELLOW}目录结构: 扁平${NC}"
else
  echo -e "${YELLOW}目录结构: 按prompt组织${NC}"
fi
echo ""

# 构建Python命令
CMD="python3 \"$SCRIPT_DIR/merge_filter_results.py\""
CMD="$CMD --asr_result \"$ASR_RESULT\""
CMD="$CMD --voiceprint_result \"$VP_RESULT\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --cer_threshold $CER_THRESHOLD"
CMD="$CMD --similarity_threshold $SIM_THRESHOLD"
CMD="$CMD --num_workers $NUM_WORKERS"

if [ "$NO_COPY_AUDIO" = true ]; then
  CMD="$CMD --no_copy_audio"
fi

if [ "$FLAT_STRUCTURE" = true ]; then
  CMD="$CMD --flat_structure"
fi

if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

echo -e "${CYAN}开始执行...${NC}"
if [ "$VERBOSE" = true ]; then
  echo -e "${YELLOW}命令: $CMD${NC}"
fi
echo ""

set +e
eval $CMD
RET=$?
set -e

echo ""
if [ $RET -eq 0 ]; then
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}   ✓ 完成${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}结果目录: $OUTPUT_DIR${NC}"
  echo -e "${CYAN}查看摘要: cat $OUTPUT_DIR/filter_summary.txt${NC}"
else
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   ✗ 失败 (exit code=$RET)${NC}"
  echo -e "${RED}========================================${NC}"
  exit $RET
fi

