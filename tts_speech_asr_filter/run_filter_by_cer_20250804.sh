#!/bin/bash

# TTS音频按CER分类筛选Shell包装脚本（20250804数据集）
# 基于ASR筛选结果，按CER值对音频进行分类筛选

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 默认配置（20250804数据集）
ASR_RESULT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_asr_filter_sensevoice/results/tts_asr_filter_merged_all_parts_20251210_100140.json"
OUTPUT_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech_sensevoice_cer0.25"
CER_THRESHOLD_DEFAULT="0.25"
NUM_WORKERS_DEFAULT="16"
TARGET_SR_DEFAULT="16000"

ASR_RESULT="$ASR_RESULT_DEFAULT"
OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
CER_THRESHOLD="$CER_THRESHOLD_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
TARGET_SR="$TARGET_SR_DEFAULT"
NO_RESAMPLE=false
VERBOSE=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --asr_result PATH         ASR筛选结果JSON文件"
  echo "                            (默认: $ASR_RESULT_DEFAULT)"
  echo "  --output_dir DIR          输出目录"
  echo "                            (默认: $OUTPUT_DIR_DEFAULT)"
  echo "  --cer_threshold FLOAT     CER阈值，只处理CER <= 此值的音频 (默认: $CER_THRESHOLD_DEFAULT)"
  echo "  --num_workers INT         并行工作进程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --target_sr INT           目标采样率，如果指定则会将所有音频重采样到此采样率"
  echo "                            (使用librosa resample fft方法，默认: $TARGET_SR_DEFAULT)"
  echo "  --no_resample             不重采样，直接复制音频文件（覆盖--target_sr）"
  echo "  --verbose                 详细日志"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "说明:"
  echo "  - 基于ASR结果中的CER值对音频进行分类筛选"
  echo "  - CER分类规则:"
  echo "    * cer0: CER == 0"
  echo "    * cer0-0.05: 0 < CER <= 0.05"
  echo "    * cer0.05-0.1: 0.05 < CER <= 0.1"
  echo "    * cer0.1-0.15: 0.1 < CER <= 0.15"
  echo "    * cer0.15-0.2: 0.15 < CER <= 0.2"
  echo "    * cer0.2-0.25: 0.2 < CER <= 0.25"
  echo "    * cer0.25+: CER > 0.25 (不会复制，因为超过阈值)"
  echo "  - 输出目录结构:"
  echo "    output_dir/"
  echo "      ├── cer0/"
  echo "      │   └── <prompt_id>/"
  echo "      │       ├── <voiceprint_id>.wav"
  echo "      │       └── <voiceprint_id>.json"
  echo "      ├── cer0-0.05/"
  echo "      │   └── <prompt_id>/"
  echo "      │       ├── <voiceprint_id>.wav"
  echo "      │       └── <voiceprint_id>.json"
  echo "      └── ..."
  echo "  - 每个音频文件旁边会生成对应的JSON文件，包含:"
  echo "    * groundtruth_text: 原始文本"
  echo "    * transcription: ASR识别文本"
  echo "    * cer: 字符错误率"
  echo "    * 其他相关信息"
  echo "  - 音频重采样:"
  echo "    * 如果指定--target_sr，所有符合要求的音频将被重采样到目标采样率"
  echo "    * 使用librosa的resample fft方法进行重采样"
  echo "    * 如果音频原始采样率与目标采样率相同，则直接复制"
  echo "    * 使用--no_resample可以禁用重采样，直接复制原始音频"
  echo ""
  echo "示例:"
  echo "  # 使用默认配置"
  echo "  $0"
  echo ""
  echo "  # 自定义阈值和进程数"
  echo "  $0 --cer_threshold 0.20 --num_workers 32"
  echo ""
  echo "  # 指定输入和输出路径，并重采样到16kHz"
  echo "  $0 --asr_result /path/to/asr_result.json --output_dir /path/to/output --target_sr 16000"
  echo ""
  echo "  # 不重采样，直接复制原始音频"
  echo "  $0 --no_resample"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr_result)         ASR_RESULT="$2"; shift 2;;
    --output_dir)         OUTPUT_DIR="$2"; shift 2;;
    --cer_threshold)      CER_THRESHOLD="$2"; shift 2;;
    --num_workers)        NUM_WORKERS="$2"; shift 2;;
    --target_sr)          TARGET_SR="$2"; shift 2;;
    --no_resample)        NO_RESAMPLE=true; shift;;
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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TTS音频按CER分类筛选 (20250804)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}ASR结果: $ASR_RESULT${NC}"
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}CER阈值: $CER_THRESHOLD${NC}"
echo -e "${YELLOW}工作进程数: $NUM_WORKERS${NC}"
if [ "$NO_RESAMPLE" = true ]; then
  echo -e "${YELLOW}重采样: 禁用（直接复制）${NC}"
else
  echo -e "${YELLOW}目标采样率: $TARGET_SR Hz${NC}"
fi
echo ""

# 构建Python命令
CMD="python3 \"$SCRIPT_DIR/filter_by_cer_ranges.py\""
CMD="$CMD --asr_result \"$ASR_RESULT\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --cer_threshold $CER_THRESHOLD"
CMD="$CMD --num_workers $NUM_WORKERS"

if [ "$NO_RESAMPLE" = false ]; then
  CMD="$CMD --target_sr $TARGET_SR"
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
  echo ""
  echo -e "${CYAN}目录结构示例:${NC}"
  echo -e "${CYAN}  $OUTPUT_DIR/cer0/<prompt_id>/<voiceprint_id>.wav${NC}"
  echo -e "${CYAN}  $OUTPUT_DIR/cer0/<prompt_id>/<voiceprint_id>.json${NC}"
else
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   ✗ 失败 (exit code=$RET)${NC}"
  echo -e "${RED}========================================${NC}"
  exit $RET
fi

