#!/bin/bash

# 按声纹相似度分档抽样脚本（Shell包装）
# 用于人工听辨不同相似度档次的音频质量差异

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 默认配置
RESULT_JSON_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_voiceprint_filter_dualsim/results/tts_voiceprint_filter_results_20251114_120918.json"
OUTPUT_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/similarity_samples"
SAMPLES_PER_BIN_DEFAULT="20"
SIMILARITY_TYPE_DEFAULT="both"
SEED_DEFAULT="42"

RESULT_JSON="$RESULT_JSON_DEFAULT"
OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
SAMPLES_PER_BIN="$SAMPLES_PER_BIN_DEFAULT"
SIMILARITY_TYPE="$SIMILARITY_TYPE_DEFAULT"
SEED="$SEED_DEFAULT"
VERBOSE=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --result_json PATH        筛选结果JSON文件（声纹筛选结果或合并结果）"
  echo "                            (默认: 声纹筛选结果文件)"
  echo "  --output_dir DIR          输出目录（存放抽样样本）"
  echo "                            (默认: $OUTPUT_DIR_DEFAULT)"
  echo "  --samples_per_bin INT     每个档次抽取的样本数 (默认: $SAMPLES_PER_BIN_DEFAULT)"
  echo "  --similarity_type TYPE    相似度类型: vad, original, both (默认: $SIMILARITY_TYPE_DEFAULT)"
  echo "  --seed INT                随机种子，确保可重复 (默认: $SEED_DEFAULT)"
  echo "  --verbose                 详细日志"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "功能说明:"
  echo "  - 直接读取声纹筛选结果，无需依赖合并脚本"
  echo "  - 按声纹相似度分档（细分负值和低值区域）:"
  echo "    -1.0~0.0（负值）, 0.0~0.3（极低）, 0.3~0.5（很低）,"
  echo "    0.5~0.6（低）, 0.6~0.7（中等偏低）, 0.7~0.8（中等）,"
  echo "    0.8~0.9（高）, 0.9~1.0（极高）"
  echo "  - 支持三种抽样模式:"
  echo "    vad: 仅按VAD处理后的相似度抽样"
  echo "    original: 仅按原始音频的相似度抽样"
  echo "    both: 同时按两种相似度分别抽样（推荐）"
  echo "  - 每个档次随机抽取N个样本"
  echo "  - 复制原始音频（source）和TTS复刻音频（tts）"
  echo "  - 生成详细信息文件（CER、相似度、文本等）"
  echo ""
  echo "输出目录结构（both模式）:"
  echo "  output_dir/"
  echo "    ├── vad/                       # VAD相似度样本"
  echo "    │   ├── similarity_-1.00-0.00/"
  echo "    │   ├── similarity_0.00-0.30/"
  echo "    │   ├── similarity_0.30-0.50/"
  echo "    │   ├── similarity_0.50-0.60/"
  echo "    │   └── ..."
  echo "    ├── original/                  # Original相似度样本"
  echo "    │   ├── similarity_-1.00-0.00/"
  echo "    │   ├── similarity_0.00-0.30/"
  echo "    │   └── ..."
  echo "    └── samples_summary.txt"
  echo ""
  echo "示例:"
  echo "  # 使用默认配置（读取声纹筛选结果，both模式，每档20个样本）"
  echo "  $0"
  echo ""
  echo "  # 只抽取VAD相似度，每档30个样本"
  echo "  $0 --similarity_type vad --samples_per_bin 30"
  echo ""
  echo "  # 指定自定义的声纹筛选结果文件"
  echo "  $0 --result_json /path/to/voiceprint_results.json"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result_json)       RESULT_JSON="$2"; shift 2;;
    --output_dir)        OUTPUT_DIR="$2"; shift 2;;
    --samples_per_bin)   SAMPLES_PER_BIN="$2"; shift 2;;
    --similarity_type)   SIMILARITY_TYPE="$2"; shift 2;;
    --seed)              SEED="$2"; shift 2;;
    --verbose)           VERBOSE=true; shift;;
    -h|--help)           show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# 检查输入文件
if [ ! -f "$RESULT_JSON" ]; then
  echo -e "${RED}错误: 结果文件不存在: $RESULT_JSON${NC}"
  echo -e "${YELLOW}提示: 请先运行声纹筛选生成结果文件${NC}"
  echo -e "${YELLOW}或者指定正确的结果文件路径: --result_json /path/to/results.json${NC}"
  exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   按声纹相似度分档抽样${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}输入文件: $RESULT_JSON${NC}"
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}每档样本数: $SAMPLES_PER_BIN${NC}"
echo -e "${YELLOW}相似度类型: $SIMILARITY_TYPE${NC}"
echo -e "${YELLOW}随机种子: $SEED${NC}"
echo ""

# 构建Python命令
CMD="python3 \"$SCRIPT_DIR/sample_by_similarity.py\""
CMD="$CMD --result_json \"$RESULT_JSON\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --samples_per_bin $SAMPLES_PER_BIN"
CMD="$CMD --similarity_type $SIMILARITY_TYPE"
CMD="$CMD --seed $SEED"

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
  echo -e "${GREEN}样本目录: $OUTPUT_DIR${NC}"
  echo -e "${CYAN}查看摘要: cat $OUTPUT_DIR/samples_summary.txt${NC}"
  echo ""
  echo -e "${YELLOW}听辨建议:${NC}"
  echo -e "  1. 先从低相似度档次(0.5-0.6)开始听"
  echo -e "  2. 对比每个样本的 source.wav 和 tts.wav"
  echo -e "  3. 查看 info.txt 了解详细信息"
  echo -e "  4. 逐步听到高相似度档次(0.9-1.0)"
  echo -e "  5. 总结不同相似度档次的音质特点"
else
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   ✗ 失败 (exit code=$RET)${NC}"
  echo -e "${RED}========================================${NC}"
  exit $RET
fi

