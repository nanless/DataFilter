#!/bin/bash

# 音频数据重组Shell包装脚本
# 将filtered_speech目录下的音频按照说话人组织到目标目录

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 默认配置
SOURCE_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech"
TARGET_DIR_DEFAULT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments_20250808/merged_datasets_20250610_vad_segments/audio/cosyvoice2-kidclone-filtered-20251116"
GLOBAL_PREFIX_DEFAULT="cosyvoice2-kidclone-filtered-20251116"
NUM_WORKERS_DEFAULT="16"

UTT2SPK_BAAI_DEFAULT="/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/utt2spk"
UTT2SPK_CESSC_DEFAULT="/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/utt2spk"
UTT2SPK_KINGASR_DEFAULT="/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/utt2spk"
UTT2SPK_OCEAN_DEFAULT="/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/utt2spk"

SOURCE_DIR="$SOURCE_DIR_DEFAULT"
TARGET_DIR="$TARGET_DIR_DEFAULT"
GLOBAL_PREFIX="$GLOBAL_PREFIX_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
UTT2SPK_BAAI="$UTT2SPK_BAAI_DEFAULT"
UTT2SPK_CESSC="$UTT2SPK_CESSC_DEFAULT"
UTT2SPK_KINGASR="$UTT2SPK_KINGASR_DEFAULT"
UTT2SPK_OCEAN="$UTT2SPK_OCEAN_DEFAULT"
VERBOSE=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --source_dir DIR          源目录（filtered_speech）"
  echo "                            (默认: $SOURCE_DIR_DEFAULT)"
  echo "  --target_dir DIR          目标目录"
  echo "                            (默认: $TARGET_DIR_DEFAULT)"
  echo "  --global_prefix STR       全局标签前缀"
  echo "                            (默认: $GLOBAL_PREFIX_DEFAULT)"
  echo "  --num_workers INT         并行工作进程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --utt2spk_baai PATH       BAAI数据集的utt2spk文件"
  echo "  --utt2spk_cessc PATH      Chinese_English数据集的utt2spk文件"
  echo "  --utt2spk_kingasr PATH    King-ASR数据集的utt2spk文件"
  echo "  --utt2spk_ocean PATH      speechocean762数据集的utt2spk文件"
  echo "  --verbose                 详细日志"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "说明:"
  echo "  - 将filtered_speech目录下的音频按照说话人重新组织"
  echo "  - 目标目录结构: {全局标签}_{数据集标签}_{speaker_id}/{全局标签}_{prompt_id}.wav"
  echo "  - 例如: cosyvoice2-kidclone-filtered-20251116_BAAI_001/cosyvoice2-kidclone-filtered-20251116_001_5_M_L_LANZHOU_Android_002.wav"
  echo "  - 数据集标签: BAAI、CESSC、King-ASR、Ocean"
  echo ""
  echo "示例:"
  echo "  # 使用默认配置"
  echo "  $0"
  echo ""
  echo "  # 自定义目标目录和进程数"
  echo "  $0 --target_dir /path/to/output --num_workers 32"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_dir)       SOURCE_DIR="$2"; shift 2;;
    --target_dir)       TARGET_DIR="$2"; shift 2;;
    --global_prefix)    GLOBAL_PREFIX="$2"; shift 2;;
    --num_workers)      NUM_WORKERS="$2"; shift 2;;
    --utt2spk_baai)     UTT2SPK_BAAI="$2"; shift 2;;
    --utt2spk_cessc)    UTT2SPK_CESSC="$2"; shift 2;;
    --utt2spk_kingasr)  UTT2SPK_KINGASR="$2"; shift 2;;
    --utt2spk_ocean)    UTT2SPK_OCEAN="$2"; shift 2;;
    --verbose)          VERBOSE=true; shift;;
    -h|--help)          show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# 检查源目录
if [ ! -d "$SOURCE_DIR" ]; then
  echo -e "${RED}错误: 源目录不存在: $SOURCE_DIR${NC}"
  exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   音频数据重组${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}源目录: $SOURCE_DIR${NC}"
echo -e "${YELLOW}目标目录: $TARGET_DIR${NC}"
echo -e "${YELLOW}全局标签: $GLOBAL_PREFIX${NC}"
echo -e "${YELLOW}工作进程数: $NUM_WORKERS${NC}"
echo ""

# 构建Python命令
CMD="python3 \"$SCRIPT_DIR/reorganize_filtered_audio.py\""
CMD="$CMD --source_dir \"$SOURCE_DIR\""
CMD="$CMD --target_dir \"$TARGET_DIR\""
CMD="$CMD --global_prefix \"$GLOBAL_PREFIX\""
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --utt2spk_baai \"$UTT2SPK_BAAI\""
CMD="$CMD --utt2spk_cessc \"$UTT2SPK_CESSC\""
CMD="$CMD --utt2spk_kingasr \"$UTT2SPK_KINGASR\""
CMD="$CMD --utt2spk_ocean \"$UTT2SPK_OCEAN\""

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
  echo -e "${GREEN}目标目录: $TARGET_DIR${NC}"
  echo -e "${CYAN}查看摘要: cat $TARGET_DIR/reorganize_summary.txt${NC}"
else
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   ✗ 失败 (exit code=$RET)${NC}"
  echo -e "${RED}========================================${NC}"
  exit $RET
fi

