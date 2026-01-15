#!/bin/bash

# 将filtered_speech目录下的TTS克隆音频及JSON文件拷贝到新的目录结构中
# 目标结构: cer_range/dataset_name/speaker_name/prompt_id/cloned_file

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 配置
SOURCE_DIRS=(
  "/root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech_sensevoice_cer0.25"
  "/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech_sensevoice_cer0.25"
)
OUTPUT_BASE_DIR="/root/group-shared/voiceprint/data/speech/voice_activity_detection/voiceclone_child_20520824_1022_copy_sensevoice_cer0.25"
NUM_WORKERS="32"

DRY_RUN=false
USE_HARDLINK=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --dry_run          模拟运行，不实际复制文件"
  echo "  --use_hardlink     使用硬链接代替复制"
  echo "  -h, --help         显示帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)      DRY_RUN=true; shift;;
    --use_hardlink) USE_HARDLINK=true; shift;;
    -h|--help)      show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TTS克隆音频及JSON复制 (V2)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}源目录:${NC}"
for DIR in "${SOURCE_DIRS[@]}"; do
  echo -e "  - $DIR"
done
echo -e "${YELLOW}目标目录: $OUTPUT_BASE_DIR${NC}"
echo -e "${YELLOW}模拟运行: $DRY_RUN${NC}"
echo -e "${YELLOW}使用硬链接: $USE_HARDLINK${NC}"
echo ""

CMD="python3 \"$SCRIPT_DIR/copy_clone_audio_v2.py\""
CMD="$CMD --source_dirs ${SOURCE_DIRS[*]}"
CMD="$CMD --output_base_dir \"$OUTPUT_BASE_DIR\""
CMD="$CMD --num_workers $NUM_WORKERS"

if [ "$DRY_RUN" = true ]; then
  CMD="$CMD --dry_run"
fi

if [ "$USE_HARDLINK" = true ]; then
  CMD="$CMD --use_hardlink"
fi

echo -e "${BLUE}执行命令: $CMD${NC}"
eval $CMD

echo -e "${GREEN}处理完成!${NC}"

