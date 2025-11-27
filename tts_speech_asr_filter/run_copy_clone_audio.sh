#!/bin/bash

# 将filtered_speech目录下的TTS克隆音频拷贝回原数据集目录结构

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 默认配置
SOURCE_DIRS_DEFAULT=(
  "/root/group-shared/voiceprint/share/voiceclone_child_20250804/filtered_speech"
  "/root/group-shared/voiceprint/share/voiceclone_child_20251022/filtered_speech"
)
OUTPUT_BASE_DIR_DEFAULT="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio"
NUM_WORKERS_DEFAULT="32"
PRINT_INTERVAL_DEFAULT="100"

SOURCE_DIRS=("${SOURCE_DIRS_DEFAULT[@]}")
OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
PRINT_INTERVAL="$PRINT_INTERVAL_DEFAULT"
REPORT_DIR=""
DRY_RUN=false
USE_HARDLINK=false
VERBOSE=false

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --source_dirs DIR1 [DIR2 ...]  源filtered_speech目录列表"
  echo "                                  (默认: ${SOURCE_DIRS_DEFAULT[*]})"
  echo "  --output_base_dir DIR           目标基础目录（包含各数据集子目录）"
  echo "                                  (默认: $OUTPUT_BASE_DIR_DEFAULT)"
  echo "  --report_dir DIR                报告输出目录"
  echo "                                  (默认: <output_base_dir>/copy_reports)"
  echo "  --num_workers INT               并行工作进程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --print_interval INT            dry_run模式下，每隔多少条音频打印一个示例"
  echo "                                  (默认: $PRINT_INTERVAL_DEFAULT)"
  echo "  --dry_run                       模拟运行，不实际复制文件"
  echo "  --use_hardlink                  使用硬链接代替复制（极快，节省空间）"
  echo "  --verbose                       详细日志"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "说明:"
  echo "  此脚本将从filtered_speech目录收集通过双重筛选的TTS克隆音频，"
  echo "  根据prompt音频id查找对应的原始数据集和说话人，"
  echo "  然后将克隆音频拷贝到目标目录的对应位置。"
  echo ""
  echo "  目标目录结构:"
  echo "    <output_base_dir>/"
  echo "      ├── childmandarin/<speaker_id>/<voiceprint_id>.wav"
  echo "      ├── chineseenglishchildren/<speaker_id>/<voiceprint_id>.wav"
  echo "      ├── king-asr-725/<speaker_id>/<voiceprint_id>.wav"
  echo "      ├── kingasr612/<speaker_id>/<voiceprint_id>.wav"
  echo "      └── speechocean762/<speaker_id>/<voiceprint_id>.wav"
  echo ""
  echo "  数据集映射关系来自以下utt2spk文件:"
  echo "    - BAAI-ChildMandarin41.25H/kaldi_files/utt2spk"
  echo "    - Chinese_English_Scripted_Speech_Corpus_Children/kaldi_files/utt2spk"
  echo "    - King-ASR-EN-Kid/kaldi_files/utt2spk"
  echo "    - speechocean762/kaldi_files/utt2spk"
  echo ""
  echo "示例:"
  echo "  # 使用默认配置"
  echo "  $0"
  echo ""
  echo "  # 自定义源目录"
  echo "  $0 --source_dirs /path/to/filtered_speech1 /path/to/filtered_speech2"
  echo ""
  echo "  # 模拟运行，查看将会如何复制"
  echo "  $0 --dry_run"
  echo ""
  echo "  # 使用更多进程加速"
  echo "  $0 --num_workers 32"
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_dirs)
      SOURCE_DIRS=()
      shift
      while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
        SOURCE_DIRS+=("$1")
        shift
      done
      ;;
    --output_base_dir)  OUTPUT_BASE_DIR="$2"; shift 2;;
    --report_dir)       REPORT_DIR="$2"; shift 2;;
    --num_workers)      NUM_WORKERS="$2"; shift 2;;
    --print_interval)   PRINT_INTERVAL="$2"; shift 2;;
    --dry_run)          DRY_RUN=true; shift;;
    --use_hardlink)     USE_HARDLINK=true; shift;;
    --verbose)          VERBOSE=true; shift;;
    -h|--help)          show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# 检查源目录
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
  if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}错误: 源目录不存在: $SOURCE_DIR${NC}"
    exit 1
  fi
done

# 检查目标目录
if [ ! -d "$OUTPUT_BASE_DIR" ]; then
  echo -e "${RED}错误: 目标基础目录不存在: $OUTPUT_BASE_DIR${NC}"
  exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TTS克隆音频复制到数据集目录${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}源目录:${NC}"
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
  echo -e "${YELLOW}  - $SOURCE_DIR${NC}"
done
echo -e "${YELLOW}目标目录: $OUTPUT_BASE_DIR${NC}"
if [ -n "$REPORT_DIR" ]; then
  echo -e "${YELLOW}报告目录: $REPORT_DIR${NC}"
else
  echo -e "${YELLOW}报告目录: $OUTPUT_BASE_DIR/copy_reports${NC}"
fi
echo -e "${YELLOW}工作进程数: $NUM_WORKERS${NC}"
if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}模拟运行: 是（不实际复制文件）${NC}"
  echo -e "${YELLOW}打印间隔: 每 $PRINT_INTERVAL 条打印一个示例${NC}"
else
  echo -e "${YELLOW}模拟运行: 否${NC}"
  if [ "$USE_HARDLINK" = true ]; then
    echo -e "${YELLOW}复制模式: 硬链接（极快，节省空间）${NC}"
  else
    echo -e "${YELLOW}复制模式: 标准复制${NC}"
  fi
fi
echo ""

# 构建Python命令
CMD="python3 \"$SCRIPT_DIR/copy_clone_audio_to_dataset.py\""

# 添加源目录参数
CMD="$CMD --source_dirs"
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
  CMD="$CMD \"$SOURCE_DIR\""
done

CMD="$CMD --output_base_dir \"$OUTPUT_BASE_DIR\""
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --print_interval $PRINT_INTERVAL"

if [ -n "$REPORT_DIR" ]; then
  CMD="$CMD --report_dir \"$REPORT_DIR\""
fi

if [ "$DRY_RUN" = true ]; then
  CMD="$CMD --dry_run"
fi

if [ "$USE_HARDLINK" = true ]; then
  CMD="$CMD --use_hardlink"
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
  echo -e "${GREEN}   ✓ 完成（全部成功）${NC}"
  echo -e "${GREEN}========================================${NC}"
  if [ -n "$REPORT_DIR" ]; then
    echo -e "${CYAN}查看摘要: cat $REPORT_DIR/copy_summary.txt${NC}"
  else
    echo -e "${CYAN}查看摘要: cat $OUTPUT_BASE_DIR/copy_reports/copy_summary.txt${NC}"
  fi
elif [ $RET -eq 2 ]; then
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}   ⚠ 完成（部分失败）${NC}"
  echo -e "${YELLOW}========================================${NC}"
  if [ -n "$REPORT_DIR" ]; then
    echo -e "${CYAN}查看详情: cat $REPORT_DIR/copy_summary.txt${NC}"
  else
    echo -e "${CYAN}查看详情: cat $OUTPUT_BASE_DIR/copy_reports/copy_summary.txt${NC}"
  fi
else
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}   ✗ 失败 (exit code=$RET)${NC}"
  echo -e "${RED}========================================${NC}"
  exit $RET
fi

