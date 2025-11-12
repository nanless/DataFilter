#!/bin/bash

# TTS 复刻音频与原始音频的声纹相似度筛选（Shell 包装）
# - 合并并解析映射 JSON
# - 从多个 wav.scp 中查找原始音频
# - 与 TTS zero_shot 下的音频成对计算相似度
# - 仅保留相似度 >= 阈值 的 TTS 音频

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_JSON="$SCRIPT_DIR/config.json"

# 读取默认配置
MAPPING_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022"
TTS_ZERO_SHOT_DEFAULT="/root/group-shared/speech_data/tts/cosyvoice2/voiceprint_enhance/20251015/test/models_batchsize16_with_voiceprint_diff-spkemb_1015-nonstream_1022/zero_shot"
WAV_SCP_DEFAULT=(
  "/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/wav.scp"
)
RESULTS_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_voiceprint_filter_addvad/results"
LOGS_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_voiceprint_filter_addvad/logs"
THRESHOLD_DEFAULT="0.70"
NUM_WORKERS_DEFAULT="4"
MODEL_DIR_DEFAULT="/root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100"
NUM_GPUS_DEFAULT="4"

MAPPING_DIR="$MAPPING_DIR_DEFAULT"
TTS_ZERO_SHOT="$TTS_ZERO_SHOT_DEFAULT"
RESULTS_DIR="$RESULTS_DIR_DEFAULT"
LOGS_DIR="$LOGS_DIR_DEFAULT"
THRESHOLD="$THRESHOLD_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
OUTPUT_PATH=""
VERBOSE=true
WAV_SCPS=()
MODEL_DIR="$MODEL_DIR_DEFAULT"
NUM_GPUS="$NUM_GPUS_DEFAULT"
DEBUG=false
DEBUG_SAMPLES="100"
DEBUG_DIR=""

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --mapping_dir DIR         映射JSON目录 (默认: $MAPPING_DIR_DEFAULT)"
  echo "  --tts_zero_shot DIR       TTS zero_shot目录 (默认: $TTS_ZERO_SHOT_DEFAULT)"
  echo "  --wav_scp PATH            Kaldi wav.scp 文件，可多次传入"
  echo "  --threshold FLOAT         相似度阈值 (默认: $THRESHOLD_DEFAULT)"
  echo "  --model_dir DIR           Multilingual WeSpeaker 模型目录 (默认: $MODEL_DIR_DEFAULT)"
  echo "  --num_gpus INT            使用GPU数量(多进程多卡) (默认: $NUM_GPUS_DEFAULT)"
  echo "  --num_workers INT         并行工作线程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --output PATH             输出JSON路径 (默认: 写入results目录)"
  echo "  --debug                   启用调试模式：随机打乱，仅取样100条并保存VAD图"
  echo "  --debug_samples INT       调试模式下采样条数 (默认: 100)"
  echo "  --debug_dir DIR           调试输出目录(存放波形+VAD图等)"
  echo "  --verbose                 详细日志"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 --threshold 0.9 --num_workers 8 --verbose"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mapping_dir)   MAPPING_DIR="$2"; shift 2;;
    --tts_zero_shot) TTS_ZERO_SHOT="$2"; shift 2;;
    --wav_scp)       WAV_SCPS+=("$2"); shift 2;;
    --threshold)     THRESHOLD="$2"; shift 2;;
    --model_dir)     MODEL_DIR="$2"; shift 2;;
    --num_gpus)      NUM_GPUS="$2"; shift 2;;
    --num_workers)   NUM_WORKERS="$2"; shift 2;;
    --output)        OUTPUT_PATH="$2"; shift 2;;
    --debug)         DEBUG=true; shift;;
    --debug_samples) DEBUG_SAMPLES="$2"; shift 2;;
    --debug_dir)     DEBUG_DIR="$2"; shift 2;;
    --verbose)       VERBOSE=true; shift;;
    -h|--help)       show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# WAV_SCP 默认
if [ ${#WAV_SCPS[@]} -eq 0 ]; then
  WAV_SCPS=("${WAV_SCP_DEFAULT[@]}")
fi

# 目录检查与准备
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
if [ -z "$OUTPUT_PATH" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT_PATH="$RESULTS_DIR/tts_voiceprint_filter_results_${TS}.json"
fi

# 激活环境
if [[ "$CONDA_DEFAULT_ENV" != "SpeakerIdentify" ]]; then
  source /root/miniforge3/etc/profile.d/conda.sh
  conda activate SpeakerIdentify
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TTS 复刻音频声纹相似度筛选${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}映射目录: $MAPPING_DIR${NC}"
echo -e "${YELLOW}TTS zero_shot: $TTS_ZERO_SHOT${NC}"
for s in "${WAV_SCPS[@]}"; do echo -e "${YELLOW}wav.scp: $s${NC}"; done
echo -e "${YELLOW}阈值: $THRESHOLD${NC}"
echo -e "${YELLOW}模型目录: $MODEL_DIR${NC}"
echo -e "${YELLOW}GPU数量: $NUM_GPUS${NC}"
echo -e "${YELLOW}并行: $NUM_WORKERS${NC}"
echo -e "${YELLOW}输出: $OUTPUT_PATH${NC}"
if [ "$DEBUG" = true ]; then
  echo -e "${YELLOW}调试模式: 开启${NC}"
  echo -e "${YELLOW}调试采样: $DEBUG_SAMPLES${NC}"
  if [ -n "$DEBUG_DIR" ]; then
    echo -e "${YELLOW}调试目录: $DEBUG_DIR${NC}"
  fi
else
  echo -e "${YELLOW}调试模式: 关闭${NC}"
fi
echo ""

CMD="python3 \"$SCRIPT_DIR/compute_similarity.py\""
CMD="$CMD --mapping_dir \"$MAPPING_DIR\""
CMD="$CMD --tts_zero_shot \"$TTS_ZERO_SHOT\""
for s in "${WAV_SCPS[@]}"; do
  CMD="$CMD --wav_scp \"$s\""
done
CMD="$CMD --threshold \"$THRESHOLD\""
CMD="$CMD --model_dir \"$MODEL_DIR\""
CMD="$CMD --num_gpus \"$NUM_GPUS\""
CMD="$CMD --num_workers \"$NUM_WORKERS\""
CMD="$CMD --output \"$OUTPUT_PATH\""
if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi
if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug --debug_samples \"$DEBUG_SAMPLES\""
  if [ -n "$DEBUG_DIR" ]; then
    CMD="$CMD --debug_dir \"$DEBUG_DIR\""
  fi
fi

echo -e "${CYAN}开始执行...${NC}"
echo -e "${YELLOW}$CMD${NC}"
eval $CMD

RET=$?
if [ $RET -eq 0 ]; then
  echo -e "${GREEN}✓ 完成: $OUTPUT_PATH${NC}"
else
  echo -e "${RED}✗ 失败 (exit code=$RET)${NC}"
  exit $RET
fi


