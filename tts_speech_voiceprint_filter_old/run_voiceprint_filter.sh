#!/bin/bash

# Prompt-vs-Clone 声纹相似度筛选（十个子目录 + 根目录JSON）

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_JSON="$SCRIPT_DIR/config.json"

# 默认配置（仅保留 Prompt-vs-Clone 新流程）
THRESHOLD_DEFAULT="0.70"
NUM_WORKERS_DEFAULT="4"
MODEL_DIR_DEFAULT="/root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100"
NUM_GPUS_DEFAULT="4"

PROMPT_ROOT_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804"
PROMPT_RESULTS_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_prompt_clone_similarity/results"
PROMPT_LOGS_DIR_DEFAULT="/root/group-shared/voiceprint/share/voiceclone_child_20250804/tts_prompt_clone_similarity/logs"
# 可选：用于定位 prompt 的 Kaldi wav.scp 默认集合
WAV_SCP_DEFAULT=(
  "/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files/wav.scp"
  "/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files/wav.scp"
)

THRESHOLD="$THRESHOLD_DEFAULT"
NUM_WORKERS="$NUM_WORKERS_DEFAULT"
OUTPUT_PATH=""
VERBOSE=true
MODEL_DIR="$MODEL_DIR_DEFAULT"
NUM_GPUS="$NUM_GPUS_DEFAULT"
DEBUG=false
DEBUG_SAMPLES="100"
DEBUG_DIR=""
WAV_SCPS=()

# Prompt-vs-Clone 选项
PROMPT_ROOT="$PROMPT_ROOT_DEFAULT"
PROMPT_RESULTS_DIR="$PROMPT_RESULTS_DIR_DEFAULT"
PROMPT_LOGS_DIR="$PROMPT_LOGS_DIR_DEFAULT"

# TEN VAD 默认参数
VAD_FRAME_MS_DEFAULT="16"
VAD_MIN_SPEECH_MS_DEFAULT="160"
VAD_MAX_SILENCE_MS_DEFAULT="200"
VAD_FRAME_MS="$VAD_FRAME_MS_DEFAULT"
VAD_MIN_SPEECH_MS="$VAD_MIN_SPEECH_MS_DEFAULT"
VAD_MAX_SILENCE_MS="$VAD_MAX_SILENCE_MS_DEFAULT"

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --threshold FLOAT         相似度阈值 (默认: $THRESHOLD_DEFAULT)"
  echo "  --model_dir DIR           Multilingual WeSpeaker 模型目录 (默认: $MODEL_DIR_DEFAULT)"
  echo "  --num_gpus INT            使用GPU数量(多进程多卡) (默认: $NUM_GPUS_DEFAULT)"
  echo "  --num_workers INT         并行工作线程数 (默认: $NUM_WORKERS_DEFAULT)"
  echo "  --output PATH             输出JSON路径 (默认: 写入results目录)"
  echo "  --debug                   启用调试模式：随机打乱，仅取样100条并保存VAD图"
  echo "  --debug_samples INT       调试模式下采样条数 (默认: 100)"
  echo "  --debug_dir DIR           调试输出目录(存放波形+VAD图等)"
  echo "  --verbose                 详细日志"
  echo ""
  echo "Prompt-vs-Clone（基于十个子目录+根目录JSON）："
  echo "  --prompt_root DIR         根目录（包含十个子目录与JSON；默认: $PROMPT_ROOT_DEFAULT）"
  echo "  --prompt_results_dir DIR  结果输出目录（默认: $PROMPT_RESULTS_DIR_DEFAULT）"
  echo "  --prompt_logs_dir DIR     日志目录（默认: $PROMPT_LOGS_DIR_DEFAULT）"
  echo "  --wav_scp PATH            Kaldi wav.scp (可多次传入，用于定位prompt音频)"
  echo ""
  echo "TEN VAD 参数："
  echo "  --vad_frame_ms INT        帧长(ms) (默认: $VAD_FRAME_MS_DEFAULT)"
  echo "  --vad_min_speech_ms INT   最短语音段(ms) (默认: $VAD_MIN_SPEECH_MS_DEFAULT)"
  echo "  --vad_max_silence_ms INT  最长填补静音(ms) (默认: $VAD_MAX_SILENCE_MS_DEFAULT)"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 --threshold 0.9 --num_workers 8 --verbose"
  echo "  $0 --prompt_root $PROMPT_ROOT_DEFAULT --threshold 0.9 --num_workers 8 --verbose"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold)     THRESHOLD="$2"; shift 2;;
    --model_dir)     MODEL_DIR="$2"; shift 2;;
    --num_gpus)      NUM_GPUS="$2"; shift 2;;
    --num_workers)   NUM_WORKERS="$2"; shift 2;;
    --output)        OUTPUT_PATH="$2"; shift 2;;
    --debug)         DEBUG=true; shift;;
    --debug_samples) DEBUG_SAMPLES="$2"; shift 2;;
    --debug_dir)     DEBUG_DIR="$2"; shift 2;;
    --verbose)       VERBOSE=true; shift;;
    --prompt_root)         PROMPT_ROOT="$2"; shift 2;;
    --prompt_results_dir)  PROMPT_RESULTS_DIR="$2"; shift 2;;
    --prompt_logs_dir)     PROMPT_LOGS_DIR="$2"; shift 2;;
    --wav_scp)             WAV_SCPS+=("$2"); shift 2;;
    --vad_frame_ms)        VAD_FRAME_MS="$2"; shift 2;;
    --vad_min_speech_ms)   VAD_MIN_SPEECH_MS="$2"; shift 2;;
    --vad_max_silence_ms)  VAD_MAX_SILENCE_MS="$2"; shift 2;;
    -h|--help)       show_help; exit 0;;
    *) echo -e "${RED}未知参数: $1${NC}"; show_help; exit 1;;
  esac
done

# 目录检查与准备（Prompt-vs-Clone）
mkdir -p "$PROMPT_RESULTS_DIR" "$PROMPT_LOGS_DIR"
if [ -z "$OUTPUT_PATH" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT_PATH="$PROMPT_RESULTS_DIR/tts_prompt_clone_similarity_${TS}.json"
fi
# 若未显式提供 wav.scp，则使用默认集合（提高prompt定位成功率）
if [ ${#WAV_SCPS[@]} -eq 0 ]; then
  WAV_SCPS=("${WAV_SCP_DEFAULT[@]}")
fi

# 激活环境
if [[ "$CONDA_DEFAULT_ENV" != "SpeakerIdentify" ]]; then
  source /root/miniforge3/etc/profile.d/conda.sh
  conda activate SpeakerIdentify
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Prompt-vs-Clone 声纹相似度筛选${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}根目录: $PROMPT_ROOT${NC}"
echo -e "${YELLOW}阈值: $THRESHOLD${NC}"
echo -e "${YELLOW}模型目录: $MODEL_DIR${NC}"
echo -e "${YELLOW}GPU数量: $NUM_GPUS${NC}"
echo -e "${YELLOW}并行: $NUM_WORKERS${NC}"
echo -e "${YELLOW}输出: $OUTPUT_PATH${NC}"
echo -e "${YELLOW}TEN VAD: frame=${VAD_FRAME_MS}ms, min_speech=${VAD_MIN_SPEECH_MS}ms, max_silence=${VAD_MAX_SILENCE_MS}ms${NC}"
for s in "${WAV_SCPS[@]}"; do
  echo -e "${YELLOW}wav.scp: $s${NC}"
done
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

CMD="python3 \"$SCRIPT_DIR/compute_similarity_prompts.py\""
CMD="$CMD --root_dir \"$PROMPT_ROOT\""
CMD="$CMD --threshold \"$THRESHOLD\""
CMD="$CMD --model_dir \"$MODEL_DIR\""
CMD="$CMD --num_gpus \"$NUM_GPUS\""
CMD="$CMD --num_workers \"$NUM_WORKERS\""
CMD="$CMD --output \"$OUTPUT_PATH\""
CMD="$CMD --vad_frame_ms \"$VAD_FRAME_MS\""
CMD="$CMD --vad_min_speech_ms \"$VAD_MIN_SPEECH_MS\""
CMD="$CMD --vad_max_silence_ms \"$VAD_MAX_SILENCE_MS\""
for s in "${WAV_SCPS[@]}"; do
  CMD="$CMD --wav_scp \"$s\""
done
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



