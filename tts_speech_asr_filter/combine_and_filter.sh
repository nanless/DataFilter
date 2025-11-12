#!/bin/bash

# 合并 JSON 并对指定 zero_shot 音频执行基于识别结果的筛选
#
# 功能：
# 1) 合并 /root/group-shared/voiceprint/share/voiceclone_child_20251022 下所有 JSON 为一个总 JSON
# 2) 对 /root/group-shared/speech_data/tts/cosyvoice2/voiceprint_enhance/20251015/test/models_batchsize16_with_voiceprint_diff-spkemb_1015-nonstream_1022/zero_shot
#    下的音频执行筛选（自动选择 Whisper+LLM；如果 LLM 不可用则回退为 Kimi 模式）
#
# 使用：
#   bash combine_and_filter.sh [选项]
# 选项（传递给 run_single_tts_filter.sh）：
#   --cer_threshold <float>
#   --num_gpus <int>
#   --language auto|zh|en
#   --use_whisper | --no-use_whisper
#   --whisper_model <tiny|base|small|medium|large|large-v2|large-v3>
#   --output </abs/path/to/output.json>
#
# 说明：
# - 默认尝试 Whisper+LLM（检测 http://localhost:8000/health），失败后自动回退到 Kimi 模式
# - 若明确指定 --no-use_whisper，则强制使用 Kimi 模式
# - 可额外传入 run_single_tts_filter.sh 支持的其他参数（如 --test_mode、--verbose、--force 等）

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_asr_filter/results"
LOG_DIR="/root/group-shared/voiceprint/share/voiceclone_child_20251022/tts_asr_filter/logs"

# 路径常量（根据需求可调整）
JSON_SRC_DIR="/root/group-shared/voiceprint/share/voiceclone_child_20251022"
AUDIO_ZERO_SHOT_DIR="/root/group-shared/speech_data/tts/cosyvoice2/voiceprint_enhance/20251015/test/models_batchsize16_with_voiceprint_diff-spkemb_1015-nonstream_1022/zero_shot"
BASE_DIR="$(dirname "$AUDIO_ZERO_SHOT_DIR")"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    合并JSON并执行TTS音频筛选${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}JSON源目录: $JSON_SRC_DIR${NC}"
echo -e "${YELLOW}音频zero_shot目录: $AUDIO_ZERO_SHOT_DIR${NC}"
echo -e "${YELLOW}BASE_DIR(自动): $BASE_DIR${NC}"
echo ""

if [ ! -d "$JSON_SRC_DIR" ]; then
    echo -e "${RED}错误: JSON源目录不存在: $JSON_SRC_DIR${NC}"
    exit 1
fi
if [ ! -d "$AUDIO_ZERO_SHOT_DIR" ]; then
    echo -e "${RED}错误: zero_shot目录不存在: $AUDIO_ZERO_SHOT_DIR${NC}"
    exit 1
fi
if [ ! -d "$BASE_DIR/zero_shot" ]; then
    echo -e "${RED}错误: BASE_DIR 下未找到 zero_shot: $BASE_DIR/zero_shot${NC}"
    exit 1
fi

# 1) 合并 JSON
COMBINED_JSON="$RESULTS_DIR/combined_voiceclone_child_20251022_$(date +%Y%m%d_%H%M%S).json"
echo -e "${CYAN}开始合并JSON...${NC}"

python3 - "$JSON_SRC_DIR" "$COMBINED_JSON" << 'PY'
import os, sys, json, glob

json_src_dir = sys.argv[1]
out_path = sys.argv[2]

merged = {}
files = sorted(glob.glob(os.path.join(json_src_dir, "*.json")))

if not files:
    print(f"[ERROR] 未在目录中找到JSON: {json_src_dir}", file=sys.stderr)
    sys.exit(2)

def normalize_entry(entry):
    # 期待格式: "voiceprint_id\ttext"
    if not isinstance(entry, str):
        return None
    parts = entry.split('\t', 1)
    if len(parts) != 2:
        return None
    vp, txt = parts[0].strip(), parts[1].strip()
    if not vp or not txt:
        return None
    return f"{vp}\t{txt}"

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for prompt_id, entries in data.items():
            if not isinstance(entries, list):
                continue
            bucket = merged.setdefault(prompt_id, [])
            # 去重合并
            existing = set(bucket)
            for e in entries:
                norm = normalize_entry(e)
                if norm and norm not in existing:
                    bucket.append(norm)
                    existing.add(norm)
    except Exception as e:
        print(f"[WARN] 读取/合并失败: {fp} -> {e}", file=sys.stderr)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"[OK] 合并完成，共 {len(merged)} 个 prompt，输出: {out_path}")
PY

if [ ! -f "$COMBINED_JSON" ]; then
    echo -e "${RED}错误: 合并结果JSON未生成${NC}"
    exit 1
fi

echo -e "${GREEN}✓ JSON合并完成: $COMBINED_JSON${NC}"
echo ""

# 2) 选择ASR模式并执行筛选
USE_WHISPER_OPT="--use_whisper"
EXPLICIT_MODE=""

# 若用户通过参数显式指定了模式，则遵从
for arg in "$@"; do
    if [ "$arg" = "--no-use_whisper" ]; then
        EXPLICIT_MODE="kimi"
    elif [ "$arg" = "--use_whisper" ]; then
        EXPLICIT_MODE="whisper"
    fi
done

if [ -z "$EXPLICIT_MODE" ]; then
    echo -e "${CYAN}检测LLM服务 (http://localhost:8000/health)...${NC}"
    if curl -s --connect-timeout 3 http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ 检测到LLM服务，使用 Whisper+LLM 模式${NC}"
        USE_WHISPER_OPT="--use_whisper"
    else
        echo -e "${YELLOW}未检测到LLM服务，回退至 Kimi-Audio 模式${NC}"
        USE_WHISPER_OPT="--no-use_whisper"
    fi
else
    if [ "$EXPLICIT_MODE" = "whisper" ]; then
        USE_WHISPER_OPT="--use_whisper"
    else
        USE_WHISPER_OPT="--no-use_whisper"
    fi
fi

OUTPUT_PATH_DEFAULT="$RESULTS_DIR/tts_asr_filter_results_combined_voiceclone_child.json"

echo -e "${CYAN}开始执行筛选...${NC}"
echo -e "${YELLOW}BASE_DIR: $BASE_DIR${NC}"
echo -e "${YELLOW}JSON_FILE: $COMBINED_JSON${NC}"
echo -e "${YELLOW}模式: ${USE_WHISPER_OPT}${NC}"
echo ""

cd "$SCRIPT_DIR"

# 若用户未指定 --output，则追加默认输出
PASS_THROUGH_ARGS=("$@")
if ! printf '%s\n' "${PASS_THROUGH_ARGS[@]}" | grep -q -- '--output'; then
    PASS_THROUGH_ARGS+=("--output" "$OUTPUT_PATH_DEFAULT")
fi

set +e
./run_single_tts_filter.sh "$BASE_DIR" "$COMBINED_JSON" ${USE_WHISPER_OPT} "${PASS_THROUGH_ARGS[@]}"
RET=$?
set -e

if [ $RET -ne 0 ]; then
    echo -e "${RED}✗ 筛选执行失败，退出码: $RET${NC}"
    exit $RET
fi

echo -e "${GREEN}✓ 完成。合并JSON: $COMBINED_JSON${NC}"
echo -e "${GREEN}✓ 筛选结果: $(printf '%s\n' "${PASS_THROUGH_ARGS[@]}" | awk '/--output/{getline; print; exit}')"${NC}
echo -e "${CYAN}日志目录: $LOG_DIR${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}    全流程完成${NC}"
echo -e "${BLUE}========================================${NC}"


