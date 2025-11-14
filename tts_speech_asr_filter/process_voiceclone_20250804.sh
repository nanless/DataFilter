#!/bin/bash

# 处理 voiceclone_child_20250804 目录下的所有 JSON 并计算 CER
#
# 功能：
# 1) 检查 /root/group-shared/voiceprint/share/voiceclone_child_20250804 下的所有 JSON 文件
# 2) 对每个 JSON 对应的音频目录执行 ASR 识别和 CER 计算
# 3) 自动选择 Whisper+LLM 模式（如果 LLM 不可用则回退为 Kimi 模式）
#
# 使用：
#   bash process_voiceclone_20250804.sh [选项]
# 选项：
#   --cer_threshold <float>       CER 阈值（默认 0.1）
#   --num_gpus <int>              使用的 GPU 数量
#   --language auto|zh|en         语言（默认 auto）
#   --use_whisper                 强制使用 Whisper+LLM 模式
#   --no-use_whisper              强制使用 Kimi 模式
#   --whisper_model <model>       Whisper 模型（默认 large-v3）
#   --process_parts <parts>       指定处理哪些 part，如 "1,2,5" 或 "all"（默认 all）
#   --merge                       是否先合并所有 JSON 再统一处理（默认分别处理）
#   --test_mode                   测试模式（只处理少量数据）
#   --verbose                     详细输出
#   --force                       强制重新处理已有结果

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/root/group-shared/voiceprint/share/voiceclone_child_20250804"
RESULTS_DIR="$DATA_DIR/tts_asr_filter/results"
LOG_DIR="$DATA_DIR/tts_asr_filter/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  处理 voiceclone_20250804 数据集${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}数据目录: $DATA_DIR${NC}"
echo ""

# 检查目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    exit 1
fi

# 解析参数
PROCESS_PARTS="all"
MERGE_MODE=false
USE_WHISPER_OPT=""
PASS_THROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --process_parts)
            PROCESS_PARTS="$2"
            shift 2
            ;;
        --merge)
            MERGE_MODE=true
            shift
            ;;
        --use_whisper)
            USE_WHISPER_OPT="--use_whisper"
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
        --no-use_whisper)
            USE_WHISPER_OPT="--no-use_whisper"
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# 获取所有 JSON 文件
echo -e "${CYAN}扫描 JSON 文件...${NC}"
JSON_FILES=($(ls "$DATA_DIR"/voiceprint_20250804_part*_20250804.json 2>/dev/null | sort))

if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 未找到任何 JSON 文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 ${#JSON_FILES[@]} 个 JSON 文件:${NC}"
for json_file in "${JSON_FILES[@]}"; do
    basename=$(basename "$json_file" .json)
    part_num=$(echo "$basename" | grep -oP 'part\K\d+')
    audio_dir="$DATA_DIR/${basename}/zero_shot"
    
    if [ -d "$audio_dir" ]; then
        audio_count=$(find "$audio_dir" -name "*.wav" 2>/dev/null | wc -l)
        echo -e "  ${CYAN}[$part_num]${NC} $json_file"
        echo -e "      → $audio_dir (${audio_count} 个音频文件)"
    else
        echo -e "  ${YELLOW}[$part_num]${NC} $json_file"
        echo -e "      ${RED}→ 音频目录不存在: $audio_dir${NC}"
    fi
done
echo ""

# 筛选要处理的 parts
PARTS_TO_PROCESS=()
if [ "$PROCESS_PARTS" = "all" ]; then
    for json_file in "${JSON_FILES[@]}"; do
        basename=$(basename "$json_file" .json)
        part_num=$(echo "$basename" | grep -oP 'part\K\d+')
        audio_dir="$DATA_DIR/${basename}/zero_shot"
        if [ -d "$audio_dir" ]; then
            PARTS_TO_PROCESS+=("$part_num")
        fi
    done
else
    IFS=',' read -ra PARTS_ARRAY <<< "$PROCESS_PARTS"
    for part in "${PARTS_ARRAY[@]}"; do
        part=$(echo "$part" | xargs)  # trim whitespace
        json_file="$DATA_DIR/voiceprint_20250804_part${part}_20250804.json"
        audio_dir="$DATA_DIR/voiceprint_20250804_part${part}_20250804/zero_shot"
        if [ -f "$json_file" ] && [ -d "$audio_dir" ]; then
            PARTS_TO_PROCESS+=("$part")
        else
            echo -e "${YELLOW}警告: Part $part 的文件不完整，跳过${NC}"
        fi
    done
fi

if [ ${#PARTS_TO_PROCESS[@]} -eq 0 ]; then
    echo -e "${RED}错误: 没有可处理的数据${NC}"
    exit 1
fi

echo -e "${GREEN}将处理 ${#PARTS_TO_PROCESS[@]} 个 part: ${PARTS_TO_PROCESS[*]}${NC}"
echo ""

# 决定 ASR 模式
if [ -z "$USE_WHISPER_OPT" ]; then
    echo -e "${CYAN}检测 LLM 服务 (http://localhost:8000/health)...${NC}"
    if curl -s --connect-timeout 3 http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ 检测到 LLM 服务，使用 Whisper+LLM 模式${NC}"
        USE_WHISPER_OPT="--use_whisper"
    else
        echo -e "${YELLOW}未检测到 LLM 服务，回退至 Kimi-Audio 模式${NC}"
        USE_WHISPER_OPT="--no-use_whisper"
    fi
fi
echo ""

# 处理模式
if [ "$MERGE_MODE" = true ]; then
    # 合并模式：先合并所有要处理的 JSON，再统一处理
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  模式: 合并处理${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    COMBINED_JSON="$RESULTS_DIR/combined_voiceclone_20250804_parts_${TIMESTAMP}.json"
    
    echo -e "${CYAN}合并 JSON 文件...${NC}"
    
    # 构建要合并的 JSON 文件列表
    JSON_TO_MERGE=()
    for part in "${PARTS_TO_PROCESS[@]}"; do
        JSON_TO_MERGE+=("$DATA_DIR/voiceprint_20250804_part${part}_20250804.json")
    done
    
    # Python 脚本合并 JSON
    python3 - "${JSON_TO_MERGE[@]}" "$COMBINED_JSON" << 'PY'
import sys, json

json_files = sys.argv[1:-1]
out_path = sys.argv[-1]

merged = {}

def normalize_entry(entry):
    if not isinstance(entry, str):
        return None
    parts = entry.split('\t', 1)
    if len(parts) != 2:
        return None
    vp, txt = parts[0].strip(), parts[1].strip()
    if not vp or not txt:
        return None
    return f"{vp}\t{txt}"

for fp in json_files:
    print(f"处理: {fp}")
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for prompt_id, entries in data.items():
            if not isinstance(entries, list):
                continue
            bucket = merged.setdefault(prompt_id, [])
            existing = set(bucket)
            for e in entries:
                norm = normalize_entry(e)
                if norm and norm not in existing:
                    bucket.append(norm)
                    existing.add(norm)
    except Exception as e:
        print(f"[WARN] 读取失败: {fp} -> {e}", file=sys.stderr)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"\n[OK] 合并完成，共 {len(merged)} 个 prompt，输出: {out_path}")
PY
    
    if [ ! -f "$COMBINED_JSON" ]; then
        echo -e "${RED}错误: 合并失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ JSON 合并完成: $COMBINED_JSON${NC}"
    echo ""
    
    # 找出所有对应的 zero_shot 目录的共同父目录
    # 这里假设所有 part 的 zero_shot 都在各自的目录下，需要特殊处理
    echo -e "${YELLOW}注意: 合并模式下需要手动指定音频基础目录${NC}"
    echo -e "${YELLOW}建议使用分别处理模式（不加 --merge 参数）${NC}"
    echo ""
    
    OUTPUT_JSON="$RESULTS_DIR/tts_asr_filter_combined_voiceclone_20250804_${TIMESTAMP}.json"
    
    echo -e "${CYAN}开始 CER 计算...${NC}"
    echo -e "${YELLOW}JSON: $COMBINED_JSON${NC}"
    echo -e "${YELLOW}输出: $OUTPUT_JSON${NC}"
    echo ""
    
    # 这里需要特殊处理，因为音频文件分散在多个 part 目录下
    # 暂时跳过实际执行，提示用户
    echo -e "${YELLOW}合并模式需要特殊处理音频路径，请使用分别处理模式${NC}"
    
else
    # 分别处理模式：对每个 part 单独处理
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  模式: 分别处理${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for part in "${PARTS_TO_PROCESS[@]}"; do
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}  处理 Part $part${NC}"
        echo -e "${BLUE}----------------------------------------${NC}"
        
        JSON_FILE="$DATA_DIR/voiceprint_20250804_part${part}_20250804.json"
        BASE_DIR="$DATA_DIR/voiceprint_20250804_part${part}_20250804"
        ZERO_SHOT_DIR="$BASE_DIR/zero_shot"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_JSON="$RESULTS_DIR/tts_asr_filter_part${part}_${TIMESTAMP}.json"
        
        echo -e "${CYAN}JSON 文件: $JSON_FILE${NC}"
        echo -e "${CYAN}音频目录: $ZERO_SHOT_DIR${NC}"
        echo -e "${CYAN}输出文件: $OUTPUT_JSON${NC}"
        echo ""
        
        # 验证文件和目录
        if [ ! -f "$JSON_FILE" ]; then
            echo -e "${RED}✗ JSON 文件不存在，跳过${NC}"
            ((FAIL_COUNT++))
            continue
        fi
        
        if [ ! -d "$ZERO_SHOT_DIR" ]; then
            echo -e "${RED}✗ zero_shot 目录不存在，跳过${NC}"
            ((FAIL_COUNT++))
            continue
        fi
        
        # 调用 run_single_tts_filter.sh
        cd "$SCRIPT_DIR"
        
        set +e
        ./run_single_tts_filter.sh "$BASE_DIR" "$JSON_FILE" $USE_WHISPER_OPT \
            --output "$OUTPUT_JSON" \
            "${PASS_THROUGH_ARGS[@]}"
        RET=$?
        set -e
        
        if [ $RET -eq 0 ]; then
            echo -e "${GREEN}✓ Part $part 处理完成${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}✗ Part $part 处理失败 (退出码: $RET)${NC}"
            ((FAIL_COUNT++))
        fi
        echo ""
    done
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  处理完成统计${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}成功: $SUCCESS_COUNT${NC}"
    echo -e "${RED}失败: $FAIL_COUNT${NC}"
    echo -e "${CYAN}结果目录: $RESULTS_DIR${NC}"
    echo -e "${CYAN}日志目录: $LOG_DIR${NC}"
    echo -e "${BLUE}========================================${NC}"
fi

echo -e "${GREEN}✓ 全部完成${NC}"

