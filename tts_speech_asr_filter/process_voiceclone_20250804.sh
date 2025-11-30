#!/bin/bash

# 处理 voiceclone_child_20250804 目录下的所有 JSON 并计算 CER
#
# 功能：
# 1) 检查 /root/group-shared/voiceprint/share/voiceclone_child_20250804 下的所有 JSON 文件
# 2) 对每个 JSON 对应的音频目录执行 ASR 识别和 CER 计算
# 3) 默认使用 SenseVoice+NeMo TN 模式（可手动指定其他模式）
#
# 使用：
#   bash process_voiceclone_20250804.sh [选项]
# 选项：
#   --cer_threshold <float>       CER 阈值（默认 0.1）
#   --num_gpus <int>              使用的 GPU 数量
#   --language auto|zh|en         语言（默认 auto）
#   --use_whisper                 强制使用 Whisper+LLM 模式
#   --no-use_whisper              强制使用 Kimi 模式
#   --use_sensevoice              强制使用 SenseVoice Small+NeMo TN 模式
#   --whisper_model <model>       Whisper 模型（默认 large-v3）
#   --sensevoice_model_dir <dir>  SenseVoice 模型路径或ID（默认 iic/SenseVoiceSmall）
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
RESULTS_DIR="$DATA_DIR/tts_asr_filter_sensevoice/results"
LOG_DIR="$DATA_DIR/tts_asr_filter_sensevoice/logs"

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
USE_SENSEVOICE_OPT=""
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
            USE_SENSEVOICE_OPT=""
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
        --no-use_whisper)
            USE_WHISPER_OPT="--no-use_whisper"
            USE_SENSEVOICE_OPT=""
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
        --use_sensevoice)
            USE_SENSEVOICE_OPT="--use_sensevoice"
            USE_WHISPER_OPT=""
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

# 决定 ASR 模式（默认使用 SenseVoice+NeMo TN）
if [ -z "$USE_WHISPER_OPT" ] && [ -z "$USE_SENSEVOICE_OPT" ]; then
    echo -e "${CYAN}检测 ASR 模式（默认使用 SenseVoice+NeMo TN）...${NC}"
    echo -e "${GREEN}✓ 默认使用 SenseVoice+NeMo TN 模式${NC}"
    USE_SENSEVOICE_OPT="--use_sensevoice"
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
    
    # 收集所有part的base_dir
    ADDITIONAL_BASE_DIRS=()
    PRIMARY_BASE_DIR=""
    
    for part in "${PARTS_TO_PROCESS[@]}"; do
        PART_BASE_DIR="$DATA_DIR/voiceprint_20250804_part${part}_20250804"
        PART_ZERO_SHOT_DIR="$PART_BASE_DIR/zero_shot"
        
        if [ ! -d "$PART_ZERO_SHOT_DIR" ]; then
            echo -e "${YELLOW}警告: Part $part 的 zero_shot 目录不存在，跳过${NC}"
            continue
        fi
        
        if [ -z "$PRIMARY_BASE_DIR" ]; then
            PRIMARY_BASE_DIR="$PART_BASE_DIR"
        else
            ADDITIONAL_BASE_DIRS+=("$PART_BASE_DIR")
        fi
    done
    
    if [ -z "$PRIMARY_BASE_DIR" ]; then
        echo -e "${RED}错误: 没有找到任何有效的part目录${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}音频文件搜索配置:${NC}"
    echo -e "${YELLOW}  主目录: $PRIMARY_BASE_DIR${NC}"
    if [ ${#ADDITIONAL_BASE_DIRS[@]} -gt 0 ]; then
        echo -e "${YELLOW}  额外目录 (${#ADDITIONAL_BASE_DIRS[@]} 个):${NC}"
        for bd in "${ADDITIONAL_BASE_DIRS[@]}"; do
            echo -e "${YELLOW}    - $bd${NC}"
        done
    fi
    echo ""
    
    OUTPUT_JSON="$RESULTS_DIR/tts_asr_filter_combined_voiceclone_20250804_${TIMESTAMP}.json"
    LOG_FILE="$LOG_DIR/combined_${TIMESTAMP}.log"
    
    echo -e "${CYAN}开始 CER 计算...${NC}"
    echo -e "${YELLOW}JSON: $COMBINED_JSON${NC}"
    echo -e "${YELLOW}主基础目录: $PRIMARY_BASE_DIR${NC}"
    echo -e "${YELLOW}输出: $OUTPUT_JSON${NC}"
    echo -e "${YELLOW}日志: $LOG_FILE${NC}"
    echo ""
    
    # 调用 run_single_tts_filter.sh
    cd "$SCRIPT_DIR" || {
        echo -e "${RED}✗ 无法切换到脚本目录: $SCRIPT_DIR${NC}"
        exit 1
    }
    
    # 使用 set +e 来捕获错误，但继续执行
    set +e
    {
        # 构建额外的base_dir参数
        ADDITIONAL_DIRS_ARGS=()
        if [ ${#ADDITIONAL_BASE_DIRS[@]} -gt 0 ]; then
            ADDITIONAL_DIRS_ARGS=("--additional_base_dirs" "${ADDITIONAL_BASE_DIRS[@]}")
        fi
        
        if [ -n "$USE_SENSEVOICE_OPT" ]; then
            ./run_single_tts_filter.sh "$PRIMARY_BASE_DIR" "$COMBINED_JSON" ${USE_SENSEVOICE_OPT} \
                --output "$OUTPUT_JSON" \
                "${ADDITIONAL_DIRS_ARGS[@]}" \
                "${PASS_THROUGH_ARGS[@]}"
        elif [ -n "$USE_WHISPER_OPT" ]; then
            ./run_single_tts_filter.sh "$PRIMARY_BASE_DIR" "$COMBINED_JSON" ${USE_WHISPER_OPT} \
                --output "$OUTPUT_JSON" \
                "${ADDITIONAL_DIRS_ARGS[@]}" \
                "${PASS_THROUGH_ARGS[@]}"
        else
            ./run_single_tts_filter.sh "$PRIMARY_BASE_DIR" "$COMBINED_JSON" \
                --output "$OUTPUT_JSON" \
                "${ADDITIONAL_DIRS_ARGS[@]}" \
                "${PASS_THROUGH_ARGS[@]}"
        fi
    } >> "$LOG_FILE" 2>&1
    RET=$?
    set -e
    
    if [ $RET -eq 0 ]; then
        echo -e "${GREEN}✓ 合并处理完成${NC}"
        echo -e "${CYAN}结果文件: $OUTPUT_JSON${NC}"
        echo -e "${CYAN}日志文件: $LOG_FILE${NC}"
    else
        echo -e "${RED}✗ 合并处理失败 (退出码: $RET)${NC}"
        echo -e "${YELLOW}  查看日志: $LOG_FILE${NC}"
        exit 1
    fi
    
else
    # 分别处理模式：对每个 part 单独处理
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  模式: 分别处理${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0
    RESULT_FILES=()  # 记录成功的结果文件
    
    for part in "${PARTS_TO_PROCESS[@]}"; do
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}  处理 Part $part${NC}"
        echo -e "${BLUE}----------------------------------------${NC}"
        
        JSON_FILE="$DATA_DIR/voiceprint_20250804_part${part}_20250804.json"
        BASE_DIR="$DATA_DIR/voiceprint_20250804_part${part}_20250804"
        ZERO_SHOT_DIR="$BASE_DIR/zero_shot"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_JSON="$RESULTS_DIR/tts_asr_filter_part${part}_${TIMESTAMP}.json"
        LOG_FILE="$LOG_DIR/part${part}_${TIMESTAMP}.log"
        
        echo -e "${CYAN}JSON 文件: $JSON_FILE${NC}"
        echo -e "${CYAN}音频目录: $ZERO_SHOT_DIR${NC}"
        echo -e "${CYAN}输出文件: $OUTPUT_JSON${NC}"
        echo -e "${CYAN}日志文件: $LOG_FILE${NC}"
        echo ""
        
        # 验证文件和目录
        if [ ! -f "$JSON_FILE" ]; then
            echo -e "${RED}✗ JSON 文件不存在，跳过${NC}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Part $part: JSON文件不存在，跳过" >> "$LOG_FILE" 2>&1 || true
            ((SKIP_COUNT++))
            continue
        fi
        
        if [ ! -d "$ZERO_SHOT_DIR" ]; then
            echo -e "${RED}✗ zero_shot 目录不存在，跳过${NC}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Part $part: zero_shot目录不存在，跳过" >> "$LOG_FILE" 2>&1 || true
            ((SKIP_COUNT++))
            continue
        fi
        
        # 调用 run_single_tts_filter.sh
        cd "$SCRIPT_DIR" || {
            echo -e "${RED}✗ 无法切换到脚本目录: $SCRIPT_DIR${NC}"
            ((FAIL_COUNT++))
            continue
        }
        
        # 使用 set +e 来捕获错误，但继续执行
        set +e
        {
            if [ -n "$USE_SENSEVOICE_OPT" ]; then
                ./run_single_tts_filter.sh "$BASE_DIR" "$JSON_FILE" ${USE_SENSEVOICE_OPT} \
                    --output "$OUTPUT_JSON" \
                    "${PASS_THROUGH_ARGS[@]}"
            elif [ -n "$USE_WHISPER_OPT" ]; then
                ./run_single_tts_filter.sh "$BASE_DIR" "$JSON_FILE" ${USE_WHISPER_OPT} \
                    --output "$OUTPUT_JSON" \
                    "${PASS_THROUGH_ARGS[@]}"
            else
                ./run_single_tts_filter.sh "$BASE_DIR" "$JSON_FILE" \
                    --output "$OUTPUT_JSON" \
                    "${PASS_THROUGH_ARGS[@]}"
            fi
        } >> "$LOG_FILE" 2>&1
        RET=$?
        # 在循环内部保持 set +e，避免因为统计命令失败导致脚本退出
        set +e
        
        if [ $RET -eq 0 ]; then
            echo -e "${GREEN}✓ Part $part 处理完成${NC}"
            if [ -f "$OUTPUT_JSON" ]; then
                RESULT_FILES+=("$OUTPUT_JSON")
            fi
            ((SUCCESS_COUNT++)) || true
        else
            echo -e "${RED}✗ Part $part 处理失败 (退出码: $RET)${NC}"
            echo -e "${YELLOW}  查看日志: $LOG_FILE${NC}"
            ((FAIL_COUNT++)) || true
        fi
        echo "" || true
    done
    
    # 恢复 set -e 以便后续命令失败时能够退出
    set -e
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  处理完成统计${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}成功: $SUCCESS_COUNT${NC}"
    echo -e "${RED}失败: $FAIL_COUNT${NC}"
    if [ $SKIP_COUNT -gt 0 ]; then
        echo -e "${YELLOW}跳过: $SKIP_COUNT${NC}"
    fi
    echo -e "${CYAN}结果目录: $RESULTS_DIR${NC}"
    echo -e "${CYAN}日志目录: $LOG_DIR${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # 如果所有part都处理完成，输出总结
    TOTAL_PARTS=${#PARTS_TO_PROCESS[@]}
    PROCESSED_PARTS=$((SUCCESS_COUNT + FAIL_COUNT + SKIP_COUNT))
    if [ $PROCESSED_PARTS -lt $TOTAL_PARTS ]; then
        echo -e "${YELLOW}警告: 只处理了 $PROCESSED_PARTS/$TOTAL_PARTS 个 part${NC}"
        echo -e "${YELLOW}未处理的 part: ${NC}"
        for part in "${PARTS_TO_PROCESS[@]}"; do
            if [ ! -f "$RESULTS_DIR/tts_asr_filter_part${part}_"*.json ] 2>/dev/null; then
                echo -e "${YELLOW}  - Part $part${NC}"
            fi
        done
    fi
    
    # 合并所有成功的结果文件
    if [ ${#RESULT_FILES[@]} -gt 0 ]; then
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}  合并结果文件${NC}"
        echo -e "${CYAN}========================================${NC}"
        
        MERGE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        MERGED_OUTPUT_JSON="$RESULTS_DIR/tts_asr_filter_merged_all_parts_${MERGE_TIMESTAMP}.json"
        
        echo -e "${CYAN}找到 ${#RESULT_FILES[@]} 个结果文件，开始合并...${NC}"
        
        # 使用Python脚本合并结果
        python3 - "${RESULT_FILES[@]}" "$MERGED_OUTPUT_JSON" << 'PY'
import sys, json
from collections import defaultdict

result_files = sys.argv[1:-1]
out_path = sys.argv[-1]

# 合并所有结果
all_filter_results = []
merged_stats = {
    'total_files': 0,
    'processed_files': 0,
    'failed_files': 0,
    'filtered_files': 0,
    'passed_files': 0,
    'skipped_files': 0,
    'cer_values': []
}

source_files = []

for result_file in result_files:
    print(f"处理: {result_file}")
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_files.append(result_file)
        
        # 合并filter_results
        filter_results = data.get('filter_results', [])
        all_filter_results.extend(filter_results)
        
        # 合并统计信息
        stats = data.get('statistics', {})
        for key in ['total_files', 'processed_files', 'failed_files', 
                    'filtered_files', 'passed_files', 'skipped_files']:
            merged_stats[key] += stats.get(key, 0)
        
        cer_values = stats.get('cer_values', [])
        merged_stats['cer_values'].extend(cer_values)
        
    except Exception as e:
        print(f"[WARN] 读取失败: {result_file} -> {e}", file=sys.stderr)

# 计算CER统计
if merged_stats['cer_values']:
    try:
        import numpy as np
        merged_stats['cer_stats'] = {
            'mean': float(np.mean(merged_stats['cer_values'])),
            'median': float(np.median(merged_stats['cer_values'])),
            'std': float(np.std(merged_stats['cer_values'])),
            'min': float(np.min(merged_stats['cer_values'])),
            'max': float(np.max(merged_stats['cer_values']))
        }
    except ImportError:
        # 如果没有numpy，使用标准库计算
        cer_vals = sorted(merged_stats['cer_values'])
        n = len(cer_vals)
        merged_stats['cer_stats'] = {
            'mean': sum(cer_vals) / n,
            'median': cer_vals[n // 2] if n % 2 == 1 else (cer_vals[n // 2 - 1] + cer_vals[n // 2]) / 2,
            'std': (sum((x - sum(cer_vals) / n) ** 2 for x in cer_vals) / n) ** 0.5,
            'min': min(cer_vals),
            'max': max(cer_vals)
        }

# 构建合并后的结果
from datetime import datetime
merged_data = {
    'source_files': source_files,
    'timestamp': datetime.now().isoformat(),
    'statistics': merged_stats,
    'filter_results': all_filter_results
}

# 如果有base_dir和json_path信息，也保存（使用第一个文件的）
first_data = None
for result_file in result_files:
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            first_data = json.load(f)
            break
    except:
        continue

if first_data:
    if 'base_dir' in first_data:
        merged_data['base_dir'] = first_data['base_dir']
    if 'json_path' in first_data:
        merged_data['json_path'] = first_data['json_path']

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"\n[OK] 合并完成，共 {len(all_filter_results)} 个结果，输出: {out_path}")
print(f"统计信息:")
print(f"  总文件数: {merged_stats['total_files']}")
print(f"  处理成功: {merged_stats['processed_files']}")
print(f"  处理失败: {merged_stats['failed_files']}")
print(f"  通过筛选: {merged_stats['passed_files']}")
print(f"  被筛选: {merged_stats['filtered_files']}")
if 'cer_stats' in merged_stats:
    print(f"  CER平均值: {merged_stats['cer_stats']['mean']:.4f}")
PY
        
        if [ -f "$MERGED_OUTPUT_JSON" ]; then
            echo -e "${GREEN}✓ 结果文件合并完成${NC}"
            echo -e "${CYAN}合并结果文件: $MERGED_OUTPUT_JSON${NC}"
        else
            echo -e "${RED}✗ 结果文件合并失败${NC}"
        fi
    else
        echo -e "${YELLOW}没有成功的结果文件需要合并${NC}"
    fi
fi

echo -e "${GREEN}✓ 全部完成${NC}"

