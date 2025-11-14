#!/bin/bash

# 检查 voiceclone_child_20250804 目录结构
# 显示所有 JSON 文件及其对应的音频目录信息

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

DATA_DIR="/root/group-shared/voiceprint/share/voiceclone_child_20250804"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  检查 voiceclone_20250804 数据集${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 目录不存在: $DATA_DIR${NC}"
    exit 1
fi

echo -e "${CYAN}数据目录: $DATA_DIR${NC}"
echo ""

# 获取所有 JSON 文件
JSON_FILES=($(ls "$DATA_DIR"/voiceprint_20250804_part*_20250804.json 2>/dev/null | sort))

if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 未找到任何 JSON 文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 ${#JSON_FILES[@]} 个 JSON 文件${NC}"
echo ""

TOTAL_JSON_ENTRIES=0
TOTAL_AUDIO_FILES=0
TOTAL_AUDIO_DIRS=0

echo -e "${BLUE}详细信息:${NC}"
echo -e "${BLUE}----------------------------------------${NC}"

for json_file in "${JSON_FILES[@]}"; do
    basename=$(basename "$json_file" .json)
    part_num=$(echo "$basename" | grep -oP 'part\K\d+')
    audio_base_dir="$DATA_DIR/${basename}"
    audio_zero_shot_dir="$audio_base_dir/zero_shot"
    
    # 统计 JSON 条目数
    json_entries=0
    if [ -f "$json_file" ]; then
        # 统计 JSON 中 prompt_id 的数量
        json_entries=$(python3 -c "import json; data=json.load(open('$json_file')); print(len(data))" 2>/dev/null || echo "0")
        
        # 统计 JSON 中所有 voiceprint 条目的总数
        json_total_items=$(python3 -c "import json; data=json.load(open('$json_file')); print(sum(len(v) for v in data.values()))" 2>/dev/null || echo "0")
    fi
    
    echo -e "${CYAN}Part $part_num:${NC}"
    echo -e "  JSON: $(basename "$json_file")"
    echo -e "    - Prompt IDs: ${GREEN}$json_entries${NC}"
    echo -e "    - 总条目数: ${GREEN}$json_total_items${NC}"
    
    if [ -d "$audio_zero_shot_dir" ]; then
        # 统计音频文件数
        audio_count=$(find "$audio_zero_shot_dir" -name "*.wav" 2>/dev/null | wc -l)
        # 统计 prompt 子目录数
        prompt_dirs=$(find "$audio_zero_shot_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        echo -e "  ${GREEN}✓${NC} 音频目录: $audio_zero_shot_dir"
        echo -e "    - Prompt 目录数: ${GREEN}$prompt_dirs${NC}"
        echo -e "    - 音频文件数: ${GREEN}$audio_count${NC}"
        
        # 显示第一个 prompt 目录作为示例
        first_prompt_dir=$(find "$audio_zero_shot_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
        if [ -n "$first_prompt_dir" ]; then
            first_prompt_name=$(basename "$first_prompt_dir")
            first_prompt_audio_count=$(ls "$first_prompt_dir"/*.wav 2>/dev/null | wc -l)
            echo -e "    ${YELLOW}示例:${NC} $first_prompt_name (${first_prompt_audio_count} 个音频)"
        fi
        
        TOTAL_AUDIO_FILES=$((TOTAL_AUDIO_FILES + audio_count))
        TOTAL_AUDIO_DIRS=$((TOTAL_AUDIO_DIRS + prompt_dirs))
    else
        echo -e "  ${RED}✗${NC} 音频目录不存在: $audio_zero_shot_dir"
    fi
    
    TOTAL_JSON_ENTRIES=$((TOTAL_JSON_ENTRIES + json_entries))
    
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  汇总统计${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "JSON 文件数: ${GREEN}${#JSON_FILES[@]}${NC}"
echo -e "总 Prompt IDs: ${GREEN}$TOTAL_JSON_ENTRIES${NC}"
echo -e "总 Prompt 音频目录: ${GREEN}$TOTAL_AUDIO_DIRS${NC}"
echo -e "总音频文件数: ${GREEN}$TOTAL_AUDIO_FILES${NC}"
echo ""

# 检查 JSON 和音频的匹配情况
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  匹配性检查${NC}"
echo -e "${BLUE}========================================${NC}"

MISMATCH_COUNT=0

for json_file in "${JSON_FILES[@]}"; do
    basename=$(basename "$json_file" .json)
    part_num=$(echo "$basename" | grep -oP 'part\K\d+')
    audio_zero_shot_dir="$DATA_DIR/${basename}/zero_shot"
    
    if [ ! -d "$audio_zero_shot_dir" ]; then
        echo -e "${RED}✗ Part $part_num: 缺少音频目录${NC}"
        ((MISMATCH_COUNT++))
        continue
    fi
    
    # 随机抽查几个 prompt_id
    python3 - "$json_file" "$audio_zero_shot_dir" "$part_num" << 'PY'
import sys, json, os, random

json_file = sys.argv[1]
audio_dir = sys.argv[2]
part_num = sys.argv[3]

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机选择最多 3 个 prompt_id 进行检查
prompt_ids = list(data.keys())
sample_size = min(3, len(prompt_ids))
sampled_ids = random.sample(prompt_ids, sample_size) if sample_size > 0 else []

all_match = True

for prompt_id in sampled_ids:
    prompt_audio_dir = os.path.join(audio_dir, prompt_id)
    
    if not os.path.isdir(prompt_audio_dir):
        print(f"\033[0;31m✗ Part {part_num}: Prompt {prompt_id} 的音频目录不存在\033[0m")
        all_match = False
        continue
    
    # 检查前 2 个条目的音频文件
    entries = data[prompt_id][:2]
    for entry in entries:
        parts = entry.split('\t', 1)
        if len(parts) != 2:
            continue
        voiceprint_id = parts[0].strip()
        audio_file = os.path.join(prompt_audio_dir, f"{voiceprint_id}.wav")
        
        if not os.path.isfile(audio_file):
            print(f"\033[0;31m✗ Part {part_num}: 音频文件不存在: {voiceprint_id}.wav (prompt: {prompt_id})\033[0m")
            all_match = False

if all_match:
    print(f"\033[0;32m✓ Part {part_num}: 抽查通过 (检查了 {sample_size} 个 prompt)\033[0m")
else:
    sys.exit(1)
PY
    
    if [ $? -ne 0 ]; then
        ((MISMATCH_COUNT++))
    fi
done

echo ""

if [ $MISMATCH_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过，JSON 与音频文件匹配良好${NC}"
else
    echo -e "${YELLOW}⚠ 发现 $MISMATCH_COUNT 个不匹配项${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}检查完成${NC}"
echo -e "${BLUE}========================================${NC}"

