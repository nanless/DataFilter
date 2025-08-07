#!/bin/bash

# 批量处理所有TTS音频的筛选脚本
# 遍历/root/group-shared/voiceprint/share/下的所有voiceprint_*.json文件

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置参数
SHARE_DIR="/root/group-shared/voiceprint/share"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
CER_THRESHOLD=0.2
NUM_GPUS=8
LANGUAGE="en"
USE_WHISPER=true
WHISPER_MODEL_SIZE="large-v3"
SKIP_EXISTING=true
FORCE_REPROCESS=false

# 创建必要的目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# 函数：显示使用帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --share_dir     voiceprint share目录 (默认: $SHARE_DIR)"
    echo "  --results_dir   结果输出目录 (默认: $RESULTS_DIR)"
    echo "  --cer_threshold CER阈值 (默认: $CER_THRESHOLD)"
    echo "  --num_gpus      每个任务使用的GPU数量 (默认: $NUM_GPUS)"
    echo "  --language      文本语言: auto/zh/en (默认: $LANGUAGE)"
    echo "  --use_whisper   使用Whisper+LLM模式（默认已开启）"
    echo "  --no-use_whisper 禁用Whisper模式，使用Kimi-Audio模式"
    echo "  --whisper_model Whisper模型大小 (默认: $WHISPER_MODEL_SIZE)"
    echo "  --pattern       JSON文件名模式 (默认: voiceprint_*_part*_*.json)"
    echo "  --skip_existing 增量处理模式，跳过已处理的音频 (默认: 开启)"
    echo "  --no-skip_existing 不使用增量处理，重新处理所有音频"
    echo "  --force         强制重新处理所有音频（等同于--no-skip_existing）"
    echo "  -h, --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 --cer_threshold 0.1 --num_gpus 4"
    echo "  $0 --pattern 'voiceprint_20250804_part*_*.json'"
    echo "  $0 --force  # 强制重新处理所有文件"
    echo "  $0 --no-skip_existing  # 不跳过已存在的结果"
}

# 解析命令行参数
PATTERN="voiceprint_*_part*_*.json"
while [[ $# -gt 0 ]]; do
    case $1 in
        --share_dir)
            SHARE_DIR="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --cer_threshold)
            CER_THRESHOLD="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --use_whisper)
            USE_WHISPER=true
            shift
            ;;
        --no-use_whisper)
            USE_WHISPER=false
            shift
            ;;
        --whisper_model)
            WHISPER_MODEL_SIZE="$2"
            shift 2
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --no-skip_existing)
            SKIP_EXISTING=false
            shift
            ;;
        --force)
            SKIP_EXISTING=false
            FORCE_REPROCESS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查share目录
if [ ! -d "$SHARE_DIR" ]; then
    echo -e "${RED}错误: share目录不存在: $SHARE_DIR${NC}"
    exit 1
fi

# 查找所有符合模式的JSON文件
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    批量TTS音频筛选${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Share目录: $SHARE_DIR${NC}"
echo -e "${YELLOW}结果目录: $RESULTS_DIR${NC}"
echo -e "${YELLOW}文件模式: $PATTERN${NC}"
echo -e "${YELLOW}CER阈值: $CER_THRESHOLD${NC}"
echo -e "${YELLOW}GPU数/任务: $NUM_GPUS${NC}"
echo -e "${YELLOW}语言设置: $LANGUAGE${NC}"
if [ "$USE_WHISPER" = true ]; then
    echo -e "${YELLOW}ASR模式: Whisper ($WHISPER_MODEL_SIZE) + LLM [默认]${NC}"
else
    echo -e "${YELLOW}ASR模式: Kimi-Audio${NC}"
fi
if [ "$SKIP_EXISTING" = true ]; then
    echo -e "${YELLOW}处理模式: 增量处理（跳过已处理的音频）${NC}"
else
    echo -e "${YELLOW}处理模式: 全量处理（重新处理所有音频）${NC}"
fi
echo ""

# 切换到share目录
cd "$SHARE_DIR"

# 查找JSON文件
JSON_FILES=($(ls $PATTERN 2>/dev/null | sort))

if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 未找到符合模式的JSON文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 ${#JSON_FILES[@]} 个JSON文件需要处理:${NC}"
for json_file in "${JSON_FILES[@]}"; do
    echo "  - $json_file"
done
echo ""

# 统计信息
TOTAL_FILES=${#JSON_FILES[@]}
PROCESSED_FILES=0
SUCCESS_FILES=0
FAILED_FILES=0
SKIPPED_FILES=0
START_TIME=$(date +%s)

# 处理每个JSON文件
for json_file in "${JSON_FILES[@]}"; do
    PROCESSED_FILES=$((PROCESSED_FILES + 1))
    
    echo -e "${CYAN}[$PROCESSED_FILES/$TOTAL_FILES] 处理: $json_file${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    # 获取基础名称（去掉.json后缀）
    base_name="${json_file%.json}"
    
    # 构建对应的目录路径
    base_dir="$SHARE_DIR/$base_name"
    json_path="$SHARE_DIR/$json_file"
    
    # 检查目录是否存在
    if [ ! -d "$base_dir" ]; then
        echo -e "${RED}✗ 错误: 对应目录不存在: $base_dir${NC}"
        FAILED_FILES=$((FAILED_FILES + 1))
        continue
    fi
    
    # 检查zero_shot目录
    if [ ! -d "$base_dir/zero_shot" ]; then
        echo -e "${RED}✗ 错误: zero_shot目录不存在: $base_dir/zero_shot${NC}"
        FAILED_FILES=$((FAILED_FILES + 1))
        continue
    fi
    
    # 设置输出文件路径
    output_file="$RESULTS_DIR/tts_filter_results_${base_name}.json"
    log_file="$LOG_DIR/tts_filter_${base_name}.log"
    
    # 检查结果文件是否存在
    if [ -f "$output_file" ]; then
        if [ "$SKIP_EXISTING" = true ]; then
            echo -e "${CYAN}⊙ 检测到已有结果，进入增量处理模式${NC}"
            echo -e "${CYAN}  文件: $output_file${NC}"
            
            # 显示已存在文件的简要统计
            total_files=$(grep -o '"total_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
            filtered_files=$(grep -o '"filtered_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
            
            if [ ! -z "$total_files" ] && [ ! -z "$filtered_files" ]; then
                filter_rate=$(awk "BEGIN {printf \"%.1f\", $filtered_files/$total_files*100}")
                echo -e "${CYAN}  已有结果: 总文件数=$total_files, 被筛选=$filtered_files ($filter_rate%)${NC}"
                echo -e "${CYAN}  将只处理新增的音频文件${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ 检测到已有结果，但使用强制重新处理模式${NC}"
            echo -e "${YELLOW}  将覆盖文件: $output_file${NC}"
        fi
        echo ""
    fi
    
    # 执行筛选
    echo -e "${YELLOW}开始筛选...${NC}"
    
    # 切换到脚本目录
    cd "$SCRIPT_DIR"
    
    # 运行筛选脚本
    CMD="./run_single_tts_filter.sh \"$base_dir\" \"$json_path\""
    CMD="$CMD --output \"$output_file\""
    CMD="$CMD --cer_threshold \"$CER_THRESHOLD\""
    CMD="$CMD --num_gpus \"$NUM_GPUS\""
    CMD="$CMD --language \"$LANGUAGE\""
    
    if [ "$USE_WHISPER" = true ]; then
        CMD="$CMD --use_whisper"
        CMD="$CMD --whisper_model \"$WHISPER_MODEL_SIZE\""
    fi
    
    # 传递skip_existing参数
    if [ "$SKIP_EXISTING" = false ]; then
        CMD="$CMD --no-skip_existing"
    fi
    
    eval "$CMD" > "$log_file" 2>&1
    
    # 检查结果
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 成功处理${NC}"
        SUCCESS_FILES=$((SUCCESS_FILES + 1))
        
        # 显示简要统计（如果输出文件存在）
        if [ -f "$output_file" ]; then
            # 提取统计信息
            total_files=$(grep -o '"total_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
            filtered_files=$(grep -o '"filtered_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
            
            if [ ! -z "$total_files" ] && [ ! -z "$filtered_files" ]; then
                filter_rate=$(awk "BEGIN {printf \"%.1f\", $filtered_files/$total_files*100}")
                echo -e "${YELLOW}  总文件数: $total_files, 被筛选: $filtered_files ($filter_rate%)${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ 处理失败，查看日志: $log_file${NC}"
        FAILED_FILES=$((FAILED_FILES + 1))
    fi
    
    echo ""
    
    # 切换回share目录
    cd "$SHARE_DIR"
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# 显示最终统计
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    批量处理完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}总文件数: $TOTAL_FILES${NC}"
echo -e "${GREEN}成功处理: $SUCCESS_FILES${NC}"
echo -e "${CYAN}跳过文件: $SKIPPED_FILES${NC}"
echo -e "${RED}处理失败: $FAILED_FILES${NC}"
echo -e "${YELLOW}总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒${NC}"
echo ""
echo -e "${CYAN}结果保存在: $RESULTS_DIR${NC}"
echo -e "${CYAN}日志保存在: $LOG_DIR${NC}"

# 生成汇总报告
SUMMARY_FILE="$RESULTS_DIR/batch_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "批量TTS音频筛选汇总报告" > "$SUMMARY_FILE"
echo "========================" >> "$SUMMARY_FILE"
echo "处理时间: $(date)" >> "$SUMMARY_FILE"
echo "总文件数: $TOTAL_FILES" >> "$SUMMARY_FILE"
echo "成功处理: $SUCCESS_FILES" >> "$SUMMARY_FILE"
echo "跳过文件: $SKIPPED_FILES" >> "$SUMMARY_FILE"
echo "处理失败: $FAILED_FILES" >> "$SUMMARY_FILE"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "处理详情:" >> "$SUMMARY_FILE"
echo "----------" >> "$SUMMARY_FILE"

# 添加每个文件的处理结果
for json_file in "${JSON_FILES[@]}"; do
    base_name="${json_file%.json}"
    output_file="$RESULTS_DIR/tts_filter_results_${base_name}.json"
    
    if [ -f "$output_file" ]; then
        total_files=$(grep -o '"total_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
        filtered_files=$(grep -o '"filtered_files": [0-9]*' "$output_file" | awk '{print $2}' | head -1)
        
        if [ ! -z "$total_files" ] && [ ! -z "$filtered_files" ]; then
            filter_rate=$(awk "BEGIN {printf \"%.1f\", $filtered_files/$total_files*100}")
            echo "$json_file: 总数=$total_files, 筛选=$filtered_files ($filter_rate%)" >> "$SUMMARY_FILE"
        else
            echo "$json_file: 已处理（统计信息不可用）" >> "$SUMMARY_FILE"
        fi
    else
        echo "$json_file: 处理失败" >> "$SUMMARY_FILE"
    fi
done

echo -e "${GREEN}汇总报告已保存到: $SUMMARY_FILE${NC}"