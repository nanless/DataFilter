#!/bin/bash

# 基于SenseVoice Small ASR和NeMo文本标准化的TTS音频筛选脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
SENSEVOICE_MODEL_DIR="iic/SenseVoiceSmall"
CER_THRESHOLD=0.05
NUM_GPUS=8
LANGUAGE="auto"
TEST_MODE=false
VERBOSE=false
SKIP_EXISTING=true
FORCE_REPROCESS=false
DEBUG_MODE=false
DEBUG_SAMPLES=1000

# 函数：显示使用帮助
show_help() {
    echo "用法: $0 <base_dir> <json_file> [选项]"
    echo ""
    echo "参数:"
    echo "  base_dir        音频文件基础目录"
    echo "  json_file       包含groundtruth的JSON文件"
    echo ""
    echo "选项:"
    echo "  --output        输出结果文件路径"
    echo "  --cer_threshold CER阈值 (默认: $CER_THRESHOLD)"
    echo "  --num_gpus      使用的GPU数量 (默认: $NUM_GPUS)"
    echo "  --language      文本语言: auto/zh/en (默认: $LANGUAGE)"
    echo "  --sensevoice_model_dir SenseVoice模型路径或ID (默认: $SENSEVOICE_MODEL_DIR)"
    echo "  --test_mode     测试模式，只处理前10个prompt"
    echo "  --verbose       输出详细的处理日志"
    echo "  --skip_existing 增量处理模式，跳过已处理的音频 (默认: 开启)"
    echo "  --no-skip_existing 不使用增量处理，重新处理所有音频"
    echo "  --force         强制重新处理所有音频（等同于--no-skip_existing）"
    echo "  --debug_mode    调试模式：强制使用8卡并限制样本数量"
    echo "  --debug_samples 调试模式下的样本数上限 (默认: $DEBUG_SAMPLES)"
    echo "  -h, --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/audio/dir /path/to/groundtruth.json"
    echo "  $0 /path/to/audio/dir /path/to/groundtruth.json --cer_threshold 0.1 --num_gpus 4"
}

# 检查参数数量
if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

# 必需参数
BASE_DIR="$1"
JSON_FILE="$2"
shift 2

# 解析可选参数
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
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
        --sensevoice_model_dir)
            SENSEVOICE_MODEL_DIR="$2"
            shift 2
            ;;
        --test_mode)
            TEST_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
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
        --debug_mode)
            DEBUG_MODE=true
            shift
            ;;
        --debug_samples)
            DEBUG_SAMPLES="$2"
            shift 2
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

# 检查输入文件和目录
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}错误: 基础目录不存在: $BASE_DIR${NC}"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo -e "${RED}错误: JSON文件不存在: $JSON_FILE${NC}"
    exit 1
fi

# 检查zero_shot目录
ZERO_SHOT_DIR="$BASE_DIR/zero_shot"
if [ ! -d "$ZERO_SHOT_DIR" ]; then
    echo -e "${RED}错误: zero_shot目录不存在: $ZERO_SHOT_DIR${NC}"
    exit 1
fi

# 激活conda环境
echo -e "${YELLOW}检查并激活Kimi-Audio环境...${NC}"
if [[ "$CONDA_DEFAULT_ENV" != "kimi-audio" ]]; then
    source /root/miniforge3/etc/profile.d/conda.sh
    conda activate kimi-audio
    echo -e "${GREEN}已激活kimi-audio环境${NC}"
else
    echo -e "${GREEN}已在kimi-audio环境中${NC}"
fi

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，无法检测GPU${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -lt 1 ]; then
    echo -e "${RED}错误: 至少需要1张GPU${NC}"
    exit 1
fi

# 调整GPU数量
# 调试模式：强制使用8卡（若可用）
if [ "$DEBUG_MODE" = true ]; then
    echo -e "${YELLOW}调试模式：强制请求使用8卡${NC}"
    NUM_GPUS=8
fi
if [ "$NUM_GPUS" -gt "$GPU_COUNT" ]; then
    echo -e "${YELLOW}警告: 请求的GPU数量($NUM_GPUS)超过可用数量($GPU_COUNT)，将使用所有可用GPU${NC}"
    NUM_GPUS=$GPU_COUNT
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    TTS音频筛选 (SenseVoice + NeMo TN)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}基础目录: $BASE_DIR${NC}"
echo -e "${YELLOW}JSON文件: $JSON_FILE${NC}"
echo -e "${YELLOW}CER阈值: $CER_THRESHOLD${NC}"
echo -e "${YELLOW}使用GPU数: $NUM_GPUS${NC}"
echo -e "${YELLOW}语言设置: $LANGUAGE${NC}"
echo -e "${YELLOW}SenseVoice模型: $SENSEVOICE_MODEL_DIR${NC}"
echo -e "${YELLOW}文本标准化: NeMo (英文) / 简单标准化 (中文)${NC}"

# 检查funasr是否安装
if ! python3 -c "import funasr" 2>/dev/null; then
    echo -e "${RED}错误: funasr库未安装${NC}"
    echo -e "${YELLOW}请安装: pip install funasr${NC}"
    exit 1
fi

# 检查NeMo文本标准化是否安装
if ! python3 -c "import nemo_text_processing" 2>/dev/null; then
    echo -e "${YELLOW}警告: NeMo文本标准化未安装，英文将使用简单文本标准化${NC}"
    echo -e "${YELLOW}建议安装: pip install nemo_text_processing${NC}"
fi

# 构建Python命令
PYTHON_CMD="python3 tts_filter_by_sensevoice_asr.py"
PYTHON_CMD="$PYTHON_CMD \"$BASE_DIR\" \"$JSON_FILE\""
PYTHON_CMD="$PYTHON_CMD --cer_threshold $CER_THRESHOLD"
PYTHON_CMD="$PYTHON_CMD --num_gpus $NUM_GPUS"
PYTHON_CMD="$PYTHON_CMD --sensevoice_model_dir \"$SENSEVOICE_MODEL_DIR\""
PYTHON_CMD="$PYTHON_CMD --language $LANGUAGE"

if [ ! -z "$OUTPUT_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --output \"$OUTPUT_FILE\""
fi

if [ "$TEST_MODE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --test_mode"
fi

# 调试模式参数
if [ "$DEBUG_MODE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --debug_mode --debug_samples $DEBUG_SAMPLES"
fi

if [ "$VERBOSE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

# 添加skip_existing参数
if [ "$SKIP_EXISTING" = false ]; then
    PYTHON_CMD="$PYTHON_CMD --no_skip_existing"
fi

# 检查结果文件是否存在，给出提示
if [ ! -z "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
    if [ "$SKIP_EXISTING" = true ]; then
        echo -e "${CYAN}⊙ 检测到已有结果，将进入增量处理模式${NC}"
        echo -e "${CYAN}  文件: $OUTPUT_FILE${NC}"
        
        # 显示已存在文件的简要统计
        total_files=$(grep -o '"total_files": [0-9]*' "$OUTPUT_FILE" | awk '{print $2}' | head -1)
        filtered_files=$(grep -o '"filtered_files": [0-9]*' "$OUTPUT_FILE" | awk '{print $2}' | head -1)
        
        if [ ! -z "$total_files" ] && [ ! -z "$filtered_files" ]; then
            filter_rate=$(awk "BEGIN {printf \"%.1f\", $filtered_files/$total_files*100}")
            echo -e "${CYAN}  已有结果: 总文件数=$total_files, 被筛选=$filtered_files ($filter_rate%)${NC}"
            echo -e "${CYAN}  将只处理新增的音频文件${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 检测到已有结果，但使用强制重新处理模式${NC}"
        echo -e "${YELLOW}  将覆盖文件: $OUTPUT_FILE${NC}"
    fi
fi

# 执行筛选
echo -e "${GREEN}开始执行TTS音频筛选...${NC}"
echo -e "${YELLOW}执行命令: $PYTHON_CMD${NC}"

eval $PYTHON_CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TTS音频筛选完成${NC}"
else
    echo -e "${RED}✗ TTS音频筛选失败${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}    处理完成${NC}"
echo -e "${BLUE}========================================${NC}"

