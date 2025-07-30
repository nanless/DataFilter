#!/bin/bash

# DataFilter 长音频处理系统启动脚本
# 用于激活conda环境并启动多GPU并行处理

set -e  # 遇到错误立即退出

# 脚本配置
CONDA_ENV="DataFilter"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "DataFilter 长音频处理系统启动脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项] --input INPUT_DIR --output OUTPUT_DIR"
    echo ""
    echo "必需参数:"
    echo "  --input DIR          输入音频文件目录"
    echo "  --output DIR         输出目录"
    echo ""
    echo "可选参数:"
    echo "  --num-gpus N         使用的GPU数量 (默认: 自动检测全部GPU)"
    echo "  --max-concurrent N   最大并发文件数 (默认: 8)"
    echo "  --log-level LEVEL    日志级别 (DEBUG/INFO/WARNING/ERROR, 默认: INFO)"
    echo "  --single-gpu         使用单GPU模式"
    echo "  --help, -h           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 使用全部GPU处理"
    echo "  $0 --input /path/to/input --output /path/to/output"
    echo ""
    echo "  # 使用2张GPU，最大4个并发文件"
    echo "  $0 --input /path/to/input --output /path/to/output --num-gpus 2 --max-concurrent 4"
    echo ""
    echo "  # 单GPU模式"
    echo "  $0 --input /path/to/input --output /path/to/output --single-gpu"
}

# 检查conda环境
check_conda_env() {
    log_info "检查conda环境: $CONDA_ENV"
    
    if ! command -v conda &> /dev/null; then
        log_error "未找到conda命令，请确保已安装conda或miniconda"
        exit 1
    fi
    
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        log_error "未找到conda环境: $CONDA_ENV"
        log_error "请先创建环境: conda create -n $CONDA_ENV python=3.8"
        exit 1
    fi
    
    log_info "✅ conda环境检查通过"
}

# 检查GPU
check_gpu() {
    log_info "检查GPU状态..."
    
    # 激活conda环境并检查GPU
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
    
    if python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        if [ "$GPU_COUNT" -gt 0 ]; then
            log_info "✅ 检测到 $GPU_COUNT 张GPU"
            python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}') for i in range(torch.cuda.device_count())]" 2>/dev/null
        else
            log_warn "⚠️ 未检测到可用GPU，将使用CPU模式"
        fi
    else
        log_warn "⚠️ 无法检测GPU状态，可能缺少PyTorch"
    fi
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    # 创建日志目录
    mkdir -p "$LOG_DIR"
    
    # 创建临时目录
    mkdir -p "${SCRIPT_DIR}/temp"
    mkdir -p "${SCRIPT_DIR}/gpu_work"
    
    log_info "✅ 目录创建完成"
}

# 启动处理
start_processing() {
    local input_dir="$1"
    local output_dir="$2"
    local num_gpus="${3:-auto}"
    local max_concurrent="${4:-8}"
    local log_level="${5:-INFO}"
    local single_gpu="${6:-false}"
    
    log_info "开始长音频处理..."
    log_info "输入目录: $input_dir"
    log_info "输出目录: $output_dir"
    log_info "GPU数量: $num_gpus"
    log_info "最大并发: $max_concurrent"
    log_info "日志级别: $log_level"
    log_info "单GPU模式: $single_gpu"
    
    # 激活conda环境
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
    
    # 切换到脚本目录
    cd "$SCRIPT_DIR"
    
    # 构建命令
    if [ "$single_gpu" = "true" ]; then
        # 单GPU模式
        local cmd="python run_processing.py --input \"$input_dir\" --output \"$output_dir\" --log-level $log_level"
        log_info "使用单GPU模式: $cmd"
    else
        # 多GPU模式
        local cmd="python run_multi_gpu.py --input \"$input_dir\" --output \"$output_dir\" --log-level $log_level"
        
        if [ "$num_gpus" != "auto" ]; then
            cmd="$cmd --num-gpus $num_gpus"
        fi
        
        cmd="$cmd --max-concurrent $max_concurrent"
        log_info "使用多GPU模式: $cmd"
    fi
    
    # 生成时间戳日志文件
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="${LOG_DIR}/processing_${timestamp}.log"
    
    log_info "开始处理，日志文件: $log_file"
    log_info "您可以使用以下命令监控进度:"
    log_info "  tail -f $log_file"
    
    # 执行命令并记录日志
    eval "$cmd" 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_info "🎉 处理完成！"
        log_info "输出目录: $output_dir"
        log_info "日志文件: $log_file"
    else
        log_error "❌ 处理失败，退出码: $exit_code"
        log_error "请检查日志文件: $log_file"
        exit $exit_code
    fi
}

# 解析命令行参数
parse_arguments() {
    local input_dir=""
    local output_dir=""
    local num_gpus="auto"
    local max_concurrent="8"
    local log_level="INFO"
    local single_gpu="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input)
                input_dir="$2"
                shift 2
                ;;
            --output)
                output_dir="$2"
                shift 2
                ;;
            --num-gpus)
                num_gpus="$2"
                shift 2
                ;;
            --max-concurrent)
                max_concurrent="$2"
                shift 2
                ;;
            --log-level)
                log_level="$2"
                shift 2
                ;;
            --single-gpu)
                single_gpu="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证必需参数
    if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
        log_error "缺少必需参数"
        show_help
        exit 1
    fi
    
    # 验证目录
    if [ ! -d "$input_dir" ]; then
        log_error "输入目录不存在: $input_dir"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 启动处理
    start_processing "$input_dir" "$output_dir" "$num_gpus" "$max_concurrent" "$log_level" "$single_gpu"
}

# 主函数
main() {
    log_info "DataFilter 长音频处理系统启动"
    log_info "================================"
    
    # 基础检查
    check_conda_env
    create_directories
    check_gpu
    
    # 解析参数并启动
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    parse_arguments "$@"
}

# 信号处理
cleanup() {
    log_warn "接收到中断信号，正在清理..."
    # 这里可以添加清理逻辑
    exit 130
}

trap cleanup SIGINT SIGTERM

# 执行主函数
main "$@" 