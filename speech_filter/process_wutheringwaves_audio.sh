#!/bin/bash

# 鸣潮音频处理脚本 - 多语言多GPU版本
# 处理 wutheringwaves_2.2 目录下的三个语言目录

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_INPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/wutheringwaves_2.2"
BASE_OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/wutheringwaves_2.2_filtered"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 语言目录配置
declare -A LANGUAGE_DIRS=(
    ["chinese"]="中文 - Chinese"
    ["japanese"]="日语 - Japanese"  
    ["english"]="英语 - English"
)

declare -A LANGUAGE_PRESETS=(
    ["chinese"]="chinese"
    ["japanese"]="japanese"
    ["english"]="english"
)

# 默认参数
DEFAULT_NUM_GPUS=8
DEFAULT_WORKERS=8
DEFAULT_BATCH_SIZE=16

# 函数：打印带颜色的信息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_highlight() {
    echo -e "${PURPLE}[重要]${NC} $1"
}

print_processing() {
    echo -e "${CYAN}[处理]${NC} $1"
}

# 函数：显示横幅
show_banner() {
    echo -e "${PURPLE}"
    echo "========================================================================"
    echo "                     鸣潮音频处理脚本 - 多语言多GPU版本"
    echo "========================================================================"
    echo -e "${NC}"
    echo "🎯 处理目标: WutheringWaves 2.2 三语言音频数据"
    echo "🔧 处理方式: 多GPU并行处理"
    echo "📊 结果保存: 每条音频的详细结果 + 最终汇总"
    echo "🌐 支持语言: 中文、日语、英语"
    echo "========================================================================"
    echo ""
}

# 函数：检查目录是否存在
check_directory() {
    local dir="$1"
    local description="$2"
    
    if [ ! -d "$dir" ]; then
        print_error "$description 不存在: $dir"
        return 1
    fi
    
    print_success "$description 检查通过: $dir"
    return 0
}

# 函数：创建目录
create_directory() {
    local dir="$1"
    local description="$2"
    
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        if [ $? -eq 0 ]; then
            print_success "创建 $description: $dir"
        else
            print_error "创建 $description 失败: $dir"
            return 1
        fi
    else
        print_info "$description 已存在: $dir"
    fi
    return 0
}

# 函数：检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    # 检查Python是否可用
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到"
        return 1
    fi
    
    # 检查必要的Python包
    python3 -c "import torch, whisper, librosa, yaml, ten_vad" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Python依赖包缺失，请运行: pip install -r requirements.txt"
        return 1
    fi
    
    # 检查CUDA
    if python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        print_success "CUDA环境检查通过，检测到 ${gpu_count} 个GPU"
    else
        print_warning "CUDA不可用，将使用CPU模式"
    fi
    
    return 0
}

# 函数：统计音频文件数量
count_audio_files() {
    local dir="$1"
    local count=0
    
    for ext in wav mp3 flac m4a; do
        count=$((count + $(find "$dir" -name "*.${ext}" -type f 2>/dev/null | wc -l)))
    done
    
    echo $count
}

# 函数：处理单个语言目录
process_language_directory() {
    local language="$1"
    local input_dir="$2"
    local output_dir="$3"
    local num_gpus="$4"
    local timestamp="$5"
    
    print_highlight "开始处理 ${language} 音频..."
    
    # 统计输入文件数量
    local input_count=$(count_audio_files "$input_dir")
    print_info "发现 $input_count 个音频文件"
    
    if [ $input_count -eq 0 ]; then
        print_warning "输入目录中没有找到音频文件"
        return 1
    fi
    
    # 创建输出目录
    create_directory "$output_dir" "${language}输出目录"
    create_directory "${output_dir}/audio" "${language}音频输出目录"
    create_directory "${output_dir}/results" "${language}结果目录"
    
    # 设置日志和结果文件
    local results_file="${output_dir}/results/${language}_results_${timestamp}.json"
    local stats_file="${output_dir}/results/${language}_stats_${timestamp}.json"
    
    # 构建多GPU处理命令
    local cmd="python3 ${SCRIPT_DIR}/main_multi_gpu.py \"$input_dir\" \
        --output-dir \"${output_dir}/audio\" \
        --config \"$CONFIG_FILE\" \
        --language-preset \"${LANGUAGE_PRESETS[$language]}\" \
        --num-gpus $num_gpus \
        --results-file \"${language}_results_${timestamp}.json\" \
        --log-file \"${language}_processing_${timestamp}.log\" \
        --export-transcriptions \
        --export-quality-report \
        --generate-html-report \
        --detailed-results"
    
    # 添加跳过处理选项
    if [ "$skip_processed" = true ]; then
        cmd="$cmd --skip-processed"
    elif [ "$force_reprocess" = true ]; then
        cmd="$cmd --force-reprocess"
    fi
    
    print_processing "执行命令: $cmd"
    print_info "处理日志: $log_file"
    print_info "处理结果: $results_file"
    
    # 执行处理
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "${language} 音频处理完成"
        
        # 统计输出文件数量
        local output_count=$(count_audio_files "${output_dir}/audio")
        print_success "输出 $output_count 个筛选后的音频文件"
        
        # 计算通过率
        if [ $input_count -gt 0 ]; then
            local pass_rate=$((output_count * 100 / input_count))
            print_success "通过率: ${pass_rate}%"
        fi
        
        # 保存统计信息
        cat > "$stats_file" << EOF
{
    "language": "$language",
    "input_count": $input_count,
    "output_count": $output_count,
    "pass_rate": $pass_rate,
    "processing_time": "$(date -Iseconds)",
    "exit_code": $exit_code
}
EOF
        
        return 0
    else
        print_error "${language} 音频处理失败，退出码: $exit_code"
        return $exit_code
    fi
}

# 函数：汇总所有结果
summarize_results() {
    local timestamp="$1"
    local summary_file="${BASE_OUTPUT_DIR}/summary_${timestamp}.json"
    local summary_html="${BASE_OUTPUT_DIR}/summary_${timestamp}.html"
    
    print_highlight "汇总处理结果..."
    
    # 创建汇总脚本
    cat > "${BASE_OUTPUT_DIR}/summarize.py" << 'EOF'
import json
import os
from pathlib import Path
from datetime import datetime

def summarize_results(base_dir, timestamp):
    """汇总所有语言的处理结果"""
    results_summary = {
        "timestamp": timestamp,
        "processing_date": datetime.now().isoformat(),
        "languages": {},
        "totals": {
            "total_input": 0,
            "total_output": 0,
            "total_pass_rate": 0,
            "successful_languages": 0,
            "failed_languages": 0
        }
    }
    
    languages = ["chinese", "japanese", "english"]
    
    for language in languages:
        stats_file = Path(base_dir) / language / "results" / f"{language}_stats_{timestamp}.json"
        results_file = Path(base_dir) / language / "results" / f"{language}_results_{timestamp}.json"
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                results_summary["languages"][language] = stats
                results_summary["totals"]["total_input"] += stats["input_count"]
                results_summary["totals"]["total_output"] += stats["output_count"]
                
                if stats["exit_code"] == 0:
                    results_summary["totals"]["successful_languages"] += 1
                else:
                    results_summary["totals"]["failed_languages"] += 1
                    
            except Exception as e:
                print(f"读取 {language} 统计失败: {e}")
                results_summary["languages"][language] = {"error": str(e)}
        else:
            print(f"未找到 {language} 统计文件: {stats_file}")
            results_summary["languages"][language] = {"error": "stats file not found"}
    
    # 计算总体通过率
    if results_summary["totals"]["total_input"] > 0:
        results_summary["totals"]["total_pass_rate"] = (
            results_summary["totals"]["total_output"] / 
            results_summary["totals"]["total_input"] * 100
        )
    
    return results_summary

def generate_html_report(summary, output_file):
    """生成HTML格式的汇总报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WutheringWaves 2.2 音频处理汇总报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .language-section {{ margin-bottom: 30px; }}
        .language-header {{ background: #e9ecef; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.2em; }}
        .language-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; }}
        .language-stat {{ background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ WutheringWaves 2.2 音频处理汇总报告</h1>
            <p>处理时间: {summary['processing_date']}</p>
        </div>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{summary['totals']['total_input']}</div>
                <div class="stat-label">总输入文件数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['totals']['total_output']}</div>
                <div class="stat-label">总输出文件数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['totals']['total_pass_rate']:.1f}%</div>
                <div class="stat-label">总体通过率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['totals']['successful_languages']}</div>
                <div class="stat-label">成功处理语言</div>
            </div>
        </div>
"""
    
    # 添加各语言详细信息
    for language, stats in summary['languages'].items():
        status_class = "success" if stats.get('exit_code') == 0 else "error"
        status_text = "✅ 成功" if stats.get('exit_code') == 0 else "❌ 失败"
        
        html_content += f"""
        <div class="language-section">
            <div class="language-header">
                {language.title()} 音频处理结果 <span class="{status_class}">{status_text}</span>
            </div>
            <div class="language-stats">
                <div class="language-stat">
                    <strong>{stats.get('input_count', 'N/A')}</strong><br>
                    <small>输入文件数</small>
                </div>
                <div class="language-stat">
                    <strong>{stats.get('output_count', 'N/A')}</strong><br>
                    <small>输出文件数</small>
                </div>
                <div class="language-stat">
                    <strong>{stats.get('pass_rate', 'N/A')}%</strong><br>
                    <small>通过率</small>
                </div>
                <div class="language-stat">
                    <strong>{stats.get('processing_time', 'N/A')}</strong><br>
                    <small>处理时间</small>
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <div class="footer">
            <p>报告生成时间: {}</p>
            <p>⚡ 由 WutheringWaves 音频处理系统生成</p>
        </div>
    </div>
</body>
</html>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python summarize.py <base_dir> <timestamp>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    timestamp = sys.argv[2]
    
    # 生成汇总
    summary = summarize_results(base_dir, timestamp)
    
    # 保存JSON汇总
    summary_file = f"{base_dir}/summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 生成HTML报告
    html_file = f"{base_dir}/summary_{timestamp}.html"
    generate_html_report(summary, html_file)
    
    print(f"汇总完成!")
    print(f"JSON文件: {summary_file}")
    print(f"HTML报告: {html_file}")
    
    # 打印简要统计
    print("\n" + "="*60)
    print("                    处理汇总")
    print("="*60)
    print(f"总输入文件数: {summary['totals']['total_input']}")
    print(f"总输出文件数: {summary['totals']['total_output']}")
    print(f"总体通过率: {summary['totals']['total_pass_rate']:.1f}%")
    print(f"成功处理语言: {summary['totals']['successful_languages']}")
    print(f"失败处理语言: {summary['totals']['failed_languages']}")
    print("="*60)
EOF

    # 运行汇总脚本
    python3 "${BASE_OUTPUT_DIR}/summarize.py" "$BASE_OUTPUT_DIR" "$timestamp"
    
    # 清理临时脚本
    rm "${BASE_OUTPUT_DIR}/summarize.py"
    
    print_success "结果汇总完成"
    print_info "汇总文件: $summary_file"
    print_info "HTML报告: $summary_html"
}

# 函数：显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -g, --gpus NUM      设置GPU数量 (默认: 4)"
    echo "  -l, --language LANG 只处理指定语言 (chinese/japanese/english)"
    echo "  -c, --check-only    仅检查环境，不执行处理"
    echo "  -v, --verbose       显示详细信息"
    echo "  --dry-run           预览将要执行的命令，不实际执行"
    echo "  --skip-processed    跳过已处理的文件"
    echo "  --force-reprocess   强制重新处理所有文件"
    echo ""
    echo "说明:"
    echo "  这个脚本用于批量处理WutheringWaves 2.2的三语言音频文件。"
    echo "  支持多GPU并行处理，每条音频都会保存详细的处理结果。"
    echo ""
    echo "输入目录: $BASE_INPUT_DIR"
    echo "输出目录: $BASE_OUTPUT_DIR"
    echo ""
    echo "示例:"
    echo "  $0                           # 处理所有语言，使用默认参数"
    echo "  $0 -g 2                     # 使用2个GPU"
    echo "  $0 -l japanese              # 只处理日语音频"
    echo "  $0 --check-only             # 仅检查环境"
    echo "  $0 --skip-processed         # 跳过已处理的文件"
    echo "  $0 --force-reprocess        # 强制重新处理所有文件"
    echo ""
}

# 主函数
main() {
    local num_gpus=$DEFAULT_NUM_GPUS
    local specific_language=""
    local check_only=false
    local verbose=false
    local dry_run=false
    local skip_processed=true
    local force_reprocess=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -g|--gpus)
                num_gpus="$2"
                shift 2
                ;;
            -l|--language)
                specific_language="$2"
                shift 2
                ;;
            -c|--check-only)
                check_only=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --skip-processed)
                skip_processed=true
                shift
                ;;
            --force-reprocess)
                force_reprocess=true
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示横幅
    show_banner
    
    # 显示配置信息
    print_info "========== 处理配置 =========="
    print_info "输入目录: $BASE_INPUT_DIR"
    print_info "输出目录: $BASE_OUTPUT_DIR"
    print_info "配置文件: $CONFIG_FILE"
    print_info "GPU数量: $num_gpus"
    if [ -n "$specific_language" ]; then
        print_info "指定语言: $specific_language"
    else
        print_info "处理语言: 全部 (中文、日语、英语)"
    fi
    if [ "$skip_processed" = true ]; then
        print_info "跳过已处理文件: 是"
    elif [ "$force_reprocess" = true ]; then
        print_info "强制重新处理: 是"
    else
        print_info "跳过已处理文件: 否"
    fi
    print_info "============================="
    echo ""
    
    # 检查输入目录
    if ! check_directory "$BASE_INPUT_DIR" "输入目录"; then
        exit 1
    fi
    
    # 检查各语言子目录
    for language in "${!LANGUAGE_DIRS[@]}"; do
        if [ -n "$specific_language" ] && [ "$specific_language" != "$language" ]; then
            continue
        fi
        
        local lang_dir="$BASE_INPUT_DIR/${LANGUAGE_DIRS[$language]}"
        if ! check_directory "$lang_dir" "${language}目录"; then
            print_warning "跳过 $language 语言处理"
            continue
        fi
    done
    
    # 创建输出和日志目录
    if ! create_directory "$BASE_OUTPUT_DIR" "输出目录"; then
        exit 1
    fi
    
    # 检查Python环境
    if ! check_python_env; then
        exit 1
    fi
    
    # 检查配置文件
    if ! check_directory "$(dirname "$CONFIG_FILE")" "配置文件目录"; then
        print_warning "配置文件不存在: $CONFIG_FILE"
        print_warning "将使用默认配置"
    fi
    
    # 检查多GPU脚本
    if [ ! -f "${SCRIPT_DIR}/main_multi_gpu.py" ]; then
        print_error "多GPU脚本不存在: ${SCRIPT_DIR}/main_multi_gpu.py"
        exit 1
    fi
    
    # 如果只是检查，则退出
    if [ "$check_only" = true ]; then
        print_success "环境检查完成"
        exit 0
    fi
    
    # 如果是预览模式，显示命令但不执行
    if [ "$dry_run" = true ]; then
        print_info "预览模式 - 将要执行的命令："
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        
        for language in "${!LANGUAGE_DIRS[@]}"; do
            if [ -n "$specific_language" ] && [ "$specific_language" != "$language" ]; then
                continue
            fi
            
            local input_dir="$BASE_INPUT_DIR/${LANGUAGE_DIRS[$language]}"
            local output_dir="$BASE_OUTPUT_DIR/$language"
            
            echo ""
            echo "# 处理 $language 语言"
            echo "python3 ${SCRIPT_DIR}/main_multi_gpu.py \"$input_dir\" \\"
            echo "    --output-dir \"${output_dir}/audio\" \\"
            echo "    --config \"$CONFIG_FILE\" \\"
            echo "    --language-preset \"${LANGUAGE_PRESETS[$language]}\" \\"
            echo "    --num-gpus $num_gpus \\"
            echo "    --results-file \"${language}_results_${timestamp}.json\" \\"
            echo "    --log-file \"${language}_processing_${timestamp}.log\" \\"
            echo "    --export-transcriptions \\"
            echo "    --export-quality-report \\"
            echo "    --generate-html-report \\"
            echo "    --detailed-results"
            
            # 添加跳过处理选项
            if [ "$skip_processed" = true ]; then
                echo "    --skip-processed"
            elif [ "$force_reprocess" = true ]; then
                echo "    --force-reprocess"
            fi
        done
        
        echo ""
        echo "# 汇总结果"
        echo "python3 汇总脚本..."
        exit 0
    fi
    
    # 开始处理
    print_highlight "开始批量处理WutheringWaves 2.2音频..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local total_success=0
    local total_failed=0
    
    # 处理各语言目录
    for language in "${!LANGUAGE_DIRS[@]}"; do
        if [ -n "$specific_language" ] && [ "$specific_language" != "$language" ]; then
            continue
        fi
        
        local input_dir="$BASE_INPUT_DIR/${LANGUAGE_DIRS[$language]}"
        local output_dir="$BASE_OUTPUT_DIR/$language"
        
        if [ ! -d "$input_dir" ]; then
            print_warning "跳过不存在的目录: $input_dir"
            continue
        fi
        
        echo ""
        print_processing "处理 $language 语言 (${LANGUAGE_DIRS[$language]})"
        
        if process_language_directory "$language" "$input_dir" "$output_dir" "$num_gpus" "$timestamp"; then
            total_success=$((total_success + 1))
            print_success "$language 语言处理完成"
        else
            total_failed=$((total_failed + 1))
            print_error "$language 语言处理失败"
        fi
    done
    
    # 汇总结果
    echo ""
    summarize_results "$timestamp"
    
    # 显示最终结果
    echo ""
    print_highlight "批量处理完成！"
    print_info "成功处理: $total_success 个语言"
    print_info "失败处理: $total_failed 个语言"
    print_info "详细结果: $BASE_OUTPUT_DIR/summary_${timestamp}.html"
    
    # 返回退出码
    if [ $total_failed -eq 0 ]; then
        print_success "所有语言处理成功！"
        exit 0
    else
        print_warning "部分语言处理失败，请检查日志"
        exit 1
    fi
}

# 运行主函数
main "$@" 