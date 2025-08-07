#!/bin/bash

# 自动启动LLM服务脚本
# 根据GPU数量和模型类型自动配置和启动相应的LLM服务

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_MODEL_NAME="qwen3:32b"
DEFAULT_MODEL_TYPE="qwen3"

# 解析命令行参数
MODEL_NAME="$DEFAULT_MODEL_NAME"
MODEL_TYPE="$DEFAULT_MODEL_TYPE"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model-name    LLM模型名称 (默认: $DEFAULT_MODEL_NAME)"
            echo "  --model-type    模型类型 qwen2.5|qwen3 (默认: $DEFAULT_MODEL_TYPE)"
            echo "  -h, --help      显示此帮助信息"
            echo ""
            echo "GPU配置说明:"
            echo "  独立GPU配置: 每张GPU运行独立的ASR+LLM服务"
            echo "  模型类型影响prompt优化策略 (qwen3减少思考链长度)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    自动启动LLM服务（独立GPU配置）${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}模型配置: ${MODEL_NAME} (类型: ${MODEL_TYPE})${NC}"

# 激活conda环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

# 创建日志目录
mkdir -p ./logs

# 设置代理绕过 - 彻底清除可能影响localhost连接的代理设置
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# 检测GPU数量
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，无法检测GPU${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${YELLOW}检测到 ${GPU_COUNT} 张GPU${NC}"

# 每张GPU独立配置服务
if [ "$GPU_COUNT" -lt 1 ]; then
    echo -e "${RED}至少需要1张GPU${NC}"
    exit 1
fi

echo -e "${GREEN}使用独立GPU配置 (${GPU_COUNT}卡): 每张GPU运行独立的ASR+LLM服务${NC}"

# 为每张GPU启动独立的Ollama服务
echo -e "${YELLOW}启动Ollama服务...${NC}"
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    ollama_port=$((11434 + gpu))
    http_port=$((8000 + gpu))
    
    echo -e "${YELLOW}启动GPU ${gpu} Ollama服务 (端口 ${ollama_port})...${NC}"
    if [ $gpu -eq 0 ]; then
        # 第一个GPU使用默认端口 - 清除代理环境变量
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        CUDA_VISIBLE_DEVICES=$gpu no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" \
        nohup ollama serve > ./logs/ollama_${gpu}.log 2>&1 &
    else
        # 其他GPU使用自定义端口 - 清除代理环境变量
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        CUDA_VISIBLE_DEVICES=$gpu OLLAMA_HOST=0.0.0.0:${ollama_port} no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" \
        nohup ollama serve > ./logs/ollama_${gpu}.log 2>&1 &
    fi
    OLLAMA_PID=$!
    echo $OLLAMA_PID > ./logs/ollama_${gpu}.pid
done

# 等待Ollama服务启动
echo -e "${YELLOW}等待Ollama服务启动...${NC}"
sleep 15

# 为每张GPU拉取模型
echo -e "${YELLOW}拉取模型...${NC}"
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    ollama_port=$((11434 + gpu))
    
    echo -e "${YELLOW}GPU ${gpu} 拉取模型 ${MODEL_NAME}...${NC}"
    if [ $gpu -eq 0 ]; then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        CUDA_VISIBLE_DEVICES=$gpu no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" \
        ollama pull ${MODEL_NAME}
    else
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        CUDA_VISIBLE_DEVICES=$gpu OLLAMA_HOST=localhost:${ollama_port} no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" \
        ollama pull ${MODEL_NAME}
    fi
done

# 为每张GPU启动独立的LLM HTTP服务
echo -e "${YELLOW}启动LLM HTTP服务...${NC}"
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    ollama_port=$((11434 + gpu))
    http_port=$((8000 + gpu))
    
    echo -e "${YELLOW}启动GPU ${gpu} LLM HTTP服务 (端口 ${http_port})...${NC}"
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    CUDA_VISIBLE_DEVICES=$gpu no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1" \
    nohup python3 llm_service.py \
        --model_name ${MODEL_NAME} \
        --ollama_host localhost:${ollama_port} \
        --host 0.0.0.0 \
        --port ${http_port} \
        --workers 1 \
        > ./logs/llm_service_${gpu}.log 2>&1 &
    LLM_PID=$!
    echo $LLM_PID > ./logs/llm_service_${gpu}.pid
done

echo -e "${GREEN}独立GPU配置启动完成 (${GPU_COUNT}张GPU)${NC}"

# 等待服务完全启动
echo -e "${YELLOW}等待服务完全启动...${NC}"
sleep 30  # 增加等待时间到30秒

# 检查服务状态
echo -e "${BLUE}检查服务状态:${NC}"

# 动态检查所有已启动的LLM服务端口
service_count=0
for port in {8000..8010}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}检查LLM服务 (端口 $port)...${NC}"
        # 给每个服务更多时间来响应
        for attempt in {1..10}; do
            if response=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY curl -s --connect-timeout 5 --max-time 10 http://localhost:$port/health 2>/dev/null); then
                if echo "$response" | grep -q "healthy\|OK\|status"; then
                    echo -e "${GREEN}✓ LLM服务 (端口 $port) 正常运行${NC}"
                    service_count=$((service_count + 1))
                    break
                fi
            fi
            if [ $attempt -lt 10 ]; then
                echo -e "${YELLOW}  等待服务就绪... (尝试 $attempt/10)${NC}"
                sleep 3
            fi
        done
        
        # 如果10次尝试都失败了
        if [ $attempt -eq 10 ]; then
            echo -e "${RED}✗ LLM服务 (端口 $port) 连接失败${NC}"
            echo -e "${YELLOW}  查看日志: tail -f ./logs/llm_service_$((port-8000)).log${NC}"
        fi
    fi
done

if [ $service_count -eq 0 ]; then
    echo -e "${RED}✗ 没有LLM服务成功启动${NC}"
    echo -e "${YELLOW}建议检查日志文件:${NC}"
    echo -e "  • Ollama日志: ls -la ./logs/ollama_*.log"
    echo -e "  • LLM服务日志: ls -la ./logs/llm_service_*.log"
else
    echo -e "${GREEN}✓ 成功启动 $service_count 个LLM服务${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}    自动启动完成${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "${YELLOW}使用提示:${NC}"
echo -e "• 查看日志: tail -f ./logs/*.log"

# 显示所有运行中的LLM服务地址
echo -e "• LLM服务地址:"
for port in {8000..8010}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "  - http://localhost:$port"
    fi
done

echo -e "• 停止服务: ./stop_multi_llm_services.sh"
echo -e "• 运行筛选: ./run_single_tts_filter.sh /path/to/data /path/to/json --use_whisper --language ${MODEL_NAME#*:}" 