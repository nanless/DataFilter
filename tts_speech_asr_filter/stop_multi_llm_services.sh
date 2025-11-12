#!/bin/bash

# 多LLM服务停止脚本
# 停止所有LLM服务实例

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="./logs"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    停止多LLM服务${NC}"
echo -e "${BLUE}========================================${NC}"

# 停止LLM监控进程
if [ -f "./logs/llm_monitor.pid" ]; then
    MONITOR_PID=$(cat ./logs/llm_monitor.pid)
    echo -e "${YELLOW}停止LLM监控进程 (PID: $MONITOR_PID)...${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    rm -f ./logs/llm_monitor.pid
    echo -e "${GREEN}✓ LLM监控已停止${NC}"
    echo ""
fi

# 函数：停止指定PID的进程
stop_process() {
    local pid=$1
    local process_name=$2
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}PID为空，跳过停止${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}停止${process_name}进程 (PID: $pid)...${NC}"
    
    # 检查进程是否存在
    if ! kill -0 $pid 2>/dev/null; then
        echo -e "${YELLOW}进程 $pid 不存在，可能已经停止${NC}"
        return 0
    fi
    
    # 优雅关闭
    echo -e "${YELLOW}发送SIGTERM信号...${NC}"
    kill $pid
    
    # 等待进程结束
    for i in {1..10}; do
        if ! kill -0 $pid 2>/dev/null; then
            echo -e "${GREEN}✓ ${process_name}已停止${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    # 如果进程仍然存在，强制终止
    echo -e "${RED}进程未响应，强制终止...${NC}"
    kill -9 $pid
    
    # 再次检查
    sleep 2
    if ! kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}✓ ${process_name}已强制停止${NC}"
        return 0
    else
        echo -e "${RED}无法停止${process_name}进程${NC}"
        return 1
    fi
}

# 停止所有LLM服务
echo -e "${YELLOW}1. 停止所有LLM服务...${NC}"

# 查找所有LLM服务PID文件
llm_pids_found=false
for pid_file in "$LOG_DIR"/llm_service_*.pid; do
    if [ -f "$pid_file" ]; then
        llm_pids_found=true
        port=$(basename "$pid_file" .pid | sed 's/llm_service_//')
        SERVICE_PID=$(cat "$pid_file")
        
        echo -e "${YELLOW}发现LLM服务: 端口${port}, PID${SERVICE_PID}${NC}"
        
        if stop_process "$SERVICE_PID" "LLM服务(端口${port})"; then
            rm -f "$pid_file"
        fi
    fi
done

# 如果没有找到PID文件，尝试查找运行中的进程
if ! $llm_pids_found; then
    echo -e "${YELLOW}未找到LLM服务PID文件，搜索运行中的进程...${NC}"
    
    # 查找运行中的llm_service.py进程
    LLM_PIDS=$(pgrep -f "llm_service.py")
    
    if [ -z "$LLM_PIDS" ]; then
        echo -e "${GREEN}未找到运行中的LLM服务${NC}"
    else
        echo -e "${YELLOW}找到运行中的LLM服务进程: $LLM_PIDS${NC}"
        
        # 终止进程
        for pid in $LLM_PIDS; do
            stop_process "$pid" "LLM服务"
        done
    fi
fi

# 停止Ollama服务（可选）
echo -e "\n${YELLOW}2. 检查Ollama服务...${NC}"
OLLAMA_PID_FILE="$LOG_DIR/ollama.pid"

if [ -f "$OLLAMA_PID_FILE" ]; then
    OLLAMA_PID=$(cat "$OLLAMA_PID_FILE")
    echo -e "${YELLOW}发现Ollama服务PID文件: $OLLAMA_PID${NC}"
    
    if stop_process "$OLLAMA_PID" "Ollama服务"; then
        rm -f "$OLLAMA_PID_FILE"
    fi
else
    echo -e "${YELLOW}未找到Ollama PID文件，检查运行中的进程...${NC}"
    
    # 查找运行中的ollama serve进程
    OLLAMA_PIDS=$(pgrep -f "ollama serve")
    
    if [ -z "$OLLAMA_PIDS" ]; then
        # 尝试查找ollama进程
        OLLAMA_PIDS=$(pgrep ollama)
    fi
    
    if [ -z "$OLLAMA_PIDS" ]; then
        echo -e "${GREEN}未找到运行中的Ollama服务${NC}"
    else
        echo -e "${YELLOW}找到运行中的Ollama服务进程: $OLLAMA_PIDS${NC}"
        
        read -p "是否停止Ollama服务? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # 终止进程
            for pid in $OLLAMA_PIDS; do
                stop_process "$pid" "Ollama服务"
            done
        else
            echo -e "${YELLOW}跳过停止Ollama服务${NC}"
        fi
    fi
fi

# 检查端口占用情况
echo -e "\n${YELLOW}3. 检查端口占用情况...${NC}"

# 检查LLM服务端口范围
echo -e "${YELLOW}检查LLM服务端口 (8000-8010)...${NC}"
for port in {8000..8010}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}端口 $port 仍被占用${NC}"
        PORT_PIDS=$(lsof -Pi :$port -sTCP:LISTEN -t)
        echo -e "${YELLOW}占用进程: $PORT_PIDS${NC}"
        
        if [ -t 0 ]; then
            # 交互模式
            read -p "是否强制释放端口 $port? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for pid in $PORT_PIDS; do
                    stop_process "$pid" "端口${port}占用进程"
                done
            fi
        else
            # 非交互模式，自动释放
            echo -e "${YELLOW}非交互模式，自动释放端口 $port${NC}"
            for pid in $PORT_PIDS; do
                stop_process "$pid" "端口${port}占用进程"
            done
        fi
    else
        echo -e "${GREEN}✓ 端口 $port 已释放${NC}"
    fi
done

# 检查Ollama端口范围
echo -e "${YELLOW}检查Ollama服务端口 (11434-11444)...${NC}"
for port in {11434..11444}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}端口 $port (Ollama) 仍被占用${NC}"
        PORT_PIDS=$(lsof -Pi :$port -sTCP:LISTEN -t)
        echo -e "${YELLOW}占用进程: $PORT_PIDS${NC}"
        
        if [ -t 0 ]; then
            # 交互模式
            read -p "是否强制释放端口 $port? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for pid in $PORT_PIDS; do
                    stop_process "$pid" "端口${port}占用进程"
                done
            fi
        else
            # 非交互模式，自动释放
            echo -e "${YELLOW}非交互模式，自动释放端口 $port${NC}"
            for pid in $PORT_PIDS; do
                stop_process "$pid" "端口${port}占用进程"
            done
        fi
    else
        echo -e "${GREEN}✓ 端口 $port 已释放${NC}"
    fi
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}    服务停止完成${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}提示:${NC}"
echo -e "• 重新启动服务: ./start_multi_llm_services.sh --daemon"
echo -e "• 查看日志: tail -f ./logs/llm_service_*.log"
echo -e "• 清理日志: rm -f ./logs/llm_service_*.log ./logs/llm_service_*.pid" 