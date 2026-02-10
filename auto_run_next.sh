#!/bin/bash

# 自动化脚本：监控所有GPU进程，当所有GPU空闲持续15分钟时自动运行下一个任务
# 用法：./auto_run_next.sh <目标窗格ID>
# 示例：./auto_run_next.sh %0
# ./auto_run_next.sh pipeline
# 配置参数
NEXT_COMMAND="python main.py --config config_iterative_refinement.yaml --mode iterative"
WORK_DIR="/data2/yixuan/image_edit_benchmark"
GPU_COUNT=6             # GPU数量
IDLE_THRESHOLD=900      # 15分钟 = 900秒
CHECK_INTERVAL=15       # 每30秒检查一次

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查参数
if [ -z "$1" ]; then
    echo -e "${RED}错误：请提供目标tmux窗格ID${NC}"
    echo "用法: $0 <窗格ID>"
    echo ""
    echo "查看所有窗格："
    echo "  tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} 窗格ID: #{pane_id}'"
    echo ""
    echo "示例："
    echo "  $0 %0"
    exit 1
fi

TMUX_PANE="$1"

# 验证窗格是否存在
TMUX_SESSION=$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}' 2>/dev/null)
if [ -z "$TMUX_SESSION" ]; then
    echo -e "${RED}错误：无法找到窗格 $TMUX_PANE${NC}"
    echo "请使用以下命令查看所有窗格："
    echo "  tmux list-panes -a"
    exit 1
fi

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误：未找到nvidia-smi命令，请确保NVIDIA驱动已安装${NC}"
    exit 1
fi

# 检测实际GPU数量
DETECTED_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$DETECTED_GPU_COUNT" -lt "$GPU_COUNT" ]; then
    echo -e "${YELLOW}警告：配置GPU数量为${GPU_COUNT}，但只检测到${DETECTED_GPU_COUNT}张GPU${NC}"
    GPU_COUNT=$DETECTED_GPU_COUNT
fi

echo "========================================"
echo -e "${GREEN}GPU空闲监控脚本${NC}"
echo "========================================"
echo "监控的tmux会话: $TMUX_SESSION"
echo "监控的tmux窗格: $TMUX_PANE"
echo "工作目录: $WORK_DIR"
echo "GPU数量: $GPU_COUNT"
echo "空闲阈值: ${IDLE_THRESHOLD}秒 (15分钟)"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo "========================================"
echo -e "${BLUE}监控逻辑: 当所有GPU上都没有进程的状态持续15分钟时，启动下一任务${NC}"
echo "========================================"
echo ""

# 获取所有GPU上的进程数量
get_gpu_process_count() {
    # 使用nvidia-smi获取所有GPU上的进程数量
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l
}

# 获取每个GPU的详细信息
get_gpu_details() {
    local total_processes=0
    echo "GPU状态详情："
    for ((i=0; i<$GPU_COUNT; i++)); do
        local gpu_procs=$(nvidia-smi -i $i --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
        local proc_count=$(echo "$gpu_procs" | grep -v '^$' | wc -l)
        total_processes=$((total_processes + proc_count))
        
        if [ $proc_count -gt 0 ]; then
            echo -e "  GPU $i: ${YELLOW}$proc_count 个进程${NC}"
            echo "$gpu_procs" | while IFS=',' read -r pid name memory; do
                echo "    - PID: $pid, 进程: $name, 显存: $memory"
            done
        else
            echo -e "  GPU $i: ${GREEN}空闲${NC}"
        fi
    done
    return $total_processes
}

# 获取简化的GPU状态
get_gpu_status_simple() {
    local gpu_status=""
    for ((i=0; i<$GPU_COUNT; i++)); do
        local proc_count=$(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ $proc_count -gt 0 ]; then
            gpu_status="${gpu_status}GPU$i:$proc_count "
        else
            gpu_status="${gpu_status}GPU$i:0 "
        fi
    done
    echo "$gpu_status"
}

# 初始化
echo -e "${YELLOW}正在初始化，检测当前GPU状态...${NC}"
sleep 2

INITIAL_PROCESS_COUNT=$(get_gpu_process_count)
echo ""
echo "初始状态："
get_gpu_details
INITIAL_TOTAL=$?
echo "总进程数: $INITIAL_PROCESS_COUNT"
echo ""

if [ $INITIAL_PROCESS_COUNT -eq 0 ]; then
    echo -e "${YELLOW}注意：当前所有GPU都是空闲的，将立即开始15分钟倒计时${NC}"
    IDLE_START_TIME=$(date +%s)
else
    echo -e "${GREEN}开始监控... 等待所有GPU空闲${NC}"
    IDLE_START_TIME=0
fi

echo "------------------------------------------------------------"
echo ""

check_count=0

while true; do
    check_count=$((check_count + 1))
    current_time=$(date +%s)
    
    # 获取当前所有GPU的进程总数
    CURRENT_PROCESS_COUNT=$(get_gpu_process_count)
    
    if [ $CURRENT_PROCESS_COUNT -eq 0 ]; then
        # 所有GPU都空闲
        if [ $IDLE_START_TIME -eq 0 ]; then
            # 刚刚变为空闲
            IDLE_START_TIME=$current_time
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}✓ 所有GPU已空闲，开始15分钟倒计时${NC}"
            echo ""
        fi
        
        # 计算空闲时长
        idle_duration=$((current_time - IDLE_START_TIME))
        remaining_time=$((IDLE_THRESHOLD - idle_duration))
        
        if [ $remaining_time -le 0 ]; then
            # 达到阈值，任务结束
            echo ""
            echo "============================================================"
            echo -e "${GREEN}✓✓✓ 所有GPU已持续空闲15分钟，任务完成！✓✓✓${NC}"
            echo "============================================================"
            echo ""
            get_gpu_details
            echo ""
            echo "等待3秒后启动下一个任务..."
            sleep 3
            
            # 检查窗格是否仍然存在
            if tmux list-panes -t "$TMUX_PANE" &>/dev/null; then
                echo ""
                echo "正在tmux窗格 $TMUX_PANE 中启动下一个任务..."
                echo "工作目录: $WORK_DIR"
                echo "命令: $NEXT_COMMAND"
                echo ""
                
                # 直接执行命令
                tmux send-keys -t "$TMUX_PANE" "$NEXT_COMMAND" C-m
                
                echo "============================================================"
                echo -e "${GREEN}✓✓✓ 下一个任务已成功启动！✓✓✓${NC}"
                echo "============================================================"
                exit 0
            else
                echo -e "${RED}错误：tmux窗格 $TMUX_PANE 已不存在${NC}"
                exit 1
            fi
        else
            # 显示倒计时
            remaining_minutes=$((remaining_time / 60))
            remaining_seconds=$((remaining_time % 60))
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}所有GPU空闲中${NC} - 已空闲: ${idle_duration}s / ${IDLE_THRESHOLD}s (剩余: ${remaining_minutes}m ${remaining_seconds}s)"
        fi
    else
        # 有GPU在运行进程
        if [ $IDLE_START_TIME -ne 0 ]; then
            # 之前是空闲的，现在又有进程了
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}检测到GPU活动，重置倒计时${NC}"
            get_gpu_details
            echo ""
            IDLE_START_TIME=0
        else
            # 持续有进程运行
            if [ $((check_count % 2)) -eq 0 ]; then
                gpu_status=$(get_gpu_status_simple)
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU运行中 - 总进程数: $CURRENT_PROCESS_COUNT [$gpu_status]"
            fi
        fi
    fi
    
    # 每5分钟（10次检查）输出一次详细状态
    if [ $((check_count % 10)) -eq 0 ]; then
        echo ""
        echo "========== 状态摘要 (检查次数: $check_count) =========="
        get_gpu_details
        
        if [ $IDLE_START_TIME -ne 0 ]; then
            idle_duration=$((current_time - IDLE_START_TIME))
            remaining_time=$((IDLE_THRESHOLD - idle_duration))
            remaining_minutes=$((remaining_time / 60))
            remaining_seconds=$((remaining_time % 60))
            echo -e "${GREEN}空闲倒计时: ${idle_duration}s / ${IDLE_THRESHOLD}s (剩余: ${remaining_minutes}m ${remaining_seconds}s)${NC}"
        else
            echo -e "${YELLOW}等待所有GPU空闲...${NC}"
        fi
        echo "=================================================="
        echo ""
    fi
    
    sleep $CHECK_INTERVAL
done
