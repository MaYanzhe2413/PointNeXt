#!/bin/bash
# ============================================================================
# 批量实验脚本: 剩余未完成实验
# 用法: ./batch_experiment.sh [--dry-run] [--gpu GPU_IDS] [--seed SEED]
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============== 默认参数 ==============
GPU_IDS="0,1,2"
SEED=7895
DRY_RUN=false
LOG_DIR="experiment_logs"

while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --gpu|-g) GPU_IDS="$2"; shift 2 ;;
        --seed|-S) SEED="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            echo -e "${BLUE}批量实验脚本${NC}"
            echo "用法: ./batch_experiment.sh [--dry-run] [--gpu GPU_IDS] [--seed SEED]"
            echo ""
            echo "选项:"
            echo "  --dry-run       只打印命令，不执行"
            echo "  --gpu, -g       GPU编号 (默认: 0,1,2)"
            echo "  --seed, -S      随机种子 (默认: 7895)"
            echo "  --log-dir       日志目录 (默认: experiment_logs)"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ============== 运行函数 ==============
run_experiment() {
    local desc="$1"
    local cmd="$2"
    local log_file="$3"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    echo -e "${BLUE}日志: ${NC}$log_file"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] 跳过执行${NC}"
        echo "$cmd" >> "$LOG_DIR/dry_run_commands.txt"
        return 0
    fi

    echo -e "${GREEN}开始运行...${NC}"
    eval "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[完成] $desc${NC}"
    else
        echo -e "${RED}[失败] $desc (exit code: $exit_code)${NC}"
    fi
    echo ""
    return $exit_code
}

# ============== 开始实验 ==============
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       批量实验 (剩余: S3DIS random + ModelNet40)  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC}  $GPU_IDS"
echo -e "${GREEN}Seed:${NC} $SEED"
echo -e "${GREEN}模式:${NC} $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo '实际运行')"
echo ""

TOTAL=0
COMPLETED=0
FAILED=0
FAILED_LIST=""
COMPLETED_LIST=""

# ============================================================
# ModelNet40: Baseline + KD+FPS + KD+Random
#    num_points=1024 (数据软链接已建好)
# ============================================================
echo -e "\n${BLUE}========== ModelNet40 ==========${NC}"

# Baseline
TOTAL=$((TOTAL + 1))
DESC="modelnet40_baseline_fps"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml seed=$SEED cfg_basename=pointnext-s_baseline"
LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"
run_experiment "$DESC" "$CMD" "$LOG_FILE"
if [ $? -eq 0 ]; then COMPLETED=$((COMPLETED + 1)); COMPLETED_LIST="$COMPLETED_LIST\n  $DESC"; else FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST\n  $DESC"; fi

# KD+FPS
MODELNET_FPS_SIZES="32,64,128,256,512,1024"
IFS=',' read -ra MODELNET_FS <<< "$MODELNET_FPS_SIZES"
for LS in "${MODELNET_FS[@]}"; do
    TOTAL=$((TOTAL + 1))
    DESC="modelnet40_kd_fps_leaf${LS}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s_kdtree_fps.yaml model.encoder_args.sampler_args.leaf_size=$LS model.encoder_args.sampler_args.strategy=fps seed=$SEED cfg_basename=pointnext-s_kdtree${LS}fps"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"
    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    if [ $? -eq 0 ]; then COMPLETED=$((COMPLETED + 1)); COMPLETED_LIST="$COMPLETED_LIST\n  $DESC"; else FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST\n  $DESC"; fi
done

# KD+Random
MODELNET_RANDOM_SIZES="64,256,512"
IFS=',' read -ra MODELNET_RS <<< "$MODELNET_RANDOM_SIZES"
for LS in "${MODELNET_RS[@]}"; do
    TOTAL=$((TOTAL + 1))
    DESC="modelnet40_kd_random_leaf${LS}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s_kdtree_fps.yaml model.encoder_args.sampler_args.leaf_size=$LS model.encoder_args.sampler_args.strategy=random seed=$SEED cfg_basename=pointnext-s_kdtree${LS}random"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"
    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    if [ $? -eq 0 ]; then COMPLETED=$((COMPLETED + 1)); COMPLETED_LIST="$COMPLETED_LIST\n  $DESC"; else FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST\n  $DESC"; fi
done

# ============== 汇总 ==============
echo -e "\n${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  实验汇总                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC}   $TOTAL"
echo -e "${GREEN}完成:${NC}   $COMPLETED"
echo -e "${RED}失败:${NC}   $FAILED"
echo ""
if [ -n "$COMPLETED_LIST" ]; then
    echo -e "${GREEN}完成的实验:${NC}$COMPLETED_LIST"
    echo ""
fi
if [ -n "$FAILED_LIST" ]; then
    echo -e "${RED}失败的实验:${NC}$FAILED_LIST"
    echo ""
fi
echo -e "${YELLOW}日志目录: $LOG_DIR/${NC}"
echo -e "${YELLOW}提取结果: grep -h 'best val miou\|best acc' $LOG_DIR/*.log${NC}"
