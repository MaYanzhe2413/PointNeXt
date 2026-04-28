#!/bin/bash
# ============================================================================
# 对照实验: axis_strategy (cycle vs max_spread) on S3DIS
# 已有数据 (cycle, fps):
#   leaf=325  -> 61.61
#   leaf=750  -> 61.21
#   leaf=1500 -> 61.95 (best)
#   leaf=3000 -> 61.63
# 已有数据 (max_spread, fps):
#   leaf=1500 -> 60.15
# 本脚本: 补齐 max_spread 在 325/750/3000 上的结果
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
            echo "axis_strategy 对照实验"
            echo "用法: ./batch_axis_strategy.sh [--dry-run] [--gpu GPU_IDS] [--seed SEED]"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

run_experiment() {
    local desc="$1"
    local cmd="$2"
    local log_file="$3"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    echo -e "${BLUE}日志: ${NC}$log_file"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"
        echo "$cmd" >> "$LOG_DIR/dry_run_axis_strategy.txt"
        return 0
    fi

    eval "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    [ $exit_code -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $exit_code
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     axis_strategy 对照实验 (max_spread 补齐)       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC}  $GPU_IDS"
echo -e "${GREEN}Seed:${NC} $SEED"
echo ""

TOTAL=0
COMPLETED=0
FAILED=0
FAILED_LIST=""
COMPLETED_LIST=""

# max_spread 在 325/750/3000 上补齐 (1500 已跑过 60.15%)
LEAF_SIZES="325 750 3000"
for LS in $LEAF_SIZES; do
    TOTAL=$((TOTAL + 1))
    DESC="s3dis_kd_fps_leaf${LS}_maxspread"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
        --cfg cfgs/s3dis/pointnext-s_kdtree84fps.yaml \
        model.encoder_args.sampler_args.leaf_size=$LS \
        model.encoder_args.sampler_args.strategy=fps \
        model.encoder_args.sampler_args.axis_strategy=max_spread \
        seed=$SEED \
        cfg_basename=pointnext-s_kdtree${LS}fps_maxspread"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"
    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        COMPLETED_LIST="$COMPLETED_LIST\n  $DESC"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST\n  $DESC"
    fi
done

# ============== 汇总 ==============
echo -e "\n${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  对照实验汇总                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC}   $TOTAL"
echo -e "${GREEN}完成:${NC}   $COMPLETED"
echo -e "${RED}失败:${NC}   $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比命令:${NC}"
echo -e "  grep -h 'best val miou' $LOG_DIR/s3dis_kd_fps_leaf*_maxspread*.log"
