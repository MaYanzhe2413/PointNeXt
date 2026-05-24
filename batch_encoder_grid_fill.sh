#!/bin/bash
# ============================================================================
# Encoder bq+dp 网格补全 (S3DIS, 方案A)
#
# 当前 4x4 网格 (8/9/10/12) 有 6 个空格, 本脚本跑这 6 个组合,
# 完整覆盖以验证"任一路 >=9bit 就完全恢复"的规律.
#
# 预测:
#   bq=8 那一行  : 全部 ~60.96 (黄, 只剩 bq8 单独惩罚)
#   dp=8 那一列  : 全部 ~60.5  (黄, 只剩 dp8 单独惩罚)
#   其他        : 全部 ~61.8  (绿, 完全恢复)
#
# 用法: bash batch_encoder_grid_fill.sh -p <checkpoint.pth>
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

GPU_IDS="0"
SEED=7895
CFG="cfgs/s3dis/pointnext-s_kdtree84fps.yaml"
PRETRAINED=""
DRY_RUN=false
LOG_DIR="experiment_logs"
EXTRA="model.encoder_args.sampler_args.leaf_size=1500 model.encoder_args.sampler_args.strategy=fps model.encoder_args.sampler_args.axis_strategy=max_spread"

while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --gpu|-g) GPU_IDS="$2"; shift 2 ;;
        --seed|-S) SEED="$2"; shift 2 ;;
        --pretrained|-p) PRETRAINED="$2"; shift 2 ;;
        --cfg|-c) CFG="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Encoder bq+dp 网格补全"
            echo "用法: bash batch_encoder_grid_fill.sh -p <checkpoint.pth> [-g GPU]"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 需要 -p <checkpoint.pth>${NC}"; exit 1
fi

# 实验列表: name|sample_nbits|bq_nbits|dp_nbits   (采样固定 int8)
EXPERIMENTS=(
    "bq8_dp10|8|8|10"      # 预测 ~60.96 (bq=8 行)
    "bq9_dp12|8|9|12"      # 预测 ~61.8 (两路都 >=9)
    "bq10_dp8|8|10|8"      # 预测 ~60.5 (dp=8 列)
    "bq10_dp12|8|10|12"    # 预测 ~61.8
    "bq12_dp9|8|12|9"      # 预测 ~61.8
    "bq12_dp10|8|12|10"    # 预测 ~61.8
)

run_experiment() {
    local desc="$1" cmd="$2" log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_grid_fill.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Encoder bq+dp 网格补全 (S3DIS, 方案A)        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME SAMPLE BQ DP <<< "$EXP"
    TOTAL=$((TOTAL + 1))
    DESC="enc_grid_${NAME}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
        --cfg $CFG \
        mode=test \
        --pretrained_path $PRETRAINED \
        $EXTRA \
        model.encoder_args.coord_sample_nbits=$SAMPLE \
        model.encoder_args.coord_bq_nbits=$BQ \
        model.encoder_args.coord_dp_nbits=$DP \
        seed=$SEED \
        cfg_basename=${DESC}"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"
    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1)); COMPLETED_LIST="$COMPLETED_LIST\n  $DESC"
    else
        FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST\n  $DESC"
    fi
done

echo -e "\n${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Encoder 网格补全汇总                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/enc_grid_*.log"
