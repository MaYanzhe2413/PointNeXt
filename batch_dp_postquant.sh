#!/bin/bash
# ============================================================================
# dp post-normalize 量化扫描 (S3DIS, 方案A)
#
# 之前的 coord_dp_nbits 量化的是【源坐标】, 这一组实验隔离【归一化后】的
# dp_norm 量化: 源坐标保 FP32, 只把 /radius 之后的 dp ∈ [-1, +1] 量化到 N bit.
# 验证硬件要做的 (S=1/2^(N-1), Z=0 中心对称) post-normalize quant 是否安全.
#
# 用法: bash batch_dp_postquant.sh -p <checkpoint.pth>
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
            echo "dp post-normalize 量化扫描"
            echo "用法: bash batch_dp_postquant.sh -p <checkpoint.pth> [-g GPU]"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 需要 -p <checkpoint.pth>${NC}"; exit 1
fi

# 实验列表: name|postquant_nbits  (源坐标全 FP32, 只 quant dp_norm)
# nbits=8 -> 256 levels in [-1,+1], 步长 1/128 = 0.0078
# nbits=6 -> 64 levels,                 步长 1/32  = 0.031
# nbits=4 -> 16 levels,                 步长 1/8   = 0.125
# nbits=3 -> 8 levels,                  步长 1/4   = 0.25
# nbits=2 -> 4 levels,                  步长 1/2   = 0.5
EXPERIMENTS=(
    "post8|8"     # 硬件标称, 应 ~baseline 61.73
    "post6|6"     # 64 级, 应几乎无损
    "post5|5"     # 32 级
    "post4|4"     # 16 级
    "post3|3"     # 8 级, 开始掉
    "post2|2"     # 4 级, 重伤
)

run_experiment() {
    local desc="$1" cmd="$2" log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_dp_postquant.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      dp post-normalize 量化扫描 (S3DIS, 方案A)    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME NB <<< "$EXP"
    TOTAL=$((TOTAL + 1))
    DESC="dp_postq_${NAME}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
        --cfg $CFG \
        mode=test \
        --pretrained_path $PRETRAINED \
        $EXTRA \
        model.encoder_args.coord_dp_postquant_nbits=$NB \
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
echo -e "${BLUE}║          dp post-normalize 扫描汇总               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/dp_postq_*.log"
