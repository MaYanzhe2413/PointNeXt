#!/bin/bash
# ============================================================================
# Encoder bq+dp 坐标精度扫描 (S3DIS, 方案A: 推理时量化)
#
# 三路隔离已确认: 采样 int8 免费, bq 和 dp 有强超加性 (各掉 ~1, 合掉 ~4).
# 本脚本固定采样=int8, 扫 bq+dp 的位宽, 找出恢复精度所需的最小精度,
# 并通过 bq8_dp12 / bq12_dp8 判断到底是 bq 还是 dp 更需要高精度.
#
# 用法: bash batch_encoder_bitsweep.sh -p <checkpoint.pth>
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
            echo "Encoder bq+dp 坐标精度扫描"
            echo "用法: bash batch_encoder_bitsweep.sh -p <checkpoint.pth> [-g GPU]"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 需要 -p <checkpoint.pth>${NC}"; exit 1
fi

# 实验列表: name|sample_nbits|bq_nbits|dp_nbits   (采样固定 int8, 已证明免费)
# nbits: -1=FP16, 8/10/12/16=intN
EXPERIMENTS=(
    "bqdp_int10|8|10|10"   # 10bit
    "bqdp_int12|8|12|12"   # 12bit
    "bqdp_int16|8|16|16"   # 16bit
    "bq8_dp12|8|8|12"      # dp 保精度, bq 省
    "bq12_dp8|8|12|8"      # bq 保精度, dp 省
    "bqdp_fp16|8|-1|-1"    # FP16
)

run_experiment() {
    local desc="$1" cmd="$2" log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_encoder_bitsweep.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Encoder bq+dp 精度扫描 (S3DIS, 方案A)        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME SAMPLE BQ DP <<< "$EXP"
    TOTAL=$((TOTAL + 1))
    DESC="enc_sweep_${NAME}"
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
echo -e "${BLUE}║          Encoder bq+dp 精度扫描汇总               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/enc_sweep_*.log"
