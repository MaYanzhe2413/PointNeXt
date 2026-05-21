#!/bin/bash
# ============================================================================
# Encoder 坐标量化三路隔离实验 (S3DIS, 方案A: 推理时量化)
#
# 已知: 全网络 int8 坐标掉 4 mIoU, decoder 部分只掉 0.07 -> 真凶在 encoder.
# encoder 用坐标三处:
#   sample : FPS/KD-tree 采样 (选哪些点保留)
#   bq     : ball_query     (选哪些邻居)
#   dp     : 相对位置 p_j-p_i (作为 +3 通道喂 Conv)
# 本脚本逐一隔离, 定位那 ~4 个点掉在哪.
#
# 用法: bash batch_encoder_coord.sh -p <checkpoint.pth>
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

GPU_IDS="0"
SEED=7895
CFG="cfgs/s3dis/pointnext-s_kdtree84fps.yaml"
PRETRAINED=""
DRY_RUN=false
LOG_DIR="experiment_logs"
# sampler 参数需匹配 checkpoint (maxspread-1500 那个)
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
            echo "Encoder 坐标量化三路隔离实验"
            echo "用法: bash batch_encoder_coord.sh -p <checkpoint.pth> [-g GPU]"
            echo "  -p  已训练 checkpoint 路径 (必填)"
            echo "  -c  配置文件 (默认 cfgs/s3dis/pointnext-s_kdtree84fps.yaml)"
            echo "  -g  GPU 编号 (默认 0)"
            echo "  --extra  sampler 参数 (默认匹配 maxspread-1500 checkpoint)"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 需要 -p <checkpoint.pth>${NC}"; exit 1
fi

# 实验列表: name|sample_nbits|bq_nbits|dp_nbits
# nbits: 0=FP32, -1=FP16, 8/12/16=int8/int12/int16
EXPERIMENTS=(
    "enc_fp32|0|0|0"        # FP32 基准
    "enc_all8|8|8|8"        # encoder 三路全 int8 (应接近全网络的 57.73)
    "enc_sample8|8|0|0"     # 只采样 int8
    "enc_bq8|0|8|0"         # 只 ball_query int8
    "enc_dp8|0|0|8"         # 只 dp 相对位置 int8 (怀疑这个掉最多)
)

run_experiment() {
    local desc="$1" cmd="$2" log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_encoder_coord.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Encoder 坐标量化三路隔离 (S3DIS, 方案A)      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME SAMPLE BQ DP <<< "$EXP"
    TOTAL=$((TOTAL + 1))
    DESC="enc_coord_${NAME}"
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
echo -e "${BLUE}║              Encoder 隔离实验汇总                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/enc_coord_*.log"
