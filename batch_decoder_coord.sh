#!/bin/bash
# ============================================================================
# Decoder 插值坐标量化消融实验 (S3DIS, 方案A: 推理时量化)
#
# 隔离 decoder three_interpolation 的坐标精度敏感度, 回答:
#   - 之前全网络 8bit 掉 4 mIoU, 到底是 encoder 还是 decoder 引起的?
#   - decoder 插值里, 是 KNN id 选错 还是 weight 精度 的问题?
#   - int8 topK + 高精度 refine 能否救回精度?
#
# 用法: bash batch_decoder_coord.sh -p <checkpoint.pth>
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
            echo "Decoder 插值坐标量化消融实验"
            echo "用法: bash batch_decoder_coord.sh -p <checkpoint.pth> [-g GPU]"
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

# 实验列表: name|knn_nbits|weight_nbits|topk
# knn/weight nbits: 0=FP32, -1=FP16, 8/12/16=int8/int12/int16
EXPERIMENTS=(
    "fp32_baseline|0|0|0"        # FP32 基准
    "step0_dec8|8|8|0"           # Step0: decoder-only 8bit, 对比全网络 57.73
    "A_knn8_wfp32|8|0|0"         # A: id 用 int8, weight 用 FP32
    "B_knnfp32_w8|0|8|0"         # B: id 用 FP32, weight 用 int8
    "C_topk8|8|0|8"              # C: int8 选 top8 候选 + FP32 refine
    "C_topk16|8|0|16"            # C: int8 选 top16 候选 + FP32 refine
    "D_int12|12|12|0"            # D: int12 定点 full KNN
    "D_int16|16|16|0"            # D: int16 定点 full KNN
    "E_fp16|-1|-1|0"             # E: FP16 full KNN
)

run_experiment() {
    local desc="$1" cmd="$2" log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_decoder_coord.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Decoder 插值坐标量化消融 (S3DIS, 方案A)      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME KNN WEIGHT TOPK <<< "$EXP"
    TOTAL=$((TOTAL + 1))
    DESC="dec_coord_${NAME}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
        --cfg $CFG \
        mode=test \
        --pretrained_path $PRETRAINED \
        $EXTRA \
        model.decoder_args.coord_knn_nbits=$KNN \
        model.decoder_args.coord_weight_nbits=$WEIGHT \
        model.decoder_args.coord_topk=$TOPK \
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
echo -e "${BLUE}║              Decoder 消融实验汇总                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/dec_coord_*.log"
