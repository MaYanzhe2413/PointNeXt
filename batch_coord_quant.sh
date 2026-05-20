#!/bin/bash
# ============================================================================
# 坐标量化敏感度实验 (S3DIS)
#
# 方案A (推理时量化, 快): 用已训练的 FP32 模型, 测试时把坐标 fake-quant 到 n-bit
#   ./batch_coord_quant.sh -p <checkpoint.pth>
#
# 方案B (训练时量化, 慢): 从头训练, 坐标全程 fake-quant
#   ./batch_coord_quant.sh --train
#
# coord_nbits=0 表示禁用 (FP32 baseline)
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============== 默认参数 ==============
GPU_IDS="0"
SEED=7895
CFG="cfgs/s3dis/pointnext-s.yaml"
PRETRAINED=""
MODE="test"          # test = 方案A; train = 方案B
DRY_RUN=false
LOG_DIR="experiment_logs"
BITS="0 8 7 6 5 4"   # 0 = FP32 baseline
EXTRA=""             # 额外的 config 覆盖 (如 sampler 参数), 需匹配 checkpoint

while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --gpu|-g) GPU_IDS="$2"; shift 2 ;;
        --seed|-S) SEED="$2"; shift 2 ;;
        --pretrained|-p) PRETRAINED="$2"; shift 2 ;;
        --cfg|-c) CFG="$2"; shift 2 ;;
        --train) MODE="train"; shift ;;
        --bits) BITS="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "坐标量化敏感度实验"
            echo "用法:"
            echo "  方案A (快): ./batch_coord_quant.sh -p <checkpoint.pth> [-g GPU]"
            echo "  方案B (慢): ./batch_coord_quant.sh --train [-g GPU0,GPU1,GPU2]"
            echo ""
            echo "选项:"
            echo "  -p, --pretrained  已训练 checkpoint 路径 (方案A 必填)"
            echo "  -c, --cfg         配置文件 (默认: cfgs/s3dis/pointnext-s.yaml)"
            echo "  -g, --gpu         GPU 编号 (默认: 0)"
            echo "  -S, --seed        随机种子 (默认: 7895)"
            echo "  --bits            要测的 bit 宽度 (默认: '0 8 7 6 5 4')"
            echo "  --extra           额外 config 覆盖 (需匹配 checkpoint 的 sampler 参数)"
            echo "  --train           方案B: 从头训练而非测试"
            echo "  --dry-run         只打印命令"
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

if [ "$MODE" = "test" ] && [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 方案A 需要 -p <checkpoint.pth>${NC}"
    echo "用 -h 查看帮助"
    exit 1
fi

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
        echo "$cmd" >> "$LOG_DIR/dry_run_coord_quant.txt"
        return 0
    fi

    eval "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    [ $exit_code -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $exit_code
}

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         坐标量化敏感度实验 (S3DIS)                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}方案:${NC} $([ "$MODE" = "test" ] && echo 'A (推理时量化)' || echo 'B (训练时量化)')"
echo -e "${GREEN}GPU:${NC}  $GPU_IDS"
echo -e "${GREEN}Bits:${NC} $BITS"
[ "$MODE" = "test" ] && echo -e "${GREEN}模型:${NC} $PRETRAINED"
echo ""

TOTAL=0
COMPLETED=0
FAILED=0
FAILED_LIST=""
COMPLETED_LIST=""

for NB in $BITS; do
    TOTAL=$((TOTAL + 1))
    if [ "$NB" = "0" ]; then
        TAG="fp32"
    else
        TAG="${NB}bit"
    fi

    if [ "$MODE" = "test" ]; then
        DESC="coord_quant_test_${TAG}"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
            --cfg $CFG \
            mode=test \
            --pretrained_path $PRETRAINED \
            model.encoder_args.coord_nbits=$NB \
            $EXTRA \
            seed=$SEED \
            cfg_basename=coord_quant_${TAG}"
    else
        DESC="coord_quant_train_${TAG}"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
            --cfg $CFG \
            model.encoder_args.coord_nbits=$NB \
            $EXTRA \
            seed=$SEED \
            cfg_basename=coord_quant_train_${TAG}"
    fi

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
echo -e "${BLUE}║              坐标量化实验汇总                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC}   $TOTAL"
echo -e "${GREEN}完成:${NC}   $COMPLETED"
echo -e "${RED}失败:${NC}   $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}对比结果:${NC}"
echo -e "  grep -h 'miou' $LOG_DIR/coord_quant_*.log"
