#!/bin/bash
# ============================================================================
# 通用 training-free coord-quant ablation ladder
# 支持: {pointnext-s, pointnet++} x {s3dis, modelnet40ply2048, shapenetpart}
#
# 设计: A0/A1 只差 sampler (CLI 覆盖 base config), 两个模型/三个任务统一接口.
#   A0       global FPS (base config, default fps)            baseline
#   A1       + KD block-local FPS (sampler=kdtree, fps)       [Z] retrain-free cost
#   A2       + int8 sample coords                             int8 sampling effect
#   A3       + bq/dp int8 (naive)                             catastrophic counter-example
#   A3-safe  + bq/dp 9-bit + dp_postquant uint8 (ours)        [A] recommended config
#   --sweep  leaf-size sensitivity
#
# 用法:
#   bash batch_ablation_general.sh --task TASK --model MODEL --dataset DS -p CKPT [opts]
#
# 例:
#   # PointNeXt-S on ModelNet40
#   bash batch_ablation_general.sh --task cls --model pointnext-s \
#       --dataset modelnet40ply2048 -p <ckpt> --sweep
#   # PointNet++ on S3DIS
#   bash batch_ablation_general.sh --task seg --model pointnet++ \
#       --dataset s3dis -p <ckpt> --sweep
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

GPU_IDS="0"
SEED=7895
TASK=""          # seg | cls | partseg
MODEL=""         # pointnext-s | pointnet++
DATASET=""       # s3dis | modelnet40ply2048 | shapenetpart
PRETRAINED=""
LEAF=""          # 留空则按数据集自动选
DRY_RUN=false
DO_SWEEP=false
LOG_DIR="experiment_logs"

while [ $# -gt 0 ]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --gpu|-g) GPU_IDS="$2"; shift 2 ;;
        --seed|-S) SEED="$2"; shift 2 ;;
        --pretrained|-p) PRETRAINED="$2"; shift 2 ;;
        --leaf|-L) LEAF="$2"; shift 2 ;;
        --sweep) DO_SWEEP=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
通用 coord-quant ablation ladder
必填: --task {seg|cls|partseg}  --model {pointnext-s|pointnet++}  --dataset {s3dis|modelnet40ply2048|shapenetpart}  -p <ckpt>
可选: -g GPU(默认0)  -S seed(默认7895)  -L leaf(默认按数据集)  --sweep  --dry-run
EOF
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

# ---- 参数校验 + 任务派发 ----
if [ -z "$TASK" ] || [ -z "$MODEL" ] || [ -z "$DATASET" ] || [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 必须指定 --task --model --dataset -p${NC}"; echo "用 -h 看帮助"; exit 1
fi
if [ ! -f "$PRETRAINED" ]; then
    echo -e "${RED}错误: checkpoint 不存在: $PRETRAINED${NC}"; exit 1
fi

case $TASK in
    seg)     MAIN="examples/segmentation/main.py";  METRIC_GREP="Best ckpt.*test_miou" ;;
    cls)     MAIN="examples/classification/main.py"; METRIC_GREP="E@.*OA:" ;;
    partseg) MAIN="examples/shapenetpart/main.py";   METRIC_GREP="Instance mIoU" ;;
    *) echo -e "${RED}未知 task: $TASK (应为 seg|cls|partseg)${NC}"; exit 1 ;;
esac

CFG="cfgs/$DATASET/$MODEL.yaml"
if [ ! -f "$CFG" ]; then
    echo -e "${RED}错误: 配置不存在: $CFG${NC}"; exit 1
fi

# ---- 按数据集自动选 leaf + sweep 列表 (保持 ~16 个叶子) ----
if [ -z "$LEAF" ]; then
    case $DATASET in
        s3dis)             LEAF=1500 ;;   # 24000 pts
        modelnet40ply2048) LEAF=64   ;;   # 1024 pts
        shapenetpart)      LEAF=128  ;;   # 2048 pts
        *)                 LEAF=128  ;;
    esac
fi
case $DATASET in
    s3dis)             SWEEP_LEAVES="256 512 1024 1500 3000 24000" ;;
    modelnet40ply2048) SWEEP_LEAVES="16 32 64 128 256 1024" ;;
    shapenetpart)      SWEEP_LEAVES="32 64 128 256 512 2048" ;;
    *)                 SWEEP_LEAVES="$LEAF" ;;
esac

PREFIX="abl_${MODEL//+/p}_${DATASET}"   # pointnet++ -> pointnetpp in tag

run_exp() {
    local desc="$1"; local extra="$2"
    local tag="${PREFIX}_${desc}"
    local log="$LOG_DIR/${tag}_seed${SEED}.log"
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python $MAIN --cfg $CFG mode=test --pretrained_path $PRETRAINED $extra seed=$SEED cfg_basename=$tag"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$tag"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"; echo "$cmd" >> "$LOG_DIR/dry_run_${PREFIX}.txt"; return 0
    fi
    eval "$cmd" 2>&1 | tee "$log"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

mkdir -p "$LOG_DIR"
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         通用 Coord-Quant Ablation Ladder (mode=test)         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}Task:${NC} $TASK   ${GREEN}Model:${NC} $MODEL   ${GREEN}Dataset:${NC} $DATASET"
echo -e "${GREEN}Cfg:${NC} $CFG   ${GREEN}Leaf:${NC} $LEAF   ${GREEN}GPU:${NC} $GPU_IDS   ${GREEN}Seed:${NC} $SEED"
echo -e "${GREEN}Ckpt:${NC} $PRETRAINED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0; FAILED_LIST=""; COMPLETED_LIST=""
mark() { if [ $1 -eq 0 ]; then COMPLETED=$((COMPLETED+1)); COMPLETED_LIST="$COMPLETED_LIST\n  $2"; else FAILED=$((FAILED+1)); FAILED_LIST="$FAILED_LIST\n  $2"; fi; }

# sampler 覆盖串 (A1+ 共用)
KD="model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=$LEAF model.encoder_args.sampler_args.strategy=fps model.encoder_args.sampler_args.proportional=True"

# ---- A0: global FPS baseline ----
TOTAL=$((TOTAL+1)); run_exp "A0_global_fps" ""; mark $? "A0_global_fps"

# ---- A1: block FPS ----
TOTAL=$((TOTAL+1)); run_exp "A1_block_fps" "$KD"; mark $? "A1_block_fps"

# ---- A2: + int8 sample ----
TOTAL=$((TOTAL+1)); run_exp "A2_sample8" "$KD model.encoder_args.coord_sample_nbits=8"; mark $? "A2_sample8"

# ---- A3: + bq/dp int8 (naive) ----
TOTAL=$((TOTAL+1)); run_exp "A3_naive_int8" "$KD model.encoder_args.coord_sample_nbits=8 model.encoder_args.coord_bq_nbits=8 model.encoder_args.coord_dp_nbits=8"; mark $? "A3_naive_int8"

# ---- A3-safe: + bq/dp 9-bit + dp_postquant uint8 ----
TOTAL=$((TOTAL+1)); run_exp "A3safe_bqdp9" "$KD model.encoder_args.coord_sample_nbits=8 model.encoder_args.coord_bq_nbits=9 model.encoder_args.coord_dp_nbits=9 model.encoder_args.coord_dp_postquant_nbits=8"; mark $? "A3safe_bqdp9"

# ---- 可选 leaf sweep (A1 block FPS) ----
if [ "$DO_SWEEP" = true ]; then
    echo -e "\n${BLUE}========== leaf size sweep ==========${NC}"
    for L in $SWEEP_LEAVES; do
        TOTAL=$((TOTAL+1))
        SWEEP_KD="model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=$L model.encoder_args.sampler_args.strategy=fps model.encoder_args.sampler_args.proportional=True"
        run_exp "sweep_leaf$L" "$SWEEP_KD"; mark $? "sweep_leaf$L"
    done
fi

# ---- 汇总 ----
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                          汇总                                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}取结果 (本次 metric grep):${NC}"
echo -e "  grep -hE '$METRIC_GREP' $LOG_DIR/${PREFIX}_*.log"
