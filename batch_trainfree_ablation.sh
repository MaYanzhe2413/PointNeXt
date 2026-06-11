#!/bin/bash
# ============================================================================
# Training-free ablation ladder (paper experiment 5.1-5 / [Z] / [A])
#
# 唯一目的: 在固定 checkpoint 上回答"KD 分块 + int8 坐标 + 量化栈"端到端
# 到底掉不掉点. 全程 mode=test, --pretrained_path 加载, 权重不动.
#
# Ladder (越往下偏差越多):
#   A0  official FP32 + 全局 FPS                    -> 基线 mIoU (paper 63.4)
#   A1  A0 + KD block-local FPS  (FP32 坐标)        -> 隔离 FPS 局部化的代价
#   A2  A1 + 采样用 int8 坐标                        -> 加上"int8 坐标对采样的影响"
#   A3  A2 + bq 和 dp 也用 int8 坐标 (灾难配置)      -> 端到端"naive 全 int8"
#   A3-safe  A2 + bq/dp 用 9-bit (我们实验的推荐配置) -> 论文最终数 [A]
#
# leaf 敏感度 sweep 见末尾.
#
# 用法:
#   bash batch_trainfree_ablation.sh -p <official_pointnext-s_checkpoint.pth>
#
# 重要: -p 必须是【全局 FPS 训练的 official PointNeXt-S checkpoint】, 否则 A0
# 跟训练设定不匹配, 整个 ladder 失去意义. 推荐 PointNeXt 官方 release 的 S3DIS
# Area-5 checkpoint (paper 63.4 mIoU). 如果只有 kdtree 训出来的 checkpoint,
# A0/A1 对比的 "FPS locality 代价" 不再可信 - 这种情况下脚本仍会跑, 但要在
# 论文里如实说明 checkpoint 来源.
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# ============== 默认参数 ==============
GPU_IDS="0"
SEED=7895
PRETRAINED=""
LEAF=1500              # SA1 leaf size, 跟我们之前的实验保持一致
DRY_RUN=false
LOG_DIR="experiment_logs"
DO_SWEEP=false         # 是否额外跑 leaf size sweep

# cfg 文件: A0 用全局 FPS 的 default, A1+ 用 kdtree 版本
CFG_A0="cfgs/s3dis/pointnext-s.yaml"
CFG_KDTREE="cfgs/s3dis/pointnext-s_kdtree84fps.yaml"

while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --gpu|-g) GPU_IDS="$2"; shift 2 ;;
        --seed|-S) SEED="$2"; shift 2 ;;
        --pretrained|-p) PRETRAINED="$2"; shift 2 ;;
        --leaf|-L) LEAF="$2"; shift 2 ;;
        --sweep) DO_SWEEP=true; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Training-free A0->A3 ablation ladder.
用法: bash batch_trainfree_ablation.sh -p <checkpoint.pth> [选项]

  -p, --pretrained  pretrained checkpoint 路径 (必填, 应为全局 FPS 训练的官方 ckpt)
  -g, --gpu         GPU 编号 (默认: 0)
  -S, --seed        seed (默认: 7895)
  -L, --leaf        SA1 KD 分块的 leaf_size (默认: 1500)
  --sweep           额外跑 leaf size 敏感度扫描 {256, 512, 1024, 1500, 24000}
  --dry-run         只打印命令不执行
EOF
            exit 0 ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [ -z "$PRETRAINED" ]; then
    echo -e "${RED}错误: 需要 -p <checkpoint.pth>${NC}"
    echo "运行 bash $0 -h 查看用法"
    exit 1
fi
if [ ! -f "$PRETRAINED" ]; then
    echo -e "${RED}错误: checkpoint 不存在: $PRETRAINED${NC}"
    exit 1
fi

# ============== 运行函数 ==============
run_exp() {
    local desc="$1"; local cmd="$2"; local log_file="$3"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}实验: ${NC}$desc"
    echo -e "${YELLOW}命令: ${NC}$cmd"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC}"
        echo "$cmd" >> "$LOG_DIR/dry_run_trainfree.txt"
        return 0
    fi
    eval "$cmd" 2>&1 | tee "$log_file"
    local ec=${PIPESTATUS[0]}
    [ $ec -eq 0 ] && echo -e "${GREEN}[完成]${NC}" || echo -e "${RED}[失败]${NC}"
    echo ""
    return $ec
}

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Training-Free Ablation Ladder (S3DIS Area-5, mode=test)    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC}   $GPU_IDS"
echo -e "${GREEN}Ckpt:${NC}  $PRETRAINED"
echo -e "${GREEN}Leaf:${NC}  $LEAF (SA1)"
echo -e "${GREEN}Seed:${NC}  $SEED"
echo ""

TOTAL=0; COMPLETED=0; FAILED=0
COMPLETED_LIST=""; FAILED_LIST=""

mark_result() {
    local desc=$1; local ec=$2
    if [ $ec -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1)); COMPLETED_LIST="$COMPLETED_LIST\n  $desc"
    else
        FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST\n  $desc"
    fi
}

# ============================================================
# A0: 官方 FP32 + 全局 FPS  (基线)
# ============================================================
TOTAL=$((TOTAL + 1))
DESC="A0_official_global_fps_fp32"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
    --cfg $CFG_A0 \
    mode=test \
    --pretrained_path $PRETRAINED \
    seed=$SEED \
    cfg_basename=trainfree_${DESC}"
LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?

# ============================================================
# A1: A0 + KD block-local FPS  (FP32 坐标, 隔离 FPS 局部化的代价)
# ============================================================
TOTAL=$((TOTAL + 1))
DESC="A1_kdtree_fps_fp32_coord"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
    --cfg $CFG_KDTREE \
    mode=test \
    --pretrained_path $PRETRAINED \
    model.encoder_args.sampler_args.leaf_size=$LEAF \
    model.encoder_args.sampler_args.strategy=fps \
    seed=$SEED \
    cfg_basename=trainfree_${DESC}"
LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?

# ============================================================
# A2: A1 + 采样用 int8 坐标  (加上 int8 sampling 影响)
# ============================================================
TOTAL=$((TOTAL + 1))
DESC="A2_sample_int8"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
    --cfg $CFG_KDTREE \
    mode=test \
    --pretrained_path $PRETRAINED \
    model.encoder_args.sampler_args.leaf_size=$LEAF \
    model.encoder_args.sampler_args.strategy=fps \
    model.encoder_args.coord_sample_nbits=8 \
    seed=$SEED \
    cfg_basename=trainfree_${DESC}"
LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?

# ============================================================
# A3: A2 + bq/dp 也 int8  (naive 全 int8, 已知会触发 -3.95 灾难)
# ============================================================
TOTAL=$((TOTAL + 1))
DESC="A3_naive_full_int8_coord"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
    --cfg $CFG_KDTREE \
    mode=test \
    --pretrained_path $PRETRAINED \
    model.encoder_args.sampler_args.leaf_size=$LEAF \
    model.encoder_args.sampler_args.strategy=fps \
    model.encoder_args.coord_sample_nbits=8 \
    model.encoder_args.coord_bq_nbits=8 \
    model.encoder_args.coord_dp_nbits=8 \
    seed=$SEED \
    cfg_basename=trainfree_${DESC}"
LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?

# ============================================================
# A3-safe: A2 + bq/dp 用 9-bit  (我们实验推荐的 sidecar 配置)
# ============================================================
TOTAL=$((TOTAL + 1))
DESC="A3safe_sample8_bqdp9"
CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
    --cfg $CFG_KDTREE \
    mode=test \
    --pretrained_path $PRETRAINED \
    model.encoder_args.sampler_args.leaf_size=$LEAF \
    model.encoder_args.sampler_args.strategy=fps \
    model.encoder_args.coord_sample_nbits=8 \
    model.encoder_args.coord_bq_nbits=9 \
    model.encoder_args.coord_dp_nbits=9 \
    model.encoder_args.coord_dp_postquant_nbits=8 \
    seed=$SEED \
    cfg_basename=trainfree_${DESC}"
LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?

# ============================================================
# 可选: leaf size sweep (FPS 局部化 vs 块粒度的敏感度)
# 对应 paper "leaf size sensitivity" 表
# ============================================================
if [ "$DO_SWEEP" = true ]; then
    echo -e "\n${BLUE}========== 可选: leaf size sensitivity sweep ==========${NC}"
    for L in 256 512 1024 1500 3000 24000; do
        TOTAL=$((TOTAL + 1))
        DESC="sweep_A1_leaf${L}"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py \
            --cfg $CFG_KDTREE \
            mode=test \
            --pretrained_path $PRETRAINED \
            model.encoder_args.sampler_args.leaf_size=$L \
            model.encoder_args.sampler_args.strategy=fps \
            seed=$SEED \
            cfg_basename=trainfree_${DESC}"
        LOG="$LOG_DIR/trainfree_${DESC}_seed${SEED}.log"
        run_exp "$DESC" "$CMD" "$LOG"; mark_result "$DESC" $?
    done
fi

# ============== 汇总 ==============
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Training-Free Ablation 汇总                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC} $TOTAL   ${GREEN}完成:${NC} $COMPLETED   ${RED}失败:${NC} $FAILED"
[ -n "$COMPLETED_LIST" ] && echo -e "${GREEN}完成:${NC}$COMPLETED_LIST"
[ -n "$FAILED_LIST" ] && echo -e "${RED}失败:${NC}$FAILED_LIST"
echo ""
echo -e "${YELLOW}取结果:${NC}"
echo -e "  grep -h 'Best ckpt' $LOG_DIR/trainfree_*.log"
echo ""
echo -e "${YELLOW}预期解读(基于我们之前的实验):${NC}"
echo "  A0 -> A1 的差值 = FPS 局部化的真实代价 (这就是 paper 缺的那个数 [Z])"
echo "  A1 -> A2 的差值 = int8 坐标在 FPS 阶段的影响 (我们之前测过 ~0.01, 应免费)"
echo "  A2 -> A3 的差值 = bq+dp 全 int8 的灾难项 (我们测过 ~-3.95, 这是 naive 路线)"
echo "  A2 -> A3-safe 的差值 = 推荐 sidecar 配置 (bq/dp 9-bit), 应接近 A2"
echo "  A3-safe 就是论文里 [A] (KDPoint 最终配置, 想宣传的"免 retrain"数)"
echo ""
echo -e "${YELLOW}注意:${NC} 这个 ladder 只测【坐标量化栈】, 没含【权重/激活 INT8 PTQ】."
echo "      完整 paper [A] 还要叠加: cd quant/ 跑 quant_wrapper.py 的 PTQ 流程"
echo "      把权重 per-channel int8、激活 per-tensor uint8 一起 convert 上去."
