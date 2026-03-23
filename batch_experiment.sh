#!/bin/bash
# ============================================================================
# 批量实验脚本: KD-Tree Sampling 跨数据集对比
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

# ============== 实验定义 ==============
# 格式: "TASK|CFG_BASE|CFG_KDTREE|DATASET|LEAF_SIZES|ENTRY_POINT"
#
# 各数据集点数:
#   S3DIS:      voxel_max=24000
#   ScanNet:    voxel_max=64000
#   ShapeNet:   num_points=2048
#   ModelNet40: num_points=1024

declare -a EXPERIMENTS=(
    # ===== S3DIS (已有大量结果, 补充 random 策略) =====
    "segmentation|cfgs/s3dis/pointnext-s.yaml|cfgs/s3dis/pointnext-s_kdtree84fps.yaml|s3dis|325,750,1500,3000|examples/segmentation/main.py"

    # ===== ScanNet =====
    "segmentation|cfgs/scannet/pointnext-s.yaml|cfgs/scannet/pointnext-s_kdtree.yaml|scannet|500,2000,4000,8000,16000,32000,64000|examples/segmentation/main.py"

    # ===== ShapeNetPart =====
    "partseg|cfgs/shapenetpart/pointnext-s.yaml|cfgs/shapenetpart/pointnext-s_kdtree.yaml|shapenetpart|64,128,256,512,1024,2048|examples/shapenetpart/main.py"

    # ===== ModelNet40 =====
    "classification|cfgs/modelnet40ply2048/pointnext-s.yaml|cfgs/modelnet40ply2048/pointnext-s_kdtree_fps.yaml|modelnet40|32,64,128,256,512,1024|examples/classification/main.py"
)

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
echo -e "${BLUE}║          批量 KD-Tree Sampling 实验             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}GPU:${NC}  $GPU_IDS"
echo -e "${GREEN}Seed:${NC} $SEED"
echo -e "${GREEN}模式:${NC} $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo '实际运行')"
echo ""

TOTAL=0
COMPLETED=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TASK CFG_BASE CFG_KDTREE DATASET LEAF_SIZES ENTRY_POINT <<< "$exp"

    echo -e "\n${BLUE}========== $DATASET ($TASK) ==========${NC}"

    # ----- 1. Baseline (原始 FPS) -----
    TOTAL=$((TOTAL + 1))
    DESC="${DATASET}_baseline_fps"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python $ENTRY_POINT --cfg $CFG_BASE seed=$SEED"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"

    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    [ $? -eq 0 ] && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))

    # ----- 2. KD-Tree + FPS (各 leaf_size) -----
    IFS=',' read -ra SIZES <<< "$LEAF_SIZES"
    for LS in "${SIZES[@]}"; do
        TOTAL=$((TOTAL + 1))
        DESC="${DATASET}_kd_fps_leaf${LS}"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python $ENTRY_POINT --cfg $CFG_KDTREE model.encoder_args.sampler_args.leaf_size=$LS model.encoder_args.sampler_args.strategy=fps seed=$SEED"
        LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"

        run_experiment "$DESC" "$CMD" "$LOG_FILE"
        [ $? -eq 0 ] && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))
    done

    # ----- 3. KD-Tree + Random (选取中间 leaf_size) -----
    # 选择 leaf_sizes 列表中间的两个值做 random 对比
    NUM_SIZES=${#SIZES[@]}
    MID1=$((NUM_SIZES / 3))
    MID2=$((NUM_SIZES * 2 / 3))
    RANDOM_SIZES="${SIZES[$MID1]},${SIZES[$MID2]}"

    IFS=',' read -ra RSIZES <<< "$RANDOM_SIZES"
    for LS in "${RSIZES[@]}"; do
        TOTAL=$((TOTAL + 1))
        DESC="${DATASET}_kd_random_leaf${LS}"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python $ENTRY_POINT --cfg $CFG_KDTREE model.encoder_args.sampler_args.leaf_size=$LS model.encoder_args.sampler_args.strategy=random seed=$SEED"
        LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"

        run_experiment "$DESC" "$CMD" "$LOG_FILE"
        [ $? -eq 0 ] && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))
    done
done

# ===== S3DIS random 补充实验 =====
echo -e "\n${BLUE}========== S3DIS random 策略补充 ==========${NC}"
S3DIS_RANDOM_SIZES="325,750,1500,3000"
IFS=',' read -ra S3DIS_RS <<< "$S3DIS_RANDOM_SIZES"
for LS in "${S3DIS_RS[@]}"; do
    TOTAL=$((TOTAL + 1))
    DESC="s3dis_kd_random_leaf${LS}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/segmentation/main.py --cfg cfgs/s3dis/pointnext-s_kdtree84fps.yaml model.encoder_args.sampler_args.leaf_size=$LS model.encoder_args.sampler_args.strategy=random seed=$SEED"
    LOG_FILE="$LOG_DIR/${DESC}_seed${SEED}.log"

    run_experiment "$DESC" "$CMD" "$LOG_FILE"
    [ $? -eq 0 ] && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))
done

# ============== 汇总 ==============
echo -e "\n${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  实验汇总                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}总计:${NC}   $TOTAL"
echo -e "${GREEN}完成:${NC}   $COMPLETED"
echo -e "${RED}失败:${NC}   $FAILED"
echo ""
echo -e "${YELLOW}日志目录: $LOG_DIR/${NC}"
echo -e "${YELLOW}提取结果: grep -h 'best val miou\|best acc' $LOG_DIR/*.log${NC}"
