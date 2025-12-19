#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo -e "${BLUE}PointNeXt 训练脚本${NC}"
    echo -e "${GREEN}用法:${NC} ./run_training.sh task model dataset [选项]"
    echo ""
    echo -e "${YELLOW}位置参数:${NC}"
    echo "  task      任务类型 [classification|segmentation|partseg]"
    echo "  model     模型 [pointnext-s_kdtree_simple|pointnext-s_kdtree|...]"
    echo "  dataset   数据集 [modelnet40|scanobjectnn|s3dis|shapenetpart]"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  -g GPU      GPU编号 (默认0)"
    echo "  -s STRAT    策略 [random|uniform|center_random|quad_fps]"
    echo "  -l SIZE     leaf_size (默认32)"
    echo "  -e EPOCHS   训练轮数"
    echo "  -b SIZE     批次大小"
    echo "  --wandb     启用wandb"
    echo "  --test      测试模式"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./run_training.sh classification pointnext-s_kdtree_simple modelnet40 -s quad_fps -l 64 -g 0"
}

if [ $# -lt 3 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

TASK=$1
MODEL=$2
DATASET=$3
shift 3

GPU_IDS="0"
STRATEGY=""
LEAF_SIZE=""
EPOCHS=""
BATCH_SIZE=""
WANDB=""
TEST_MODE=""

while [ $# -gt 0 ]; do
    case $1 in
        -g) GPU_IDS="$2"; shift 2 ;;
        -s) STRATEGY="$2"; shift 2 ;;
        -l) LEAF_SIZE="$2"; shift 2 ;;
        -e) EPOCHS="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        --wandb) WANDB="true"; shift ;;
        --test) TEST_MODE="true"; shift ;;
        *) echo -e "${RED}未知选项: $1${NC}"; exit 1 ;;
    esac
done

case $TASK in
    classification|cls) TASK_FOLDER="classification" ;;
    segmentation|seg) TASK_FOLDER="segmentation" ;;
    partseg|part) TASK_FOLDER="shapenetpart" ;;
    *) echo -e "${RED}未知任务: $TASK${NC}"; exit 1 ;;
esac

case $TASK in
    classification)
        case $DATASET in
            modelnet40) CONFIG_PATH="cfgs/modelnet40ply2048/${MODEL}.yaml" ;;
            scanobjectnn) CONFIG_PATH="cfgs/scanobjectnn/${MODEL}.yaml" ;;
            *) CONFIG_PATH="" ;;
        esac ;;
    segmentation)
        case $DATASET in
            s3dis) CONFIG_PATH="cfgs/s3dis/${MODEL}.yaml" ;;
            scannet) CONFIG_PATH="cfgs/scannet/${MODEL}.yaml" ;;
            *) CONFIG_PATH="" ;;
        esac ;;
    partseg)
        case $DATASET in
            shapenetpart) CONFIG_PATH="cfgs/shapenetpart/${MODEL}.yaml" ;;
            *) CONFIG_PATH="" ;;
        esac ;;
esac

if [ -z "$CONFIG_PATH" ] || [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}配置文件不存在: $CONFIG_PATH${NC}"
    exit 1
fi

ARGS=""
[ -n "$STRATEGY" ] && ARGS="$ARGS model.encoder_args.sampler_args.strategy=$STRATEGY"
[ -n "$LEAF_SIZE" ] && ARGS="$ARGS model.encoder_args.sampler_args.leaf_size=$LEAF_SIZE"
[ -n "$EPOCHS" ] && ARGS="$ARGS epochs=$EPOCHS"
[ -n "$BATCH_SIZE" ] && ARGS="$ARGS batch_size=$BATCH_SIZE"
[ "$WANDB" == "true" ] && ARGS="$ARGS wandb.use_wandb=True"
[ "$TEST_MODE" == "true" ] && ARGS="$ARGS mode=test"

COMMAND="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/${TASK_FOLDER}/main.py --cfg $CONFIG_PATH $ARGS"

echo -e "${BLUE}========== 训练配置 ==========${NC}"
echo -e "${GREEN}任务:${NC} $TASK"
echo -e "${GREEN}模型:${NC} $MODEL"
echo -e "${GREEN}数据集:${NC} $DATASET"
echo -e "${GREEN}GPU:${NC} $GPU_IDS"
[ -n "$STRATEGY" ] && echo -e "${GREEN}策略:${NC} $STRATEGY"
[ -n "$LEAF_SIZE" ] && echo -e "${GREEN}leaf_size:${NC} $LEAF_SIZE"
echo -e "${YELLOW}命令:${NC} $COMMAND"
echo -e "${BLUE}================================${NC}"
echo ""

read -p "继续? (y/N): " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

eval $COMMAND
