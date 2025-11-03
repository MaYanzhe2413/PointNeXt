#!/bin/bash

# PointNeXt 训练脚本
# 使用方法: ./run_training.sh [task] [model] [dataset] [gpu_ids] [additional_args]

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo -e "${BLUE}PointNeXt 训练脚本使用说明${NC}"
    echo -e "${GREEN}使用方法:${NC} ./run_training.sh [task] [model] [dataset] [gpu_ids] [additional_args]"
    echo ""
    echo -e "${YELLOW}参数说明:${NC}"
    echo "  task     : 任务类型 [classification|segmentation|partseg]"
    echo "  model    : 模型名称 [pointnext-s|pointnext-b|pointnext-l|pointnet++|pointmlp|dgcnn]"
    echo "  dataset  : 数据集名称 [modelnet40|scanobjectnn|s3dis|shapenetpart]"
    echo "  gpu_ids  : GPU编号 (可选, 默认为0, 多GPU用逗号分隔如0,1,2,3)"
    echo "  additional_args : 额外参数 (可选)"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./run_training.sh classification pointnext-s modelnet40"
    echo "  ./run_training.sh segmentation pointnext-s s3dis 0"
    echo "  ./run_training.sh classification pointnet++ scanobjectnn 0,1"
    echo "  ./run_training.sh segmentation pointnext-b s3dis 0 \"epochs=300 batch_size=16\""
    echo ""
    echo -e "${YELLOW}支持的组合:${NC}"
    echo -e "${GREEN}分类任务:${NC}"
    echo "  - ModelNet40: pointnext-s, pointnet++, pointmlp, dgcnn"
    echo "  - ScanObjectNN: pointnext-s, pointnet++, pointmlp, dgcnn"
    echo -e "${GREEN}分割任务:${NC}"
    echo "  - S3DIS: pointnext-s, pointnext-b, pointnext-l, pointnet++, dgcnn"
    echo -e "${GREEN}部件分割:${NC}"
    echo "  - ShapeNetPart: pointnext-s, pointnet++"
    echo ""
    echo -e "${YELLOW}常用额外参数:${NC}"
    echo "  wandb.use_wandb=True     : 启用wandb日志"
    echo "  epochs=300               : 设置训练轮数"
    echo "  batch_size=32            : 设置批次大小"
    echo "  mode=test                : 测试模式"
    echo "  --pretrained_path=path   : 预训练模型路径"
}

# 检查参数
if [ $# -lt 3 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 1
fi

# 解析参数
TASK=$1
MODEL=$2
DATASET=$3
GPU_IDS=${4:-"0"}
ADDITIONAL_ARGS=${5:-""}

# 验证任务类型
case $TASK in
    "classification"|"cls")
        TASK_FOLDER="classification"
        ;;
    "segmentation"|"seg")
        TASK_FOLDER="segmentation"
        ;;
    "partseg"|"part")
        TASK_FOLDER="shapenetpart"
        ;;
    *)
        echo -e "${RED}错误: 不支持的任务类型 '$TASK'${NC}"
        echo -e "${YELLOW}支持的任务类型: classification, segmentation, partseg${NC}"
        exit 1
        ;;
esac

# 根据任务和数据集确定配置文件
get_config_path() {
    local task=$1
    local model=$2
    local dataset=$3
    
    case $task in
        "classification")
            case $dataset in
                "modelnet40")
                    echo "cfgs/modelnet40ply2048/${model}.yaml"
                    ;;
                "scanobjectnn")
                    echo "cfgs/scanobjectnn/${model}.yaml"
                    ;;
                *)
                    echo ""
                    ;;
            esac
            ;;
        "segmentation")
            case $dataset in
                "s3dis")
                    echo "cfgs/s3dis/${model}.yaml"
                    ;;
                "scannet")
                    echo "cfgs/scannet/${model}.yaml"
                    ;;
                *)
                    echo ""
                    ;;
            esac
            ;;
        "partseg")
            case $dataset in
                "shapenetpart")
                    echo "cfgs/shapenetpart/${model}.yaml"
                    ;;
                *)
                    echo ""
                    ;;
            esac
            ;;
    esac
}

# 获取配置文件路径
CONFIG_PATH=$(get_config_path $TASK $MODEL $DATASET)

if [ -z "$CONFIG_PATH" ]; then
    echo -e "${RED}错误: 不支持的模型-数据集组合: $MODEL + $DATASET${NC}"
    show_help
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_PATH${NC}"
    echo -e "${YELLOW}请检查模型名称和数据集名称是否正确${NC}"
    exit 1
fi

# 构建完整命令
if [ "$GPU_IDS" == "all" ]; then
    # 使用所有GPU
    FULL_COMMAND="python examples/${TASK_FOLDER}/main.py --cfg $CONFIG_PATH $ADDITIONAL_ARGS"
else
    # 使用指定GPU
    FULL_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_IDS python examples/${TASK_FOLDER}/main.py --cfg $CONFIG_PATH $ADDITIONAL_ARGS"
fi

# 显示运行信息
echo -e "${BLUE}========== PointNeXt 训练开始 ==========${NC}"
echo -e "${GREEN}任务类型:${NC} $TASK"
echo -e "${GREEN}模型:${NC} $MODEL"
echo -e "${GREEN}数据集:${NC} $DATASET"
echo -e "${GREEN}GPU:${NC} $GPU_IDS"
echo -e "${GREEN}配置文件:${NC} $CONFIG_PATH"
echo -e "${GREEN}额外参数:${NC} $ADDITIONAL_ARGS"
echo -e "${YELLOW}执行命令:${NC} $FULL_COMMAND"
echo -e "${BLUE}=======================================${NC}"
echo ""

# 确认是否继续
read -p "是否继续执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# 检查环境
echo -e "${BLUE}检查环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 找不到Python${NC}"
    exit 1
fi

if ! python -c "import torch" &> /dev/null; then
    echo -e "${RED}错误: 找不到PyTorch${NC}"
    exit 1
fi

# 创建日志目录
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_${TASK}_${MODEL}_${DATASET}"
mkdir -p $LOG_DIR

echo -e "${GREEN}环境检查通过${NC}"
echo -e "${GREEN}日志目录:${NC} $LOG_DIR"
echo ""

# 执行训练命令
echo -e "${BLUE}开始训练...${NC}"
eval $FULL_COMMAND 2>&1 | tee $LOG_DIR/training.log

# 检查训练结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}========== 训练完成 ==========${NC}"
    echo -e "${GREEN}日志文件: $LOG_DIR/training.log${NC}"
else
    echo -e "${RED}========== 训练失败 ==========${NC}"
    echo -e "${RED}请检查日志文件: $LOG_DIR/training.log${NC}"
    exit 1
fi
