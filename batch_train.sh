#!/bin/bash

# 批量训练脚本 - 用于运行多个实验

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 帮助信息
show_help() {
    echo -e "${BLUE}批量训练脚本使用说明${NC}"
    echo -e "${GREEN}使用方法:${NC} ./batch_train.sh [experiment_type]"
    echo ""
    echo -e "${YELLOW}实验类型:${NC}"
    echo "  compare_models    : 比较不同模型在同一数据集上的性能"
    echo "  compare_datasets  : 比较同一模型在不同数据集上的性能"
    echo "  ablation_study    : 消融实验"
    echo "  custom           : 自定义批量实验"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./batch_train.sh compare_models"
    echo "  ./batch_train.sh compare_datasets"
}

# 比较模型实验
compare_models_experiment() {
    echo -e "${BLUE}========== 模型比较实验 ==========${NC}"
    echo -e "${YELLOW}选择数据集:${NC}"
    echo "1) ModelNet40 (分类)"
    echo "2) ScanObjectNN (分类)" 
    echo "3) S3DIS (分割)"
    read -p "请选择数据集 (1-3): " dataset_choice
    
    case $dataset_choice in
        1)
            DATASET="modelnet40"
            TASK="classification"
            MODELS=("pointnext-s" "pointnet++" "pointmlp" "dgcnn")
            ;;
        2)
            DATASET="scanobjectnn"
            TASK="classification"
            MODELS=("pointnext-s" "pointnet++" "pointmlp" "dgcnn")
            ;;
        3)
            DATASET="s3dis"
            TASK="segmentation"
            MODELS=("pointnext-s" "pointnext-b" "pointnet++" "dgcnn")
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            return 1
            ;;
    esac
    
    read -p "请输入GPU编号 (默认为0): " gpu_ids
    gpu_ids=${gpu_ids:-"0"}
    
    read -p "是否启用wandb? (y/N): " enable_wandb
    if [[ $enable_wandb =~ ^[Yy]$ ]]; then
        WANDB_ARG="wandb.use_wandb=True"
    else
        WANDB_ARG=""
    fi
    
    echo -e "${GREEN}开始模型比较实验...${NC}"
    echo -e "${GREEN}数据集: $DATASET${NC}"
    echo -e "${GREEN}模型: ${MODELS[*]}${NC}"
    
    for model in "${MODELS[@]}"; do
        echo -e "${BLUE}开始训练: $model${NC}"
        ./run_training.sh $TASK $model $DATASET $gpu_ids "$WANDB_ARG wandb.name=${model}_${DATASET}"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}模型 $model 训练失败${NC}"
        else
            echo -e "${GREEN}模型 $model 训练完成${NC}"
        fi
        echo ""
    done
    
    echo -e "${GREEN}所有模型训练完成!${NC}"
}

# 比较数据集实验
compare_datasets_experiment() {
    echo -e "${BLUE}========== 数据集比较实验 ==========${NC}"
    echo -e "${YELLOW}选择模型:${NC}"
    echo "1) PointNeXt-S"
    echo "2) PointNet++"
    read -p "请选择模型 (1-2): " model_choice
    
    case $model_choice in
        1)
            MODEL="pointnext-s"
            ;;
        2)
            MODEL="pointnet++"
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            return 1
            ;;
    esac
    
    read -p "请输入GPU编号 (默认为0): " gpu_ids
    gpu_ids=${gpu_ids:-"0"}
    
    read -p "是否启用wandb? (y/N): " enable_wandb
    if [[ $enable_wandb =~ ^[Yy]$ ]]; then
        WANDB_ARG="wandb.use_wandb=True"
    else
        WANDB_ARG=""
    fi
    
    # 分类数据集
    CLASSIFICATION_DATASETS=("modelnet40" "scanobjectnn")
    
    echo -e "${GREEN}开始数据集比较实验...${NC}"
    echo -e "${GREEN}模型: $MODEL${NC}"
    
    # 训练分类任务
    for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
        echo -e "${BLUE}开始训练分类任务: $dataset${NC}"
        ./run_training.sh classification $MODEL $dataset $gpu_ids "$WANDB_ARG wandb.name=${MODEL}_${dataset}"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}数据集 $dataset 训练失败${NC}"
        else
            echo -e "${GREEN}数据集 $dataset 训练完成${NC}"
        fi
        echo ""
    done
    
    # 训练分割任务
    echo -e "${BLUE}开始训练分割任务: S3DIS${NC}"
    ./run_training.sh segmentation $MODEL s3dis $gpu_ids "$WANDB_ARG wandb.name=${MODEL}_s3dis"
    
    echo -e "${GREEN}所有数据集训练完成!${NC}"
}

# 消融实验
ablation_study() {
    echo -e "${BLUE}========== 消融实验 ==========${NC}"
    echo -e "${YELLOW}选择消融实验类型:${NC}"
    echo "1) 不同epochs数量对比"
    echo "2) 不同batch size对比"
    echo "3) 自定义消融实验"
    read -p "请选择实验类型 (1-3): " ablation_choice
    
    echo -e "${YELLOW}选择基础配置:${NC}"
    echo "1) PointNeXt-S + ModelNet40"
    echo "2) PointNeXt-S + S3DIS"
    read -p "请选择基础配置 (1-2): " base_choice
    
    case $base_choice in
        1)
            TASK="classification"
            MODEL="pointnext-s"
            DATASET="modelnet40"
            ;;
        2)
            TASK="segmentation"
            MODEL="pointnext-s"
            DATASET="s3dis"
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            return 1
            ;;
    esac
    
    read -p "请输入GPU编号 (默认为0): " gpu_ids
    gpu_ids=${gpu_ids:-"0"}
    
    case $ablation_choice in
        1)
            EPOCHS_LIST=(100 200 300 500)
            for epochs in "${EPOCHS_LIST[@]}"; do
                echo -e "${BLUE}训练 epochs=$epochs${NC}"
                ./run_training.sh $TASK $MODEL $DATASET $gpu_ids "epochs=$epochs wandb.use_wandb=True wandb.name=${MODEL}_${DATASET}_epochs${epochs}"
            done
            ;;
        2)
            BATCH_SIZES=(8 16 32 64)
            for batch_size in "${BATCH_SIZES[@]}"; do
                echo -e "${BLUE}训练 batch_size=$batch_size${NC}"
                ./run_training.sh $TASK $MODEL $DATASET $gpu_ids "batch_size=$batch_size wandb.use_wandb=True wandb.name=${MODEL}_${DATASET}_bs${batch_size}"
            done
            ;;
        3)
            echo -e "${YELLOW}请手动编辑此脚本添加自定义消融实验${NC}"
            ;;
    esac
}

# 自定义批量实验
custom_batch() {
    echo -e "${BLUE}========== 自定义批量实验 ==========${NC}"
    echo -e "${YELLOW}请编辑 experiments.txt 文件，每行一个实验配置${NC}"
    echo -e "${YELLOW}格式: task model dataset gpu_ids additional_args${NC}"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "classification pointnext-s modelnet40 0 epochs=300"
    echo "segmentation pointnext-s s3dis 0 wandb.use_wandb=True"
    echo ""
    
    if [ ! -f "experiments.txt" ]; then
        cat > experiments.txt << EOF
# 实验配置文件
# 格式: task model dataset gpu_ids additional_args
# 示例:
# classification pointnext-s modelnet40 0 epochs=300
# segmentation pointnext-s s3dis 0 wandb.use_wandb=True

EOF
        echo -e "${GREEN}已创建 experiments.txt 模板文件${NC}"
    fi
    
    read -p "请编辑 experiments.txt 文件后按回车继续..." 
    
    if [ -f "experiments.txt" ]; then
        echo -e "${GREEN}开始执行批量实验...${NC}"
        while IFS= read -r line || [ -n "$line" ]; do
            # 跳过注释和空行
            if [[ $line =~ ^#.*$ ]] || [[ -z "$line" ]]; then
                continue
            fi
            
            echo -e "${BLUE}执行实验: $line${NC}"
            ./run_training.sh $line
            echo ""
        done < experiments.txt
        echo -e "${GREEN}所有实验完成!${NC}"
    else
        echo -e "${RED}找不到 experiments.txt 文件${NC}"
    fi
}

# 主程序
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 1
fi

case $1 in
    "compare_models")
        compare_models_experiment
        ;;
    "compare_datasets")
        compare_datasets_experiment
        ;;
    "ablation_study")
        ablation_study
        ;;
    "custom")
        custom_batch
        ;;
    *)
        echo -e "${RED}错误: 不支持的实验类型 '$1'${NC}"
        show_help
        exit 1
        ;;
esac
