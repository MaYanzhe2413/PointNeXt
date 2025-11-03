#!/bin/bash

# 快速训练脚本 - 预设常用训练组合

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_menu() {
    echo -e "${BLUE}========== PointNeXt 快速训练菜单 ==========${NC}"
    echo ""
    echo -e "${GREEN}分类任务:${NC}"
    echo "  1) PointNeXt-S + ModelNet40"
    echo "  2) PointNeXt-S + ScanObjectNN"
    echo "  3) PointNet++ + ModelNet40"
    echo "  4) PointNet++ + ScanObjectNN"
    echo ""
    echo -e "${GREEN}分割任务:${NC}"
    echo "  5) PointNeXt-S + S3DIS"
    echo "  6) PointNeXt-B + S3DIS"
    echo "  7) PointNeXt-L + S3DIS"
    echo "  8) PointNet++ + S3DIS"
    echo ""
    echo -e "${GREEN}部件分割:${NC}"
    echo "  9) PointNeXt-S + ShapeNetPart"
    echo ""
    echo -e "${YELLOW}其他选项:${NC}"
    echo "  10) 自定义训练 (调用完整脚本)"
    echo "  0) 退出"
    echo ""
    echo -e "${BLUE}============================================${NC}"
}

# 获取GPU设置
get_gpu_setting() {
    echo -e "${YELLOW}GPU设置:${NC}"
    echo "1) 使用GPU 0 (默认)"
    echo "2) 使用所有GPU"
    echo "3) 自定义GPU编号"
    read -p "请选择GPU设置 (1-3): " gpu_choice
    
    case $gpu_choice in
        1|"")
            echo "0"
            ;;
        2)
            echo "all"
            ;;
        3)
            read -p "请输入GPU编号 (例如: 0,1,2): " custom_gpu
            echo "$custom_gpu"
            ;;
        *)
            echo "0"
            ;;
    esac
}

# 获取额外参数
get_additional_args() {
    echo -e "${YELLOW}是否需要添加额外参数? (y/N):${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}常用参数:${NC}"
        echo "  wandb.use_wandb=True     - 启用wandb日志"
        echo "  epochs=300               - 设置训练轮数"
        echo "  batch_size=32            - 设置批次大小"
        echo ""
        read -p "请输入额外参数: " additional_args
        echo "$additional_args"
    else
        echo ""
    fi
}

# 主循环
while true; do
    show_menu
    read -p "请选择训练选项 (0-10): " choice
    
    case $choice in
        1)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh classification pointnext-s modelnet40 "$gpu_ids" "$additional_args"
            ;;
        2)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh classification pointnext-s scanobjectnn "$gpu_ids" "$additional_args"
            ;;
        3)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh classification pointnet++ modelnet40 "$gpu_ids" "$additional_args"
            ;;
        4)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh classification pointnet++ scanobjectnn "$gpu_ids" "$additional_args"
            ;;
        5)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh segmentation pointnext-s s3dis "$gpu_ids" "$additional_args"
            ;;
        6)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh segmentation pointnext-b s3dis "$gpu_ids" "$additional_args"
            ;;
        7)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh segmentation pointnext-l s3dis "$gpu_ids" "$additional_args"
            ;;
        8)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh segmentation pointnet++ s3dis "$gpu_ids" "$additional_args"
            ;;
        9)
            gpu_ids=$(get_gpu_setting)
            additional_args=$(get_additional_args)
            ./run_training.sh partseg pointnext-s shapenetpart "$gpu_ids" "$additional_args"
            ;;
        10)
            echo -e "${YELLOW}请直接使用 ./run_training.sh 命令${NC}"
            echo -e "${YELLOW}使用方法: ./run_training.sh [task] [model] [dataset] [gpu_ids] [additional_args]${NC}"
            ./run_training.sh --help
            ;;
        0)
            echo -e "${GREEN}再见!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择，请重新选择${NC}"
            ;;
    esac
    
    echo ""
    read -p "按任意键继续..." -n 1
    echo ""
done
