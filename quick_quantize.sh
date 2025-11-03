#!/bin/bash

# PointNeXt 快速量化脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 PointNeXt 快速量化工具${NC}"
echo ""

# 显示菜单
show_quantization_menu() {
    echo -e "${BLUE}========== 选择要量化的模型 ==========${NC}"
    echo ""
    echo -e "${GREEN}分类模型 (静态量化):${NC}"
    echo "  1) PointNeXt-S + ModelNet40"
    echo "  2) PointNet++ + ModelNet40" 
    echo "  3) PointMLP + ModelNet40"
    echo ""
    echo -e "${GREEN}分割模型 (静态量化):${NC}"
    echo "  4) PointNeXt-S + S3DIS"
    echo "  5) PointNeXt-B + S3DIS"
    echo "  6) PointNet++ + S3DIS"
    echo ""
    echo -e "${YELLOW}QAT量化:${NC}"
    echo "  7) QAT - PointNeXt-S + ModelNet40"
    echo "  8) QAT - PointNeXt-S + S3DIS"
    echo "  9) 对比静态量化和QAT"
    echo ""
    echo -e "${YELLOW}其他选项:${NC}"
    echo "  10) 自定义配置文件"
    echo "  0) 退出"
    echo ""
    echo -e "${BLUE}========================================${NC}"
}

# 获取配置文件路径
get_config_path() {
    case $1 in
        1) echo "cfgs/modelnet40ply2048/pointnext-s.yaml" ;;
        2) echo "cfgs/modelnet40ply2048/pointnet++.yaml" ;;
        3) echo "cfgs/modelnet40ply2048/pointmlp.yaml" ;;
        4) echo "cfgs/s3dis/pointnext-s.yaml" ;;
        5) echo "cfgs/s3dis/pointnext-b.yaml" ;;
        6) echo "cfgs/s3dis/pointnet++.yaml" ;;
        *) echo "" ;;
    esac
}

# 运行量化
run_quantization() {
    local config_file=$1
    local model_name=$2
    local method=${3:-"static"}  # 默认静态量化
    
    echo -e "${BLUE}🔥 开始量化: $model_name${NC}"
    echo -e "${YELLOW}配置文件: $config_file${NC}"
    echo -e "${YELLOW}量化方法: $method${NC}"
    
    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}❌ 配置文件不存在: $config_file${NC}"
        return 1
    fi
    
    # 询问是否使用预训练模型
    echo ""
    read -p "是否有预训练模型? (y/N): " -n 1 -r
    echo
    
    local pretrained_arg=""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入预训练模型路径: " pretrained_path
        if [ -f "$pretrained_path" ]; then
            pretrained_arg="--pretrained $pretrained_path"
        else
            echo -e "${YELLOW}⚠️  预训练模型不存在，使用随机初始化模型${NC}"
        fi
    fi
    
    # 如果是QAT，询问训练轮数
    local epochs_arg=""
    if [ "$method" = "qat" ]; then
        read -p "请输入QAT训练轮数 (默认3): " epochs
        epochs=${epochs:-3}
        epochs_arg="--epochs $epochs"
    fi
    
    # 设置输出路径
    local model_basename=$(basename "$config_file" .yaml)
    local output_path="quantized_models/${model_basename}_${method}_quantized.pth"
    
    echo ""
    echo -e "${BLUE}开始量化...${NC}"
    echo -e "${YELLOW}量化方法: $method${NC}"
    echo -e "${YELLOW}输出路径: $output_path${NC}"
    if [ "$method" = "qat" ]; then
        echo -e "${YELLOW}训练轮数: $epochs${NC}"
    fi
    echo ""
    
    # 运行量化命令
    if python quantize_fx.py --cfg "$config_file" --method "$method" $pretrained_arg $epochs_arg --save_path "$output_path"; then
        echo ""
        echo -e "${GREEN}🎉 量化成功完成!${NC}"
        echo -e "${GREEN}📁 量化模型保存在: $output_path${NC}"
        
        # 显示模型信息
        if [ -f "$output_path" ]; then
            local file_size=$(du -h "$output_path" | cut -f1)
            echo -e "${BLUE}📊 量化模型大小: $file_size${NC}"
        fi
    else
        echo ""
        echo -e "${RED}❌ 量化失败!${NC}"
    fi
}

# 主循环
while true; do
    show_quantization_menu
    read -p "请选择要量化的模型 (0-10): " choice
    
    case $choice in
        1)
            run_quantization "cfgs/modelnet40ply2048/pointnext-s.yaml" "PointNeXt-S (ModelNet40)" "static"
            ;;
        2)
            run_quantization "cfgs/modelnet40ply2048/pointnet++.yaml" "PointNet++ (ModelNet40)" "static"
            ;;
        3)
            run_quantization "cfgs/modelnet40ply2048/pointmlp.yaml" "PointMLP (ModelNet40)" "static"
            ;;
        4)
            run_quantization "cfgs/s3dis/pointnext-s.yaml" "PointNeXt-S (S3DIS)" "static"
            ;;
        5)
            run_quantization "cfgs/s3dis/pointnext-b.yaml" "PointNeXt-B (S3DIS)" "static"
            ;;
        6)
            run_quantization "cfgs/s3dis/pointnet++.yaml" "PointNet++ (S3DIS)" "static"
            ;;
        7)
            run_quantization "cfgs/modelnet40ply2048/pointnext-s.yaml" "PointNeXt-S (ModelNet40) QAT" "qat"
            ;;
        8)
            run_quantization "cfgs/s3dis/pointnext-s.yaml" "PointNeXt-S (S3DIS) QAT" "qat"
            ;;
        9)
            run_quantization "cfgs/modelnet40ply2048/pointnext-s.yaml" "PointNeXt-S 对比量化" "compare"
            ;;
        10)
            echo ""
            read -p "请输入配置文件路径: " custom_config
            if [ -f "$custom_config" ]; then
                echo ""
                echo "选择量化方法:"
                echo "1) 静态量化"
                echo "2) QAT量化"
                echo "3) 对比两种方法"
                read -p "请选择 (1-3): " method_choice
                
                case $method_choice in
                    1) method="static" ;;
                    2) method="qat" ;;
                    3) method="compare" ;;
                    *) method="static" ;;
                esac
                
                local custom_name=$(basename "$custom_config" .yaml)
                run_quantization "$custom_config" "自定义模型 ($custom_name)" "$method"
            else
                echo -e "${RED}❌ 配置文件不存在: $custom_config${NC}"
            fi
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
