#!/bin/bash

# PointNeXt QAT量化示例脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎯 PointNeXt QAT量化示例${NC}"
echo ""

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch未安装${NC}"
    exit 1
fi

if ! python -c "import torch.fx" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch FX不可用，需要PyTorch 1.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python环境检查通过${NC}"
echo ""

echo -e "${BLUE}🔥 QAT量化使用示例${NC}"
echo ""

# 示例1: 静态量化
echo -e "${YELLOW}示例1: 静态量化 (Post-Training Quantization)${NC}"
echo -e "${GREEN}命令:${NC}"
echo "python quantize_fx.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --method static"
echo ""

# 示例2: QAT量化
echo -e "${YELLOW}示例2: QAT量化 (Quantization Aware Training)${NC}"
echo -e "${GREEN}命令:${NC}"
echo "python quantize_fx.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --method qat --epochs 5"
echo ""

# 示例3: 对比两种方法
echo -e "${YELLOW}示例3: 对比静态量化和QAT量化${NC}"
echo -e "${GREEN}命令:${NC}"
echo "python quantize_fx.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --method compare"
echo ""

# 示例4: 使用预训练模型进行QAT
echo -e "${YELLOW}示例4: 使用预训练模型进行QAT量化${NC}"
echo -e "${GREEN}命令:${NC}"
echo "python quantize_fx.py \\"
echo "    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \\"
echo "    --method qat \\"
echo "    --pretrained /path/to/pretrained_model.pth \\"
echo "    --epochs 3 \\"
echo "    --save_path quantized_models/pointnext_qat.pth"
echo ""

echo -e "${BLUE}📊 量化方法对比:${NC}"
echo ""
echo -e "${GREEN}静态量化 (Static/PTQ):${NC}"
echo "  ✅ 优点: 快速、不需要训练、适合已训练好的模型"
echo "  ❌ 缺点: 精度可能略有损失"
echo "  🎯 适用: 模型已经训练完成，需要快速部署"
echo ""

echo -e "${GREEN}QAT量化 (Quantization Aware Training):${NC}"
echo "  ✅ 优点: 精度损失最小、量化效果最佳"
echo "  ❌ 缺点: 需要训练过程、耗时较长"
echo "  🎯 适用: 对精度要求高、可以进行训练的场景"
echo ""

echo -e "${YELLOW}推荐使用策略:${NC}"
echo "1. 🚀 先尝试静态量化，快速验证效果"
echo "2. 📈 如果精度损失可接受，直接使用静态量化"
echo "3. 🎯 如果需要更高精度，使用QAT量化"
echo "4. 🔬 使用compare模式对比两种方法"
echo ""

echo -e "${BLUE}💡 QAT量化原理:${NC}"
echo "QAT在训练过程中模拟量化的影响，让模型学会适应量化误差"
echo "• 前向传播: 使用伪量化(fake quantization)模拟INT8计算"
echo "• 反向传播: 仍然使用FP32进行梯度计算"  
echo "• 训练完成: 将伪量化替换为真实的INT8算子"
echo ""

echo -e "${GREEN}🎉 开始你的量化之旅吧!${NC}"
