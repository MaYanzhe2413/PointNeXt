#!/bin/bash

# PointNeXt 量化脚本使用示例

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========== PointNeXt 模型量化示例 ==========${NC}"
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

# 量化示例
echo -e "${BLUE}🔥 开始量化示例${NC}"
echo ""

# 示例1: PointNeXt-S 分类模型量化
echo -e "${YELLOW}示例1: PointNeXt-S ModelNet40 分类模型量化${NC}"
if [ -f "cfgs/modelnet40ply2048/pointnext-s.yaml" ]; then
    python quantize_fx.py \
        --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
        --save_path quantized_models/pointnext-s-cls-quantized.pth \
        --device cpu
    echo ""
else
    echo -e "${RED}⚠️  配置文件不存在，跳过示例1${NC}"
fi

# 示例2: PointNeXt-S 分割模型量化
echo -e "${YELLOW}示例2: PointNeXt-S S3DIS 分割模型量化${NC}"
if [ -f "cfgs/s3dis/pointnext-s.yaml" ]; then
    python quantize_fx.py \
        --cfg cfgs/s3dis/pointnext-s.yaml \
        --save_path quantized_models/pointnext-s-seg-quantized.pth \
        --device cpu
    echo ""
else
    echo -e "${RED}⚠️  配置文件不存在，跳过示例2${NC}"
fi

# 示例3: 使用预训练模型量化
echo -e "${YELLOW}示例3: 使用预训练模型进行量化${NC}"
echo -e "${BLUE}如果你有预训练模型，可以这样使用:${NC}"
echo -e "${GREEN}python quantize_fx.py \\${NC}"
echo -e "${GREEN}    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \\${NC}"
echo -e "${GREEN}    --pretrained /path/to/your/pretrained_model.pth \\${NC}"
echo -e "${GREEN}    --save_path quantized_models/pretrained-quantized.pth \\${NC}"
echo -e "${GREEN}    --device cpu${NC}"
echo ""

echo -e "${GREEN}🎉 量化示例完成!${NC}"
echo ""
echo -e "${BLUE}📁 量化后的模型保存在 quantized_models/ 目录下${NC}"
echo -e "${BLUE}📊 查看量化效果请检查终端输出的性能对比${NC}"
