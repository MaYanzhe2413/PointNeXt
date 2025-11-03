# PointNeXt 训练脚本使用指南

本目录包含了几个便捷的训练脚本，帮助你轻松运行PointNeXt的各种训练任务。

## 📁 脚本文件

### 1. `run_training.sh` - 主训练脚本
通用的训练脚本，支持所有任务类型和模型。

**使用方法:**
```bash
./run_training.sh [task] [model] [dataset] [gpu_ids] [additional_args]
```

**参数说明:**
- `task`: 任务类型 (`classification`, `segmentation`, `partseg`)
- `model`: 模型名称 (`pointnext-s`, `pointnext-b`, `pointnext-l`, `pointnet++`, `pointmlp`, `dgcnn`)
- `dataset`: 数据集名称 (`modelnet40`, `scanobjectnn`, `s3dis`, `shapenetpart`)
- `gpu_ids`: GPU编号 (可选，默认为0)
- `additional_args`: 额外参数 (可选)

**示例:**
```bash
# 基础训练
./run_training.sh classification pointnext-s modelnet40

# 指定GPU
./run_training.sh segmentation pointnext-s s3dis 0

# 多GPU训练
./run_training.sh classification pointnext-s modelnet40 0,1,2,3

# 添加额外参数
./run_training.sh classification pointnext-s modelnet40 0 "epochs=300 batch_size=16 wandb.use_wandb=True"

# 查看帮助
./run_training.sh --help
```

### 2. `quick_train.sh` - 快速训练菜单
交互式菜单，适合初学者使用。

**使用方法:**
```bash
./quick_train.sh
```

通过菜单选择预设的训练组合，系统会引导你完成所有设置。

### 3. `batch_train.sh` - 批量训练脚本
用于运行批量实验和对比研究。

**使用方法:**
```bash
# 模型对比实验
./batch_train.sh compare_models

# 数据集对比实验  
./batch_train.sh compare_datasets

# 消融实验
./batch_train.sh ablation_study

# 自定义批量实验
./batch_train.sh custom
```

## 🎯 支持的训练组合

### 分类任务
| 数据集 | 支持的模型 |
|--------|------------|
| ModelNet40 | pointnext-s, pointnet++, pointmlp, dgcnn |
| ScanObjectNN | pointnext-s, pointnet++, pointmlp, dgcnn |

### 分割任务
| 数据集 | 支持的模型 |
|--------|------------|
| S3DIS | pointnext-s, pointnext-b, pointnext-l, pointnet++, dgcnn |

### 部件分割
| 数据集 | 支持的模型 |
|--------|------------|
| ShapeNetPart | pointnext-s, pointnet++ |

## ⚙️ 常用参数

### 训练参数
- `epochs=300` - 设置训练轮数
- `batch_size=32` - 设置批次大小
- `lr=0.001` - 设置学习率

### 日志参数
- `wandb.use_wandb=True` - 启用wandb日志
- `wandb.name=experiment_name` - 设置实验名称
- `wandb.project=pointnext` - 设置项目名称

### 测试参数
- `mode=test` - 测试模式
- `--pretrained_path=/path/to/model.pth` - 预训练模型路径

## 📊 实验管理

### 日志文件
训练日志自动保存在 `logs/` 目录下，按时间戳和实验配置命名：
```
logs/20240906_143020_classification_pointnext-s_modelnet40/training.log
```

### WandB集成
启用wandb后可以在线查看训练过程：
```bash
./run_training.sh classification pointnext-s modelnet40 0 "wandb.use_wandb=True wandb.name=my_experiment"
```

## 🚀 快速开始

### 1. 新手推荐
```bash
# 使用交互式菜单
./quick_train.sh
```

### 2. 简单分类训练
```bash
# PointNeXt-S on ModelNet40
./run_training.sh classification pointnext-s modelnet40
```

### 3. 简单分割训练
```bash
# PointNeXt-S on S3DIS
./run_training.sh segmentation pointnext-s s3dis
```

### 4. 模型对比实验
```bash
# 比较不同模型在ModelNet40上的性能
./batch_train.sh compare_models
```

## 🔧 故障排除

### 常见问题

1. **配置文件不存在**
   - 检查模型名称和数据集名称是否正确
   - 确保配置文件存在于 `cfgs/` 目录下

2. **CUDA错误**
   - 检查GPU编号是否正确
   - 确保CUDA环境正确安装

3. **内存不足**
   - 减小batch_size：`batch_size=8`
   - 使用较小的模型：如pointnext-s

4. **数据集未找到**
   - 检查数据集是否下载到正确位置
   - 参考OpenPoints文档下载数据集

### 调试技巧

1. **测试环境**
```bash
# 检查Python和PyTorch
python -c "import torch; print(torch.__version__)"
```

2. **查看详细错误**
```bash
# 查看训练日志
cat logs/latest_experiment/training.log
```

3. **小规模测试**
```bash
# 用更少的epochs测试
./run_training.sh classification pointnext-s modelnet40 0 "epochs=1"
```

## 📝 自定义配置

如需自定义训练配置，可以：

1. 复制现有配置文件
2. 修改参数
3. 使用新配置文件训练

```bash
cp cfgs/modelnet40ply2048/pointnext-s.yaml cfgs/modelnet40ply2048/my-config.yaml
# 编辑 my-config.yaml
./run_training.sh classification my-config modelnet40
```

---

**提示:** 第一次运行时，脚本会检查环境并创建必要的目录。建议先运行一个简单的实验来验证环境配置是否正确。
