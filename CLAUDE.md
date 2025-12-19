# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PointNeXt is a PyTorch-based point cloud learning framework implementing improved training and scaling strategies for PointNet++. The repository is organized around a modular, configuration-driven architecture using a registry-based plugin system.

**Key Paper**: "PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies" (NeurIPS 2022)

## Core Architecture

### Registry-Based Plugin System

The entire framework is built on a **registry pattern** (`openpoints/utils/registry.py`):

- All major components (models, datasets, losses, transforms) use registries
- Components are registered with decorators: `@MODELS.register_module()`, `@DATASETS.register_module()`, etc.
- Components are instantiated from YAML configs using `build_*_from_cfg()` functions
- To add new components: create the class, register it, then reference by NAME in config

### Configuration System

Uses **hierarchical YAML-based configuration** via `EasyConfig`:

1. Base config: `cfgs/default.yaml`
2. Dataset defaults: `cfgs/[dataset]/default.yaml`
3. Model-specific: `cfgs/[dataset]/[model].yaml`
4. Command-line overrides

Config values are accessible as attributes (e.g., `cfg.model.encoder_args.in_channels`)

### Model Composition Pattern

Models follow a **compositional architecture**:

```
BaseCls (or task-specific head)
├── encoder (backbone: PointNext, PointNet++, DGCNN, etc.)
├── prediction (task head: ClsHead, SegHead, etc.)
└── criterion (loss function)
```

All built from config using the registry system.

### Directory Structure

```
PointNeXt/
├── openpoints/              # Core framework (git submodule)
│   ├── models/              # Model implementations
│   │   ├── backbone/        # 25+ point cloud backbones
│   │   ├── classification/  # Classification heads
│   │   ├── segmentation/    # Segmentation heads
│   │   └── layers/          # Atomic operations
│   ├── dataset/             # Dataset loaders
│   ├── loss/                # Loss functions
│   ├── optim/               # Optimizers
│   ├── scheduler/           # LR schedulers
│   ├── transforms/          # Data augmentation
│   └── utils/               # Registry, config, metrics, etc.
├── examples/                # Task-specific training scripts
│   ├── classification/      # Classification tasks
│   ├── segmentation/        # Segmentation tasks
│   └── shapenetpart/        # Part segmentation
├── cfgs/                    # YAML configuration files
│   ├── modelnet40ply2048/   # ModelNet40 configs
│   ├── s3dis/               # S3DIS configs
│   ├── scannet/             # ScanNet configs
│   └── scanobjectnn/        # ScanObjectNN configs
├── script/                  # Shell scripts for training
└── tools/                   # Utility scripts
```

## Common Commands

### Environment Setup

```bash
# Install dependencies (requires CUDA 11.3, modify install.sh for other versions)
source update.sh
source install.sh

# The install.sh script:
# - Updates git submodules (openpoints)
# - Creates conda environment 'openpoints' with Python 3.7
# - Installs PyTorch 1.10.1 + CUDA 11.3
# - Compiles C++ extensions (pointnet2_batch, pointops, subsampling, etc.)
```

### Training

**Basic training pattern:**
```bash
CUDA_VISIBLE_DEVICES=$GPUs python examples/$task_folder/main.py --cfg $cfg [kwargs]
```

**Examples:**

```bash
# Classification on ModelNet40 (single GPU)
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml

# Segmentation on S3DIS
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
    --cfg cfgs/s3dis/pointnext-s.yaml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml

# Override config parameters
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    epochs=300 batch_size=16 wandb.use_wandb=True

# Using convenience scripts
./run_training.sh classification pointnext-s modelnet40 0
./quick_train.sh  # Interactive menu
```

### Testing

```bash
# Test with pretrained model
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=test \
    --pretrained_path /path/to/checkpoint.pth

# S3DIS 6-fold cross validation
python examples/segmentation/test_s3dis_6fold.py \
    --cfg cfgs/s3dis/pointnext-s.yaml
```

### Quantization

**Static Quantization (Post-Training Quantization):**
```bash
# Basic FX-based quantization
python quantize_fx.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml

# With pretrained model
python quantize_fx.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    --pretrained /path/to/model.pth \
    --save_path quantized_models/my_model.pth
```

**QAT (Quantization-Aware Training):**
```bash
# QAT quantization (higher accuracy, requires training)
python quantize_fx.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    --method qat \
    --epochs 3

# Interactive quantization menu
./quick_quantize.sh
```

**Running batch quantization examples:**
```bash
./run_quantization_examples.sh
```

### Network Analysis

```bash
# Print model structure (text format)
python tools/print_all_network_layers.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml

# Export as JSON
python tools/print_all_network_layers.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    --format json

# Batch process all configs in a directory
python tools/print_all_network_layers.py \
    --cfg-dir cfgs/modelnet40ply2048 \
    --format json

# Analyze KD-Tree usage
python analyze_kdtree.py
python kdtree_summary.py
```

## Development Workflow

### Adding a New Model

1. **Create model file**: `openpoints/models/backbone/mymodel.py`
2. **Register the model**:
   ```python
   from ..build import MODELS

   @MODELS.register_module()
   class MyModel(nn.Module):
       def __init__(self, in_channels, out_channels, ...):
           super().__init__()
           # Implementation

       def forward(self, p, f):  # p=points, f=features
           # Forward pass
           return features

       @property
       def out_channels(self):  # Required property
           return self._out_channels
   ```
3. **Create config**: `cfgs/[dataset]/mymodel.yaml`
   ```yaml
   model:
     NAME: BaseCls  # or BasePartSeg, BaseSeg
     encoder_args:
       NAME: MyModel
       in_channels: 3
       # ... model-specific args
     cls_args:
       NAME: ClsHead
       num_classes: 40
   ```
4. **Train**: `python examples/classification/main.py --cfg cfgs/[dataset]/mymodel.yaml`

### Adding a New Dataset

1. **Create dataset file**: `openpoints/dataset/mydataset/mydataset.py`
2. **Inherit from DatasetBase** or implement `__getitem__` and `__len__`
3. **Register**:
   ```python
   from ..build import DATASETS

   @DATASETS.register_module()
   class MyDataset(DatasetBase):
       def __init__(self, data_dir, split='train', ...):
           super().__init__()
           # Load data

       def __getitem__(self, idx):
           # Return dict with 'pos', 'x', 'y', etc.
           return {'pos': points, 'x': features, 'y': label}

       def __len__(self):
           return len(self.data)
   ```
4. **Update config**:
   ```yaml
   dataset:
     common:
       NAME: MyDataset
       data_dir: ./data/mydataset
     train:
       split: train
     val:
       split: val
   ```

### Adding a New Loss Function

1. **Create in** `openpoints/loss/build.py` or separate file
2. **Register**:
   ```python
   from .build import LOSS

   @LOSS.register_module()
   class MyLoss(nn.Module):
       def __init__(self, ...):
           super().__init__()

       def forward(self, logits, targets):
           # Compute loss
           return loss
   ```
3. **Use in config**:
   ```yaml
   criterion_args:
     NAME: MyLoss
     # ... loss-specific args
   ```

## Key Implementation Details

### Model Building Flow

```
Config (YAML)
    ↓
build_model_from_cfg(cfg.model)
    ↓
MODELS.build() → Instantiate by NAME
    ↓
BaseCls/BaseSeg/BasePartSeg
    ├─ encoder = build_model_from_cfg(cfg.model.encoder_args)
    ├─ prediction = build_model_from_cfg(cfg.model.cls_args)
    └─ criterion = build_criterion_from_cfg(cfg.criterion_args)
```

### Training Pipeline

Entry point: `examples/[task]/main.py`

```python
# 1. Load and merge configs
cfg = EasyConfig()
cfg.load(args.cfg, recursive=True)  # Hierarchical loading
cfg.update(opts)  # CLI overrides

# 2. Build components from config
model = build_model_from_cfg(cfg.model).to(device)
optimizer = build_optimizer_from_cfg(model, cfg.optimizer)
scheduler = build_scheduler_from_cfg(cfg, optimizer)
train_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, split='train')
val_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, split='val')

# 3. Training loop (examples/[task]/train.py)
for epoch in range(start_epoch, epochs):
    train_one_epoch(model, train_loader, optimizer, scheduler)
    metrics = validate(model, val_loader)
    save_checkpoint(model, optimizer, epoch, metrics)
```

### Data Transforms

Transforms are composable and registered:

```python
# Registry-based
DataTransforms = registry.Registry('datatransforms')

@DataTransforms.register_module()
class PointsToTensor:
    def __call__(self, data):
        # Convert to tensor
        return data

# Config usage
datatransforms:
  train:
    - PointsToTensor
    - PointCloudScaleAndTranslate
    - PointCloudRotation
  val:
    - PointsToTensor
```

### Distributed Training

Automatic multi-GPU support:

```bash
# Automatically uses DistributedDataParallel if multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/classification/main.py ...
```

Uses SyncBatchNorm for multi-GPU training.

## Important Files

### Configuration Files
- `cfgs/default.yaml` - Global defaults for all experiments
- `cfgs/[dataset]/default.yaml` - Dataset-specific defaults
- `cfgs/[dataset]/[model].yaml` - Model-specific config

### Training Scripts
- `examples/classification/main.py` - Classification entry point
- `examples/segmentation/main.py` - Segmentation entry point
- `examples/shapenetpart/main.py` - Part segmentation entry point
- `examples/*/train.py` - Task-specific training loops

### Core Framework (openpoints/)
- `openpoints/models/build.py` - Model registry and builder
- `openpoints/dataset/build.py` - Dataset registry and builder
- `openpoints/utils/registry.py` - Registry implementation
- `openpoints/utils/config.py` - EasyConfig for YAML handling

### Shell Scripts
- `run_training.sh` - Universal training script with arguments
- `quick_train.sh` - Interactive training menu
- `batch_train.sh` - Batch experiments (model comparison, ablation)
- `quick_quantize.sh` - Interactive quantization menu
- `script/main_*.sh` - Task-specific training scripts

### Quantization
- `quantize_fx.py` - PyTorch FX-based quantization (clean, graph-mode)
- `simple_quantize.py` - Simplified quantization script
- `quantize_eager_clean.py` - Eager mode quantization
- `test_quantized_model.py` - Test quantized models

## KD-Tree Variants

This repository includes KD-Tree accelerated variants for efficient neighbor search:

- **KD-Tree models**: `pointnet++_kdtree.yaml`, `pointnext-s_kdtree.yaml`
- **Adaptive KD-Tree**: `pointnext-s_kdtree_adaptive.yaml`
- **Analysis tools**: `analyze_kdtree.py`, `kdtree_summary.py`

KD-Tree variants provide faster ball query operations compared to brute-force search.

## Quantization Notes

Two quantization approaches are provided:

1. **PyTorch FX Quantization** (`quantize_fx.py`):
   - Graph-mode quantization
   - Automatic operator fusion
   - Cleaner, more intuitive code
   - Recommended for most use cases

2. **QAT (Quantization-Aware Training)**:
   - Train with fake quantization for higher accuracy
   - Requires 3-5 epochs of fine-tuning
   - Use `--method qat` flag
   - Typically 1-2% better accuracy than static quantization

**Trade-offs:**
- Static quantization: Fast (minutes), moderate accuracy drop (1-2%)
- QAT: Slower (hours), minimal accuracy drop (0.4%)

## Logging and Experiment Tracking

### Weights & Biases Integration

```bash
# Enable wandb logging
python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    wandb.use_wandb=True \
    wandb.name=my_experiment \
    wandb.project=pointnext
```

### Log Files

Training logs are saved to `logs/` directory with timestamps:
```
logs/[timestamp]_[task]_[model]_[dataset]/
├── training.log
├── config.yaml
└── checkpoints/
```

## Datasets

### Supported Datasets

- **Classification**: ModelNet40, ScanObjectNN
- **Segmentation**: S3DIS, ScanNet v2
- **Part Segmentation**: ShapeNetPart

### Data Directory Structure

Expected structure (configure via `cfg.dataset.common.data_dir`):

```
data/
├── ModelNet40/
├── S3DIS/
├── ScanNet/
└── ShapeNetPart/
```

### Downloading Datasets

```bash
# S3DIS download script provided
./script/download_s3dis.sh
```

For other datasets, refer to [online documentation](https://guochengqian.github.io/PointNeXt/).

## Testing

### Running Tests

```bash
# Quick test script
python quick_test.py

# Test specific model
python test_pointnext_kdtree.py

# Test quantized model
python test_quantized_model.py --model_path quantized_models/model.pth
```

## Performance Profiling

```bash
# Profile FLOPs and parameters
./script/profile_flops.sh

# General profiling
python examples/profile.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```
