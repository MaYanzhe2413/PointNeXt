# 网络层结构打印工具

根据不同的cfg配置文件打印完整的网络结构，包括所有层的详细信息（类似`layout_jason/config.json`的结构化展示）。

## 快速使用

### 1. 打印单个模型的结构（文本格式，输出到屏幕）
```bash
python tools/print_all_network_layers.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

### 2. 导出为JSON格式（类似config.json）
```bash
python tools/print_all_network_layers.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --format json
```

### 3. 批量处理整个目录的所有配置文件
```bash
python tools/print_all_network_layers.py --cfg-dir cfgs/modelnet40ply2048 --format json
```

### 4. 处理特定模式的配置文件
```bash
# 只处理PointNeXt系列模型
python tools/print_all_network_layers.py --cfg-dir cfgs/modelnet40ply2048 --pattern "*pointnext*.yaml" --format json

# 处理所有数据集的PointNeXt模型
python tools/print_all_network_layers.py --cfg-dir cfgs --pattern "*pointnext*.yaml" --format json
```

## 输出格式

### Text格式（层次化文本）
```
====================================================================================================
Network Structure: pointnext-s
====================================================================================================

Total Parameters: 1,374,000
Trainable Parameters: 1,374,000
Total Layers: 156

[Layer 0] Model
├─ Type: BaseCls
├─ Parameters: 1,374,000 (Trainable: 1,374,000)

[Layer 1]   encoder
├─ Type: PointNextEncoder
└─ Details: in_channels=3, width=32
...
```

### JSON格式（结构化数据）
```json
{
  "network_name": "pointnext-s",
  "total_parameters": 1374000,
  "trainable_parameters": 1374000,
  "total_layers": 156,
  "layers": [
    {
      "name": "encoder.stages.0.0.conv",
      "type": "Conv1d",
      "depth": 4,
      "trainable_params": 9216,
      "total_params": 9216,
      "in_channels": 32,
      "out_channels": 128,
      "kernel_size": 1,
      "stride": 1
    }
  ],
  "model_config": {...}
}
```

## 参数说明

- `--cfg`: 配置文件路径
- `--cfg-dir`: 配置文件目录（批量处理）
- `--pattern`: 文件匹配模式（默认: `*.yaml`）
- `--format`: 输出格式，`text` 或 `json`（默认: `text`）
- `--output-dir`: 输出目录（默认: `network_structures`）
- `--device`: 运行设备，`cpu` 或 `cuda`（默认: `cpu`）

## 输出文件位置

- Text格式: 输出到屏幕或 `{output_dir}/{模型名}_structure.txt`
- JSON格式: `{output_dir}/{模型名}_structure.json`

## 实际应用

### 比较不同模型
```bash
# 导出所有ModelNet40配置的结构
python tools/print_all_network_layers.py --cfg-dir cfgs/modelnet40ply2048 --format json --output-dir modelnet_structures

# 然后可以用Python分析
import json, glob
for f in glob.glob('modelnet_structures/*.json'):
    data = json.load(open(f))
    print(f"{data['network_name']}: {data['total_parameters']:,} params")
```

### 用于硬件映射
JSON输出可以作为硬件仿真工具（如SCALE-Sim）的输入，参考 `layout_jason/config.json` 的格式。

## 与原工具对比

- `tools/print_model_structure.py`: 表格格式，快速查看
- `tools/print_all_network_layers.py`: 详细层信息，支持JSON，适合分析和映射
