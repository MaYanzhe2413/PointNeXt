"""
根据不同的cfg打印完整的网络结构（所有层的详细信息）

使用方法:
    # 打印单个配置的网络结构
    python tools/print_all_network_layers.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
    
    # 输出为JSON格式
    python tools/print_all_network_layers.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --format json
    
    # 批量处理目录
    python tools/print_all_network_layers.py --cfg-dir cfgs/modelnet40ply2048 --format json
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg


class NetworkStructurePrinter:
    """网络结构打印器"""
    
    def __init__(self, model: nn.Module, cfg: EasyConfig):
        self.model = model
        self.cfg = cfg
        self.layer_info = []
        self.current_stage = 0
        
    def extract_layer_info(self, name: str, module: nn.Module, depth: int = 0) -> Dict[str, Any]:
        """提取单个层的详细信息"""
        info = {
            "name": name if name else "root",
            "type": module.__class__.__name__,
            "depth": depth,
            "trainable_params": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in module.parameters()),
        }
        
        # 提取常见的层属性
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            info["in_features"] = module.in_features
            info["out_features"] = module.out_features
        
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            info["in_channels"] = module.in_channels
            info["out_channels"] = module.out_channels
            
        if hasattr(module, 'kernel_size'):
            info["kernel_size"] = module.kernel_size if not isinstance(module.kernel_size, tuple) else list(module.kernel_size)
            
        if hasattr(module, 'stride'):
            info["stride"] = module.stride if not isinstance(module.stride, tuple) else list(module.stride)
            
        if hasattr(module, 'padding'):
            info["padding"] = module.padding if not isinstance(module.padding, tuple) else list(module.padding)
            
        if hasattr(module, 'groups'):
            info["groups"] = module.groups
            
        if hasattr(module, 'bias') and module.bias is not None:
            info["has_bias"] = True
            
        # 提取dropout rate
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            info["dropout_rate"] = module.p
            
        # 提取normalization信息
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            info["num_features"] = module.num_features
            info["eps"] = module.eps
            info["momentum"] = module.momentum
            
        # 提取激活函数信息
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU)):
            if hasattr(module, 'negative_slope'):
                info["negative_slope"] = module.negative_slope
            if hasattr(module, 'inplace'):
                info["inplace"] = module.inplace
                
        return info
    
    def collect_all_layers(self) -> List[Dict[str, Any]]:
        """收集模型中所有层的信息"""
        layers = []
        
        def traverse(name: str, module: nn.Module, depth: int = 0):
            # 获取当前模块信息
            layer_info = self.extract_layer_info(name, module, depth)
            
            # 检查是否有子模块
            children = list(module.named_children())
            layer_info["has_children"] = len(children) > 0
            layer_info["num_children"] = len(children)
            
            layers.append(layer_info)
            
            # 递归遍历子模块
            for child_name, child_module in children:
                full_name = f"{name}.{child_name}" if name else child_name
                traverse(full_name, child_module, depth + 1)
        
        traverse("", self.model, 0)
        return layers
    
    def print_hierarchical_structure(self, output_file=None):
        """打印层次化的网络结构"""
        layers = self.collect_all_layers()
        
        output_lines = []
        output_lines.append("=" * 100)
        output_lines.append(f"Network Structure: {self.cfg.get('cfg_name', 'Unknown')}")
        output_lines.append("=" * 100)
        output_lines.append("")
        
        # 统计信息
        total_params = sum(layer['total_params'] for layer in layers if not layer['has_children'])
        trainable_params = sum(layer['trainable_params'] for layer in layers if not layer['has_children'])
        
        output_lines.append(f"Total Parameters: {total_params:,}")
        output_lines.append(f"Trainable Parameters: {trainable_params:,}")
        output_lines.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        output_lines.append(f"Total Layers: {len(layers)}")
        output_lines.append("")
        output_lines.append("-" * 100)
        
        # 打印每一层
        for idx, layer in enumerate(layers):
            indent = "  " * layer['depth']
            name = layer['name'] if layer['name'] else "Model"
            
            # 基本信息
            output_lines.append(f"\n[Layer {idx}] {indent}{name}")
            output_lines.append(f"{indent}├─ Type: {layer['type']}")
            
            if layer['total_params'] > 0:
                output_lines.append(f"{indent}├─ Parameters: {layer['total_params']:,} (Trainable: {layer['trainable_params']:,})")
            
            # 详细属性
            details = []
            for key, value in layer.items():
                if key not in ['name', 'type', 'depth', 'trainable_params', 'total_params', 'has_children', 'num_children']:
                    details.append(f"{key}={value}")
            
            if details:
                output_lines.append(f"{indent}└─ Details: {', '.join(details)}")
            
            if layer['has_children']:
                output_lines.append(f"{indent}   ({layer['num_children']} sub-modules)")
        
        output_lines.append("\n" + "=" * 100)
        
        # 输出到文件或屏幕
        result = "\n".join(output_lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Network structure saved to: {output_file}")
        else:
            print(result)
        
        return result
    
    def export_to_json(self, output_file: str):
        """导出为JSON格式（类似config.json的格式）"""
        layers = self.collect_all_layers()
        
        # 构建输出结构
        output = {
            "network_name": self.cfg.get('cfg_name', 'Unknown'),
            "total_parameters": sum(layer['total_params'] for layer in layers if not layer['has_children']),
            "trainable_parameters": sum(layer['trainable_params'] for layer in layers if not layer['has_children']),
            "total_layers": len(layers),
            "layers": layers,
            "model_config": dict(self.cfg.model) if hasattr(self.cfg, 'model') else {}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nNetwork structure (JSON) saved to: {output_file}")


def process_single_config(cfg_path: str, args):
    """处理单个配置文件"""
    print(f"\n{'#' * 80}")
    print(f"Processing: {cfg_path}")
    print(f"{'#' * 80}\n")
    
    # 加载配置
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.cfg_name = Path(cfg_path).stem
    
    # 构建模型
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg.model).to(device)
    model.eval()
    
    # 创建打印器
    printer = NetworkStructurePrinter(model, cfg)
    
    # 根据输出格式选择
    if args.format == 'text':
        output_file = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"{cfg.cfg_name}_structure.txt")
        printer.print_hierarchical_structure(output_file)
        
    elif args.format == 'json':
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"{cfg.cfg_name}_structure.json")
        else:
            output_file = f"{cfg.cfg_name}_structure.json"
        printer.export_to_json(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="根据cfg打印完整网络结构（所有层的详细信息）"
    )
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    parser.add_argument("--cfg-dir", type=str, help="配置文件目录")
    parser.add_argument("--pattern", type=str, default="*.yaml", help="文件匹配模式")
    parser.add_argument("--format", choices=['text', 'json'], default='text', 
                       help="输出格式: text(文本) 或 json(JSON)")
    parser.add_argument("--output-dir", type=str, default="network_structures", help="输出目录")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help="运行设备")
    
    args = parser.parse_args()
    
    # 收集要处理的配置文件
    cfg_files = []
    
    if args.cfg:
        cfg_path = Path(args.cfg)
        if cfg_path.is_file():
            cfg_files.append(str(cfg_path))
        elif cfg_path.is_dir():
            cfg_files.extend([str(f) for f in cfg_path.rglob(args.pattern)])
        else:
            raise FileNotFoundError(f"配置文件或目录不存在: {args.cfg}")
    
    if args.cfg_dir:
        cfg_dir = Path(args.cfg_dir)
        if not cfg_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {args.cfg_dir}")
        cfg_files.extend([str(f) for f in cfg_dir.rglob(args.pattern)])
    
    if not cfg_files:
        print("错误: 请指定 --cfg 或 --cfg-dir 参数")
        parser.print_help()
        return
    
    print(f"找到 {len(cfg_files)} 个配置文件")
    
    # 处理每个配置文件
    for cfg_file in cfg_files:
        try:
            process_single_config(cfg_file, args)
        except Exception as e:
            print(f"处理 {cfg_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成! 共处理 {len(cfg_files)} 个配置文件")
    if args.output_dir and os.path.exists(args.output_dir):
        print(f"输出文件保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
