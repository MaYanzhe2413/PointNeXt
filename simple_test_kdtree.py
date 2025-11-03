#!/usr/bin/env python3
"""
简化版PointNeXt KDTree配置测试
仅测试配置加载和模型构建，不涉及前向传播
"""

import sys
sys.path.append('/workspace/PointNeXt')

from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg

def simple_test_config(config_path):
    """简化测试：仅验证配置和模型构建"""
    config_name = config_path.split('/')[-1]
    print(f"🧪 测试: {config_name}")
    
    try:
        # 加载配置
        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        
        # 设置必要参数
        if not hasattr(cfg, 'num_classes'):
            cfg.num_classes = 40
        if not hasattr(cfg, 'input_channels'):
            cfg.input_channels = 3
            
        print(f"   ✅ 配置加载成功")
        print(f"   📊 采样器: {cfg.model.encoder_args.sampler}")
        print(f"   🏗️  网络宽度: {cfg.model.encoder_args.width}")
        print(f"   📏 网络深度: {len(cfg.model.encoder_args.blocks)}")
        
        # 构建模型
        model = build_model_from_cfg(cfg.model)
        print(f"   ✅ 模型构建成功")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   🔢 参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False

def main():
    print("🌳 PointNeXt KDTree配置语法测试")
    print("=" * 50)
    
    configs = [
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s_kdtree.yaml",
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-b_kdtree.yaml", 
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s_kdtree_adaptive.yaml"
    ]
    
    results = []
    for config_path in configs:
        success = simple_test_config(config_path)
        results.append(success)
        print()
    
    success_count = sum(results)
    print(f"📊 结果: {success_count}/{len(configs)} 配置测试通过")
    
    if success_count == len(configs):
        print("\\n🎉 所有配置文件语法正确！")
        print("\\n🚀 可以开始训练:")
        print("   ./run_training.sh classification pointnext-s_kdtree modelnet40")
        print("   ./run_training.sh classification pointnext-b_kdtree modelnet40")
        print("   ./run_training.sh classification pointnext-s_kdtree_adaptive modelnet40")
        
        print("\\n📁 创建的配置文件:")
        for config in configs:
            print(f"   📄 {config}")
    else:
        print("⚠️  部分配置需要修复")

if __name__ == "__main__":
    main()