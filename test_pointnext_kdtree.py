#!/usr/bin/env python3
"""
测试PointNeXt KDTree配置文件
验证模型能否正确加载和运行
"""

import sys
import torch
sys.path.append('/workspace/PointNeXt')

from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg

def test_pointnext_kdtree_config(config_path):
    """测试PointNeXt KDTree配置"""
    print(f"\\n🧪 测试配置文件: {config_path}")
    print("=" * 60)
    
    try:
        # 加载配置
        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        
        # 设置默认参数
        if not hasattr(cfg, 'num_classes'):
            cfg.num_classes = 40
        if not hasattr(cfg, 'input_channels'):
            cfg.input_channels = 3
            
        print(f"✅ 配置文件加载成功")
        print(f"   编码器: {cfg.model.encoder_args.NAME}")
        print(f"   采样器: {cfg.model.encoder_args.sampler}")
        print(f"   网络宽度: {cfg.model.encoder_args.width}")
        print(f"   网络深度: {len(cfg.model.encoder_args.blocks)}")
        
        # 构建模型
        model = build_model_from_cfg(cfg.model)
        print(f"✅ 模型构建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 测试前向传播
        batch_size = 2
        num_points = 1024
        
        # 创建符合PointNeXt格式的输入
        pos = torch.randn(batch_size, num_points, 3)
        
        print(f"\\n🚀 测试前向传播...")
        print(f"   输入形状: {pos.shape}")
        
        # 设置为评估模式
        model.eval()
        
        with torch.no_grad():
            # PointNeXt使用字典格式输入
            data = {'pos': pos}
            
            # 前向传播
            output = model(data)
            
            print(f"✅ 前向传播成功")
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   输出 {key}: {value.shape}")
            else:
                print(f"   输出形状: {output.shape}")
                
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数：测试所有KDTree配置"""
    print("🌳 PointNeXt KDTree配置测试")
    print("=" * 60)
    
    configs = [
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s_kdtree.yaml",
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-b_kdtree.yaml",
        "/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s_kdtree_adaptive.yaml"
    ]
    
    results = {}
    
    for config_path in configs:
        config_name = config_path.split('/')[-1].replace('.yaml', '')
        results[config_name] = test_pointnext_kdtree_config(config_path)
        print()
    
    # 汇总结果
    print("\\n📊 测试结果汇总:")
    print("=" * 60)
    for config_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {config_name:<30} {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\\n总体结果: {success_count}/{total_count} 配置测试通过")
    
    if success_count == total_count:
        print("🎉 所有KDTree配置测试通过！")
        print("\\n🚀 可以开始训练:")
        print("   ./run_training.sh classification pointnext-s_kdtree modelnet40")
        print("   ./run_training.sh classification pointnext-b_kdtree modelnet40")
    else:
        print("⚠️  部分配置需要修复")

if __name__ == "__main__":
    main()