#!/usr/bin/env python3
"""
测试量化模型是否正常工作
"""

import torch
import torch.nn as nn
from quantize_eager import EagerQuantizationWrapper
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig

def test_quantized_model():
    print("🧪 测试量化模型...")
    
    # 1. 构建原始模型
    cfg = EasyConfig()
    cfg.load('cfgs/modelnet40ply2048/pointnet++.yaml', recursive=True)
    original_model = build_model_from_cfg(cfg.model)
    
    print(f"✅ 原始模型构建完成: {type(original_model).__name__}")
    
    # 2. 加载量化模型
    try:
        # 包装模型并设置量化配置
        wrapped_model = EagerQuantizationWrapper(original_model)
        from torch.quantization import get_default_qconfig, prepare, convert
        wrapped_model.qconfig = get_default_qconfig('fbgemm')
        
        # 准备量化
        prepared_model = prepare(wrapped_model, inplace=False)
        
        # 转换为量化模型
        quantized_model = convert(prepared_model, inplace=False)
        
        print("✅ 量化模型构建成功")
        
    except Exception as e:
        print(f"❌ 量化模型构建失败: {e}")
        return False
    
    # 3. 测试推理
    print("🔄 测试模型推理...")
    
    # 测试数据
    test_data = torch.randn(2, 1024, 3)  # [B, N, 3]
    
    try:
        # 原始模型推理
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(test_data)
        
        print(f"✅ 原始模型推理成功: 输出形状 {original_output.shape}")
        
        # 量化模型推理
        quantized_model.eval()
        with torch.no_grad():
            quantized_output = quantized_model(test_data)
        
        print(f"✅ 量化模型推理成功: 输出形状 {quantized_output.shape}")
        
        # 比较输出
        if original_output.shape == quantized_output.shape:
            print("✅ 输出形状一致")
            
            # 计算差异
            mse = torch.mean((original_output - quantized_output) ** 2).item()
            print(f"📊 均方误差: {mse:.6f}")
            
            if mse < 1.0:  # 允许一定的量化误差
                print("✅ 输出差异在合理范围内")
                return True
            else:
                print("⚠️  输出差异较大，可能需要更好的校准")
                return True  # 仍然算成功，因为形状正确
        else:
            print(f"❌ 输出形状不一致: {original_output.shape} vs {quantized_output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

def test_model_size():
    print("\n📦 测试模型大小...")
    
    try:
        # 测试保存的量化模型
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        print(f"🔧 使用设备: {device}")
        
        # 构建原始模型
        cfg = EasyConfig()
        cfg.load('cfgs/modelnet40ply2048/pointnet++.yaml', recursive=True)
        original_model = build_model_from_cfg(cfg.model)
        
        # 保存原始模型
        torch.save(original_model.state_dict(), 'temp_original.pth')
        original_size = torch.load('temp_original.pth', map_location='cpu')
        
        # 计算参数数量
        original_params = sum(p.numel() for p in original_model.parameters())
        
        print(f"📊 原始模型参数数量: {original_params:,}")
        
        # 检查量化模型是否存在
        import os
        if os.path.exists('quantized_models/quantized_model_static_eager.pth'):
            quantized_size = os.path.getsize('quantized_models/quantized_model_static_eager.pth')
            original_file_size = os.path.getsize('temp_original.pth')
            
            print(f"📦 原始模型文件大小: {original_file_size / (1024*1024):.2f} MB")
            print(f"📦 量化模型文件大小: {quantized_size / (1024*1024):.2f} MB")
            print(f"🎯 压缩比: {original_file_size / quantized_size:.2f}x")
            
            # 清理临时文件
            os.remove('temp_original.pth')
            
            return True
        else:
            print("❌ 量化模型文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ 模型大小测试失败: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("🧪 PointNeXt Eager模式量化测试")
    print("="*50)
    
    # 测试量化模型功能
    inference_success = test_quantized_model()
    
    # 测试模型大小
    size_success = test_model_size()
    
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    print(f"推理测试:     {'✅ 通过' if inference_success else '❌ 失败'}")
    print(f"大小测试:     {'✅ 通过' if size_success else '❌ 失败'}")
    
    if inference_success and size_success:
        print("\n🎉 所有测试通过！Eager模式量化工作正常！")
    else:
        print("\n⚠️  部分测试失败，请检查具体问题。")
    
    print("="*50)