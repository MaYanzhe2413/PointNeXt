#!/usr/bin/env python3
"""
简化版 PointNeXt 量化脚本
直接从配置文件构建模型并进行量化，避免复杂的配置解析
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path

# 添加路径
sys.path.append('openpoints')

def simple_quantize_pointnext(config_file, pretrained_path=None, output_path=None):
    """
    简化的PointNeXt量化函数
    
    Args:
        config_file: 配置文件路径 
        pretrained_path: 预训练模型路径
        output_path: 输出路径
    """
    
    print("🚀 开始简化量化流程")
    print(f"📁 配置文件: {config_file}")
    print(f"💾 预训练模型: {pretrained_path if pretrained_path else '无'}")
    
    # 1. 直接导入和构建模型
    try:
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig
        
        # 加载配置
        cfg = EasyConfig()
        cfg.load(config_file)
        
        # 构建模型
        print("🏗️  构建模型...")
        model = build_model_from_cfg(cfg.model)
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"📥 加载预训练权重...")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 加载权重 (忽略不匹配的键)
            model.load_state_dict(state_dict, strict=False)
            print("✅ 权重加载完成")
            
        model.eval()
        print(f"✅ 模型构建完成")
        
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return None
    
    # 2. 创建示例输入
    print("📊 创建校准数据...")
    
    def create_sample_input():
        """创建示例输入数据"""
        batch_size = 1
        num_points = 1024
        
        # 创建点云数据
        pos = torch.randn(batch_size, num_points, 3)
        features = torch.randn(batch_size, num_points, 3)  # RGB特征
        
        # 构建输入字典
        sample_input = {
            'pos': pos,
            'x': features
        }
        
        return sample_input
    
    sample_input = create_sample_input()
    
    # 3. 测试模型前向传播
    print("🧪 测试模型前向传播...")
    try:
        with torch.no_grad():
            original_output = model(sample_input)
        print("✅ 前向传播测试成功")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return None
    
    # 4. 使用静态量化
    print("🔥 开始静态量化...")
    
    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 递归设置子模块的量化配置
    def set_qconfig_recursive(module):
        for name, child in module.named_children():
            if hasattr(child, 'qconfig'):
                child.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            set_qconfig_recursive(child)
    
    set_qconfig_recursive(model)
    
    # 准备量化
    print("⚙️  准备量化...")
    try:
        model_prepared = torch.quantization.prepare(model)
        print("✅ 量化准备完成")
    except Exception as e:
        print(f"❌ 量化准备失败: {e}")
        return None
    
    # 校准（使用多个样本）
    print("📈 开始校准...")
    model_prepared.eval()
    
    with torch.no_grad():
        for i in range(10):  # 使用10个样本进行校准
            try:
                # 创建不同的校准样本
                cal_input = create_sample_input()
                _ = model_prepared(cal_input)
            except Exception as e:
                print(f"⚠️  校准样本 {i} 失败: {e}")
                continue
    
    print("✅ 校准完成")
    
    # 转换为量化模型
    print("🔄 转换为量化模型...")
    try:
        quantized_model = torch.quantization.convert(model_prepared)
        print("✅ 量化转换成功")
    except Exception as e:
        print(f"❌ 量化转换失败: {e}")
        return None
    
    # 5. 验证量化模型
    print("🧪 验证量化模型...")
    try:
        with torch.no_grad():
            quantized_output = quantized_model(sample_input)
        print("✅ 量化模型验证成功")
    except Exception as e:
        print(f"❌ 量化模型验证失败: {e}")
        return None
    
    # 6. 性能对比
    print("📊 性能对比...")
    
    def measure_inference_time(model, input_data, num_runs=50):
        """测量推理时间"""
        model.eval()
        total_time = 0
        
        with torch.no_grad():
            # 预热
            for _ in range(5):
                _ = model(input_data)
            
            # 正式测量
            import time
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_data)
                total_time += time.time() - start_time
        
        return (total_time / num_runs) * 1000  # 返回毫秒
    
    def get_model_size(model):
        """计算模型大小 (MB)"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / 1024 / 1024
    
    # 测量性能
    original_time = measure_inference_time(model, sample_input)
    quantized_time = measure_inference_time(quantized_model, sample_input)
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    # 输出结果
    print("\n" + "="*50)
    print("📊 量化结果对比")
    print("="*50)
    print(f"📈 推理时间:")
    print(f"  原始模型: {original_time:.2f} ms")
    print(f"  量化模型: {quantized_time:.2f} ms")
    print(f"  速度提升: {original_time/quantized_time:.2f}x")
    
    print(f"\n💾 模型大小:")
    print(f"  原始模型: {original_size:.2f} MB")
    print(f"  量化模型: {quantized_size:.2f} MB")  
    print(f"  大小压缩: {original_size/quantized_size:.2f}x")
    
    # 7. 保存量化模型
    if output_path:
        print(f"\n💾 保存量化模型到: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.save({
            'model': quantized_model,
            'model_state_dict': quantized_model.state_dict(),
            'config': cfg.model,
            'performance': {
                'original_time': original_time,
                'quantized_time': quantized_time,
                'original_size': original_size,
                'quantized_size': quantized_size,
                'speed_up': original_time/quantized_time,
                'compression': original_size/quantized_size
            }
        }, output_path)
        print("✅ 模型保存成功")
    
    print("\n🎉 量化完成!")
    
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='PointNeXt 简化量化脚本')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--pretrained', type=str, help='预训练模型路径')
    parser.add_argument('--output', type=str, help='输出路径')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.cfg):
        print(f"❌ 配置文件不存在: {args.cfg}")
        return
    
    # 设置默认输出路径
    if not args.output:
        config_name = Path(args.cfg).stem
        args.output = f"quantized_models/{config_name}_quantized.pth"
    
    # 开始量化
    try:
        quantized_model = simple_quantize_pointnext(
            args.cfg, 
            args.pretrained, 
            args.output
        )
        
        if quantized_model:
            print(f"\n✅ 量化成功完成!")
            print(f"📁 量化模型保存在: {args.output}")
        else:
            print(f"\n❌ 量化失败!")
            
    except Exception as e:
        print(f"❌ 量化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
