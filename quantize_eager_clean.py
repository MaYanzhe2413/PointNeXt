#!/usr/bin/env python3
"""
PointNeXt Eager模式量化脚本（仅使用真实ModelNet40数据）
使用PyTorch Eager模式进行量化，完全兼容CUDA操作和控制流
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import warnings
import argparse
from typing import Dict, Any, Optional, Tuple

# 兼容不同PyTorch版本的导入
try:
    from torch.quantization import prepare, convert, prepare_qat
    from torch.quantization import get_default_qconfig, get_default_qat_qconfig
    from torch.quantization import QuantStub, DeQuantStub
except ImportError:
    print("❌ PyTorch量化模块导入失败，请检查PyTorch版本")
    sys.exit(1)

warnings.filterwarnings("ignore")

# 添加openpoints到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'openpoints'))

try:
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig
    from openpoints.dataset import build_dataloader_from_cfg
except ImportError:
    print("❌ 无法导入PointNeXt模块，请检查安装")
    sys.exit(1)


class EagerQuantizationWrapper(nn.Module):
    """
    Eager模式量化包装器
    为PointNeXt模型添加量化/反量化操作
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()      # 输入量化
        self.model = model            # 原始模型
        self.dequant = DeQuantStub()  # 输出反量化
        
    def forward(self, data):
        # 处理PointNeXt的标准数据格式
        if isinstance(data, dict):
            # 量化位置信息
            if 'pos' in data and data['pos'] is not None:
                data['pos'] = self.quant(data['pos'])
            # 量化特征信息（如果存在）
            if 'x' in data and data['x'] is not None:
                data['x'] = self.quant(data['x'])
            
            # 模型前向传播
            output = self.model(data)
            
            # 反量化输出
            if isinstance(output, torch.Tensor):
                output = self.dequant(output)
            elif isinstance(output, dict):
                # 处理分类输出
                if 'logits' in output:
                    output['logits'] = self.dequant(output['logits'])
                # 处理其他可能的输出格式
                elif 'out' in output:
                    output['out'] = self.dequant(output['out'])
                
        else:
            # 兼容简单tensor输入
            if isinstance(data, torch.Tensor):
                # 假设是 [B, N, 3] 或 [N, 3] 格式，转换为PointNeXt期望的字典格式
                if data.dim() == 2:  # [N, 3]
                    data = {'pos': self.quant(data)}
                elif data.dim() == 3:  # [B, N, 3]
                    data = {'pos': self.quant(data.squeeze(0))}
                else:
                    data = self.quant(data)
            
            output = self.model(data)
            
            if isinstance(output, torch.Tensor):
                output = self.dequant(output)
            elif isinstance(output, dict) and 'logits' in output:
                output['logits'] = self.dequant(output['logits'])
                
        return output


class PointNeXtEagerQuantizer:
    """
    PointNeXt Eager模式量化器
    支持静态量化和QAT量化（仅使用真实ModelNet40数据）
    """
    
    def __init__(self, config_path: str, pretrained_path: str = None, device: str = 'cuda'):
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.cfg = EasyConfig()
        self.cfg.load(config_path, recursive=True)
        
        print(f"🎯 Eager模式量化器初始化（仅使用真实ModelNet40数据）")
        print(f"📁 配置文件: {config_path}")
        print(f"🔧 设备: {self.device}")
        
    def build_model(self) -> nn.Module:
        """构建模型"""
        print("🏗️  构建模型...")
        
        # 构建原始模型
        model = build_model_from_cfg(self.cfg.model)
        
        # 加载预训练权重
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"📦 加载预训练权重: {self.pretrained_path}")
            checkpoint = torch.load(self.pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("📦 使用随机初始化权重")
        
        return model
    
    def prepare_calibration_data(self) -> torch.utils.data.DataLoader:
        """准备校准数据 - 仅使用真实的ModelNet40数据"""
        print(f"📊 准备ModelNet40校准数据...")
        
        # 从默认ModelNet40配置文件加载数据集配置
        modelnet_cfg = EasyConfig()
        modelnet_cfg.load('cfgs/modelnet40ply2048/default.yaml')
        
        # 使用配置文件中的dataset配置
        dataset_cfg = modelnet_cfg.dataset
        dataloader_cfg = modelnet_cfg.dataloader
        datatransforms_cfg = modelnet_cfg.datatransforms
        
        # 修改batch_size为1用于校准
        batch_size = 1
        
        print("🔄 构建ModelNet40数据加载器...")
        print(f"   数据路径: {dataset_cfg.common.data_dir}")
        print(f"   点数: {dataset_cfg.train.num_points}")
        
        # 构建数据加载器
        dataloader = build_dataloader_from_cfg(
            batch_size=batch_size,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            datatransforms_cfg=datatransforms_cfg,
            split='train',
            distributed=False  # 不使用分布式训练
        )
        
        print("✅ 成功构建ModelNet40数据加载器")
        
        # 测试数据加载器
        print("🔍 测试数据加载器...")
        test_iter = iter(dataloader)
        sample_batch = next(test_iter)
        print(f"   样本格式: {type(sample_batch)}")
        if isinstance(sample_batch, (list, tuple)):
            data, label = sample_batch
            print(f"   数据类型: {type(data)}")
            print(f"   数据形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"   标签形状: {label.shape if hasattr(label, 'shape') else type(label)}")
        
        return dataloader
    
    def _create_qconfig_dict(self, method: str = 'static') -> Dict[str, Any]:
        """
        创建量化配置字典
        Eager模式的配置更简单，不需要处理FX兼容性问题
        """
        if method == 'qat':
            default_qconfig = get_default_qat_qconfig('fbgemm')
        else:
            default_qconfig = get_default_qconfig('fbgemm')
        
        qconfig_dict = {
            '': default_qconfig  # 全局默认配置
        }
        
        print(f"📋 量化配置: {method}模式")
        print(f"   默认qconfig: {default_qconfig}")
        
        return qconfig_dict
    
    def _detect_task_type(self) -> str:
        """检测任务类型"""
        config_path_lower = self.config_path.lower()
        if 'classification' in config_path_lower or 'modelnet' in config_path_lower:
            return 'classification'
        elif 'segmentation' in config_path_lower or 's3dis' in config_path_lower:
            return 'segmentation'
        else:
            return 'classification'  # 默认
    
    def static_quantize(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """静态量化 (Post-Training Quantization)"""
        print("🔄 开始静态量化...")
        
        # 1. 包装模型
        wrapped_model = EagerQuantizationWrapper(model)
        wrapped_model.eval()
        
        # 2. 设置量化配置
        qconfig_dict = self._create_qconfig_dict('static')
        wrapped_model.qconfig = qconfig_dict['']
        
        # 3. 准备量化
        prepared_model = prepare(wrapped_model, inplace=False)
        prepared_model = prepared_model.to(self.device)
        
        print("📊 使用真实ModelNet40数据进行模型校准...")
        # 4. 校准阶段
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 50:  # 限制校准样本数量
                    break
                
                try:
                    # 处理ModelNet40数据格式
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        data, _ = batch
                        # 转换为PointNeXt期望的字典格式
                        if isinstance(data, torch.Tensor):
                            # data是 [B, N, 3] 格式，转换为字典
                            data = {'pos': data.squeeze(0).to(self.device)}  # 去掉batch维度
                        elif isinstance(data, dict):
                            for key in data:
                                if isinstance(data[key], torch.Tensor):
                                    data[key] = data[key].to(self.device)
                    else:
                        print(f"⚠️  未知的数据格式: {type(batch)}")
                        continue
                    
                    # 前向传播进行校准
                    _ = prepared_model(data)
                    
                    if i % 10 == 0:
                        print(f"   校准进度: {i+1}/50")
                        
                except Exception as e:
                    print(f"⚠️  校准样本 {i} 失败: {e}")
                    if i < 5:  # 前几个样本失败时显示详细信息
                        print(f"      批次类型: {type(batch)}")
                        if isinstance(batch, (list, tuple)):
                            print(f"      批次长度: {len(batch)}")
                    continue
        
        # 5. 转换为量化模型
        print("🔄 转换为量化模型...")
        # 转换必须在CPU上进行
        prepared_model_cpu = prepared_model.cpu()
        quantized_model = convert(prepared_model_cpu, inplace=False)
        
        print("✅ 静态量化完成")
        return quantized_model
    
    def qat_quantize(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                     num_epochs: int = 3, lr: float = 0.0001) -> nn.Module:
        """QAT量化 (Quantization-Aware Training)"""
        print(f"🔄 开始QAT量化 (训练{num_epochs}个epoch)...")
        
        # 1. 包装模型
        wrapped_model = EagerQuantizationWrapper(model)
        
        # 2. 设置量化配置
        qconfig_dict = self._create_qconfig_dict('qat')
        wrapped_model.qconfig = qconfig_dict['']
        
        # 3. 准备QAT
        prepared_model = prepare_qat(wrapped_model, inplace=False)
        prepared_model = prepared_model.to(self.device)
        
        # 4. QAT训练
        task_type = self._detect_task_type()
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=lr)
        
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        print("🎯 开始QAT训练...")
        for epoch in range(num_epochs):
            prepared_model.train()
            total_loss = 0
            num_batches = 0
            
            for i, batch in enumerate(train_loader):
                if i >= 30:  # 限制训练批次
                    break
                
                try:
                    optimizer.zero_grad()
                    
                    # 处理输入数据
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        data, targets = batch
                        # 转换为PointNeXt期望的字典格式
                        if isinstance(data, torch.Tensor):
                            data = {'pos': data.squeeze(0).to(self.device)}
                        elif isinstance(data, dict):
                            for key in data:
                                if isinstance(data[key], torch.Tensor):
                                    data[key] = data[key].to(self.device)
                        
                        targets = targets.to(self.device)
                    else:
                        continue
                    
                    # 前向传播
                    outputs = prepared_model(data)
                    
                    # 计算损失
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    
                    if task_type == 'classification':
                        loss = criterion(outputs, targets.long().squeeze())
                    else:
                        # 分割任务
                        outputs = outputs.view(-1, outputs.shape[-1])
                        targets = targets.view(-1)
                        valid_mask = targets != -1
                        if valid_mask.sum() > 0:
                            loss = criterion(outputs[valid_mask], targets[valid_mask].long())
                        else:
                            continue
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if i % 10 == 0:
                        print(f"   Epoch {epoch+1}/{num_epochs}, Batch {i+1}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"⚠️  训练批次 {i} 失败: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"📊 Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
        
        # 5. 转换为量化模型
        print("🔄 转换为量化模型...")
        prepared_model.eval()
        prepared_model_cpu = prepared_model.cpu()
        quantized_model = convert(prepared_model_cpu, inplace=False)
        
        print("✅ QAT量化完成")
        return quantized_model
    
    def evaluate_model(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, 
                      name: str = "模型") -> Dict[str, float]:
        """评估模型性能"""
        print(f"📊 评估{name}...")
        
        # 检测是否为量化模型
        is_quantized = any(hasattr(m, '_weight_bias') or 'quantized' in str(type(m)).lower() 
                          for m in model.modules())
        
        if is_quantized:
            print(f"🔄 检测到量化模型，在CPU上评估...")
            model = model.cpu()
            eval_device = torch.device('cpu')
        else:
            model = model.to(self.device)
            eval_device = self.device
        
        model.eval()
        total_samples = 0
        correct = 0
        total_loss = 0
        
        task_type = self._detect_task_type()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 20:  # 限制评估样本
                    break
                
                try:
                    # 处理ModelNet40数据格式
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        data, targets = batch
                        # 转换为PointNeXt期望的字典格式
                        if isinstance(data, torch.Tensor):
                            data = {'pos': data.squeeze(0).to(eval_device)}
                        elif isinstance(data, dict):
                            for key in data:
                                if isinstance(data[key], torch.Tensor):
                                    data[key] = data[key].to(eval_device)
                        
                        targets = targets.to(eval_device)
                    else:
                        print(f"⚠️  未知的评估数据格式: {type(batch)}")
                        continue
                    
                    # 前向传播
                    outputs = model(data)
                    
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    
                    # 计算准确率和损失
                    if task_type == 'classification':
                        loss = criterion(outputs, targets.long().squeeze())
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == targets.squeeze()).sum().item()
                        total_samples += targets.size(0)
                    else:
                        # 分割任务
                        outputs_flat = outputs.view(-1, outputs.shape[-1])
                        targets_flat = targets.view(-1)
                        valid_mask = targets_flat != -1
                        
                        if valid_mask.sum() > 0:
                            loss = criterion(outputs_flat[valid_mask], targets_flat[valid_mask].long())
                            _, predicted = torch.max(outputs_flat[valid_mask], 1)
                            correct += (predicted == targets_flat[valid_mask]).sum().item()
                            total_samples += valid_mask.sum().item()
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"⚠️  评估批次 {i} 失败: {e}")
                    continue
        
        accuracy = 100 * correct / max(total_samples, 1)
        avg_loss = total_loss / max(i + 1, 1)
        
        print(f"📊 {name}结果:")
        print(f"   准确率: {accuracy:.2f}%")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   评估样本数: {total_samples}")
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'samples': total_samples
        }
    
    def compare_models(self, original_model: nn.Module, quantized_model: nn.Module, 
                      test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """对比原始模型和量化模型"""
        print("🔍 对比原始模型和量化模型...")
        
        # 评估原始模型
        original_results = self.evaluate_model(original_model, test_loader, "原始模型")
        
        # 评估量化模型
        quantized_results = self.evaluate_model(quantized_model, test_loader, "量化模型")
        
        # 计算模型大小
        def get_model_size(model):
            torch.save(model.state_dict(), 'temp_model.pth')
            size = os.path.getsize('temp_model.pth')
            os.remove('temp_model.pth')
            return size
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        # 总结对比结果
        comparison = {
            'original': original_results,
            'quantized': quantized_results,
            'accuracy_drop': original_results['accuracy'] - quantized_results['accuracy'],
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio
        }
        
        print("\n" + "="*50)
        print("📊 模型对比结果")
        print("="*50)
        print(f"原始模型准确率:    {original_results['accuracy']:.2f}%")
        print(f"量化模型准确率:    {quantized_results['accuracy']:.2f}%")
        print(f"准确率下降:        {comparison['accuracy_drop']:.2f}%")
        print(f"原始模型大小:      {comparison['original_size_mb']:.2f} MB")
        print(f"量化模型大小:      {comparison['quantized_size_mb']:.2f} MB")
        print(f"压缩比:           {compression_ratio:.2f}x")
        print("="*50)
        
        return comparison
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            # 确保模型在CPU上进行保存
            model_cpu = model.cpu()
            torch.save(model_cpu.state_dict(), save_path)
            print(f"💾 量化模型已保存: {save_path}")
            
            # 打印文件大小
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"📦 文件大小: {size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='PointNeXt Eager模式量化（仅使用真实ModelNet40数据）')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--method', type=str, choices=['static', 'qat', 'compare'], 
                       default='static', help='量化方法')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--epochs', type=int, default=3, help='QAT训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='QAT学习率')
    parser.add_argument('--pretrained', type=str, help='预训练模型路径')
    parser.add_argument('--save_dir', type=str, default='quantized_models', help='保存目录')
    
    args = parser.parse_args()
    
    print("🚀 开始PointNeXt Eager模式量化 (仅使用真实ModelNet40数据)")
    print("="*60)
    
    # 初始化量化器
    quantizer = PointNeXtEagerQuantizer(
        config_path=args.cfg,
        pretrained_path=args.pretrained,
        device=args.device
    )
    
    # 构建模型
    model = quantizer.build_model()
    
    # 准备数据
    calibration_loader = quantizer.prepare_calibration_data()
    test_loader = calibration_loader  # 使用相同数据进行测试（演示用）
    
    if args.method == 'static':
        # 静态量化
        quantized_model = quantizer.static_quantize(model, calibration_loader)
        
        # 保存模型
        save_path = os.path.join(args.save_dir, 'quantized_model_static_eager.pth')
        quantizer.save_quantized_model(quantized_model, save_path)
        
        # 对比性能
        comparison = quantizer.compare_models(model, quantized_model, test_loader)
        
    elif args.method == 'qat':
        # QAT量化
        quantized_model = quantizer.qat_quantize(
            model, calibration_loader, 
            num_epochs=args.epochs, 
            lr=args.lr
        )
        
        # 保存模型
        save_path = os.path.join(args.save_dir, 'quantized_model_qat_eager.pth')
        quantizer.save_quantized_model(quantized_model, save_path)
        
        # 对比性能
        comparison = quantizer.compare_models(model, quantized_model, test_loader)
        
    elif args.method == 'compare':
        # 比较两种量化方法
        print("🔄 比较静态量化和QAT...")
        
        static_model = quantizer.static_quantize(model, calibration_loader)
        qat_model = quantizer.qat_quantize(copy.deepcopy(model), calibration_loader, args.epochs, args.lr)
        
        print("\n" + "="*60)
        print("📊 静态量化 vs QAT量化对比")
        print("="*60)
        
        static_comparison = quantizer.compare_models(model, static_model, test_loader)
        print(f"\n静态量化结果:")
        print(f"  准确率下降: {static_comparison['accuracy_drop']:.2f}%")
        print(f"  压缩比: {static_comparison['compression_ratio']:.2f}x")
        
        qat_comparison = quantizer.compare_models(copy.deepcopy(model), qat_model, test_loader)
        print(f"\nQAT量化结果:")
        print(f"  准确率下降: {qat_comparison['accuracy_drop']:.2f}%")
        print(f"  压缩比: {qat_comparison['compression_ratio']:.2f}x")
        
        # 保存两个模型
        static_path = os.path.join(args.save_dir, 'quantized_model_static_eager.pth')
        qat_path = os.path.join(args.save_dir, 'quantized_model_qat_eager.pth')
        quantizer.save_quantized_model(static_model, static_path)
        quantizer.save_quantized_model(qat_model, qat_path)
    
    print("✅ 量化完成!")


if __name__ == "__main__":
    main()