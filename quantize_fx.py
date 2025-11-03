#!/usr/bin/env python3
"""
PointNeXt PyTorch FX 量化脚本
简化版本，直接使用PyTorch FX进行图模式量化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.fx import symbolic_trace
# 兼容不同PyTorch版本的导入
try:
    from torch.quantization import get_default_qconfig_mapping
    from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
except ImportError:
    # PyTorch 1.10及以下版本的兼容性导入
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
    except ImportError:
        # 最基础的量化API
        from torch.quantization.quantize_fx import prepare_fx, convert_fx
        try:
            from torch.quantization.quantize_fx import prepare_qat_fx
        except ImportError:
            prepare_qat_fx = None
        
        def get_default_qconfig_mapping(backend='fbgemm'):
            """兼容旧版本的qconfig mapping"""
            if backend == 'fbgemm':
                return torch.quantization.get_default_qconfig('fbgemm')
            elif backend == 'qnnpack':
                return torch.quantization.get_default_qconfig('qnnpack')
            else:
                return torch.quantization.get_default_qconfig('fbgemm')

import torch.quantization.observer as observer
import torch.optim as optim
import copy
import numpy as np
import yaml
import argparse
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

# 添加openpoints到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'openpoints'))

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from openpoints.dataset import build_dataloader_from_cfg

# 导入PointNeXt中需要跳过量化的层类
try:
    from openpoints.models.layers.subsample import FurthestPointSampling
    from openpoints.models.layers.group import QueryAndGroup, BallQuery, GroupingOperation
    FX_LAYER_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️ 无法导入PointNeXt层类，将使用字符串匹配方式跳过层")
    FurthestPointSampling = None
    QueryAndGroup = None
    BallQuery = None
    GroupingOperation = None
    FX_LAYER_IMPORTS_AVAILABLE = False

# 导入FX补丁
try:
    from fx_subsample_patch import apply_fx_patches
    FX_PATCH_AVAILABLE = True
except ImportError:
    print("⚠️ FX补丁不可用")
    FX_PATCH_AVAILABLE = False


class SimplePointNeXtQuantizer:
    """
    简化的PointNeXt量化器
    使用PyTorch FX进行图模式量化
    """
    
    def __init__(self, config_path: str, pretrained_path: str = None):
        """
        初始化量化器
        
        Args:
            config_path: 配置文件路径
            pretrained_path: 预训练模型路径
        """
        self.cfg = EasyConfig()
        self.cfg.load(config_path)
        self.pretrained_path = pretrained_path
        
        # 设置量化配置 - 兼容不同PyTorch版本
        try:
            self.qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        except:
            # 旧版本的量化配置
            self.qconfig_mapping = torch.quantization.get_default_qconfig('fbgemm')
        
        print(f"🔧 加载配置文件: {config_path}")
        print(f"📦 预训练模型: {pretrained_path if pretrained_path else '无'}")
        
    def build_model(self) -> nn.Module:
        """构建模型"""
        print("🏗️  构建模型...")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  模型设备: {device}")
        
        # 构建模型
        model = build_model_from_cfg(self.cfg.model)
        
        # 加载预训练权重
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"📥 加载预训练权重: {self.pretrained_path}")
            checkpoint = torch.load(self.pretrained_path, map_location=device)
            
            # 处理不同的checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 移除不匹配的键
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            # 找到匹配的键
            matched_keys = model_keys.intersection(checkpoint_keys)
            unmatched_model_keys = model_keys - checkpoint_keys
            unmatched_checkpoint_keys = checkpoint_keys - model_keys
            
            print(f"✅ 匹配的参数: {len(matched_keys)}")
            if unmatched_model_keys:
                print(f"⚠️  模型中未匹配的参数: {len(unmatched_model_keys)}")
            if unmatched_checkpoint_keys:
                print(f"⚠️  checkpoint中未匹配的参数: {len(unmatched_checkpoint_keys)}")
            
            # 加载匹配的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in matched_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
        
        # 将模型移到GPU
        model = model.to(device)
        model.eval()
        print(f"✅ 模型构建完成: {type(model).__name__}")
        return model

    def qat_train_model(self, model, train_loader, num_epochs=3, lr=0.001):
        """
        QAT训练过程
        
        Args:
            model: 已经准备好的QAT模型
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            lr: 学习率
        """
        print(f"🔥 开始QAT训练，训练轮数: {num_epochs}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  训练设备: {device}")
        
        # 将模型移到GPU
        model = model.to(device)
        
        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 设置损失函数
        if hasattr(self.cfg.model, 'cls_args'):
            # 分类任务
            criterion = nn.CrossEntropyLoss().to(device)
            task_type = 'classification'
        else:
            # 分割任务
            criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
            task_type = 'segmentation'
        
        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            print(f"📈 Epoch {epoch+1}/{num_epochs}")
            
            for i, data in enumerate(train_loader):
                if i >= 20:  # 限制每个epoch的batch数量，用于快速验证
                    break
                
                try:
                    optimizer.zero_grad()
                    
                    # 处理输入数据并移到GPU
                    if isinstance(data, dict):
                        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
                        if task_type == 'classification':
                            targets = inputs.get('y', torch.randint(0, 40, (inputs['pos'].shape[0],))).to(device)
                        else:
                            targets = inputs.get('y', torch.randint(0, 13, inputs['pos'].shape[:2])).to(device)
                    else:
                        inputs = data[0] if isinstance(data, (list, tuple)) else data
                        if torch.is_tensor(inputs):
                            inputs = inputs.to(device)
                        elif isinstance(inputs, dict):
                            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                        
                        if task_type == 'classification':
                            targets = torch.randint(0, 40, (inputs['pos'].shape[0] if isinstance(inputs, dict) else inputs.shape[0],)).to(device)
                        else:
                            targets = torch.randint(0, 13, inputs['pos'].shape[:2] if isinstance(inputs, dict) else inputs.shape[:2]).to(device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    
                    # 计算损失
                    if task_type == 'classification':
                        loss = criterion(outputs, targets.long())
                    else:
                        # 分割任务需要reshape
                        outputs = outputs.view(-1, outputs.shape[-1])
                        targets = targets.view(-1)
                        loss = criterion(outputs, targets.long())
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if i % 10 == 0:
                        print(f"  Batch {i}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"⚠️  训练批次 {i} 失败: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"✅ Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        
        print("✅ QAT训练完成")
        return model

    def qat_version_model(self, model, train_loader):
        """
        FX QAT量化流程 - 专注调试
        """
        print("🔥 开始FX QAT量化准备...")
        
        # 检查FX QAT支持
        if prepare_qat_fx is None:
            raise RuntimeError("当前PyTorch版本不支持FX QAT")
        
        try:
            # 1. 复制模型并准备QAT
            model_to_quantize = copy.deepcopy(model)
            model_to_quantize.eval()  # QAT准备时需要eval模式
            
            # 2. 应用FX兼容补丁
            restore_patches = None
            if FX_PATCH_AVAILABLE:
                restore_patches = apply_fx_patches()
            
            # 3. 创建示例输入
            example_inputs = self._create_example_input()
            print(f"📝 示例输入形状: {example_inputs['pos'].shape}")
            
            # 4. 尝试符号化追踪，添加详细调试信息
            print("🔍 开始符号化追踪...")
            try:
                # 先分析模型结构
                self._analyze_model_structure(model_to_quantize)
                
                traced_model = symbolic_trace(model_to_quantize)
                print("✅ 符号化追踪成功")
                print(f"📊 追踪图节点数: {len(traced_model.graph.nodes)}")
            except Exception as trace_error:
                print(f"❌ 符号化追踪失败: {trace_error}")
                print("🔍 尝试分析失败原因...")
                
                # 详细分析失败原因
                self._debug_trace_failure(model_to_quantize, example_inputs)
                raise trace_error
            
            # 4. 准备QAT配置
            qconfig_dict = {
                "": torch.quantization.get_default_qat_qconfig('fbgemm'),
            }
            
            # 添加object_type配置来跳过特定类型的层
            if FX_LAYER_IMPORTS_AVAILABLE:
                qconfig_dict["object_type"] = [
                    (FurthestPointSampling, None),  # 跳过FPS层
                    (QueryAndGroup, None),          # 跳过查询和分组层
                    (BallQuery, None),              # 跳过球查询层
                    (GroupingOperation, None),      # 跳过分组操作层
                ]
                print("🚫 配置跳过的层类型:")
                print("  - FurthestPointSampling (最远点采样)")
                print("  - QueryAndGroup (查询和分组)")
                print("  - BallQuery (球查询)")
                print("  - GroupingOperation (分组操作)")
            else:
                print("⚠️ 无法使用object_type配置，将在后续使用模块名匹配")
            
            # 添加基于模块名的跳过配置（更精确的控制）
            # 遍历模型找到需要跳过的具体模块
            skip_module_names = []
            for name, module in model_to_quantize.named_modules():
                module_type = type(module).__name__
                if any(pattern in module_type for pattern in [
                    'FurthestPointSampling', 'QueryAndGroup', 'BallQuery', 
                    'GroupingOperation', 'GroupAll', 'KNNGroup'
                ]):
                    skip_module_names.append(name)
                    qconfig_dict[name] = None  # 跳过这个具体模块
            
            if skip_module_names:
                print("🚫 基于模块名跳过的层:")
                for name in skip_module_names[:5]:  # 只显示前5个
                    print(f"  - {name}")
                if len(skip_module_names) > 5:
                    print(f"  - ... 以及其他 {len(skip_module_names) - 5} 个模块")
            
            print(f"📊 量化配置统计: 跳过 {len(skip_module_names)} 个模块")
            
            # 5. 准备QAT模型
            print("🔧 准备QAT模型...")
            model_prepared = prepare_qat_fx(traced_model, qconfig_dict, example_inputs)
            print("✅ QAT模型准备完成")
            
            # 6. QAT训练
            model_trained = self.qat_train_model(model_prepared, train_loader, num_epochs=3)
            
            # 7. 转换为量化模型
            print("🔄 转换为量化模型...")
            model_trained.eval()  # 转换前必须设置为eval模式
            quantized_model = convert_fx(model_trained)
            print("✅ FX QAT量化转换完成")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ FX QAT量化失败: {e}")
            print("� 详细错误信息:")
            import traceback
            traceback.print_exc()
            raise e

    def _legacy_qat_quantize(self, model, train_loader):
        """
        传统QAT量化方法 - 兼容旧版本PyTorch
        """
        print("🔧 使用传统QAT量化方法...")
        
        try:
            # 1. 复制模型并移到CPU进行量化
            model_to_quantize = copy.deepcopy(model)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"⚠️  注意：量化操作需要在CPU上进行，训练在{device}上进行")
            
            # 先在GPU上训练，然后移到CPU量化
            model_to_quantize = model_to_quantize.to(device)
            
            # 2. 设置QAT配置
            model_to_quantize.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            # 3. 准备QAT
            model_prepared = torch.quantization.prepare_qat(model_to_quantize)
            print("✅ 传统QAT模型准备完成")
            
            # 4. QAT训练（在GPU上）
            model_trained = self.qat_train_model(model_prepared, train_loader, num_epochs=3)
            
            # 5. 转换为量化模型（移到CPU进行）
            model_trained.eval()
            model_trained = model_trained.cpu()  # 移到CPU进行量化转换
            print("🔄 将模型移至CPU进行量化转换...")
            
            quantized_model = torch.quantization.convert(model_trained)
            print("✅ 传统QAT量化转换完成")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ 传统QAT量化也失败: {e}")
            print("🔄 回退到静态量化...")
            # 确保模型在CPU上进行静态量化
            model_cpu = model.cpu()
            return self.quantize_model(model_cpu, train_loader)

    def _analyze_model_structure(self, model):
        """分析模型结构以找出FX追踪失败的原因"""
        print("🔍 模型结构分析:")
        
        # 1. 检查模型层级
        print("📋 模型层级结构:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                print(f"  {name}: {type(module).__name__}")
        
        # 2. 检查前向传播中的问题节点
        print("\n🔍 检查问题操作:")
        problematic_ops = []
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if any(op in module_type.lower() for op in ['assert', 'conditional', 'if', 'while']):
                problematic_ops.append((name, module_type))
        
        if problematic_ops:
            print("⚠️  发现可能导致追踪失败的操作:")
            for name, op_type in problematic_ops:
                print(f"    {name}: {op_type}")
        
        # 3. 尝试单步前向传播
        print("\n🔍 尝试单步前向传播调试:")
        try:
            example_input = self._create_example_input()
            with torch.no_grad():
                # 设置hook来捕获每层的输出
                def debug_hook(name):
                    def hook_fn(module, input, output):
                        print(f"  ✅ {name}: {type(module).__name__} -> {type(output)}")
                        if hasattr(output, 'shape'):
                            print(f"     形状: {output.shape}")
                        elif isinstance(output, (list, tuple)):
                            print(f"     输出类型: {type(output)}, 长度: {len(output)}")
                    return hook_fn
                
                # 注册hooks
                hooks = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # 只在叶子节点注册
                        hook = module.register_forward_hook(debug_hook(name))
                        hooks.append(hook)
                
                # 执行前向传播
                output = model(example_input)
                print(f"✅ 前向传播成功，输出形状: {output.shape}")
                
                # 清理hooks
                for hook in hooks:
                    hook.remove()
                    
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _debug_trace_failure(self, model, example_input):
        """调试追踪失败的具体原因"""
        print("🔍 开始追踪失败调试...")
        
        # 1. 尝试逐层追踪
        print("📋 尝试逐层追踪:")
        modules = list(model.named_modules())
        
        for i, (name, module) in enumerate(modules[:10]):  # 只检查前10层
            if len(list(module.children())) == 0:  # 叶子节点
                try:
                    print(f"  测试 {name}: {type(module).__name__}")
                    traced_module = torch.fx.symbolic_trace(module)
                    print(f"    ✅ 可追踪")
                except Exception as e:
                    print(f"    ❌ 不可追踪: {e}")
        
        # 2. 检查模型中的控制流
        print("\n🔍 检查控制流:")
        model_code = str(model.__class__)
        print(f"模型类: {model_code}")
        
        # 3. 尝试部分追踪
        print("\n🔍 尝试部分追踪:")
        try:
            # 检查是否有自定义forward方法
            forward_method = getattr(model, 'forward', None)
            if forward_method:
                import inspect
                source = inspect.getsource(forward_method)
                print("Forward方法源码片段:")
                lines = source.split('\n')[:10]  # 前10行
                for line in lines:
                    print(f"  {line}")
        except Exception as e:
            print(f"无法获取源码: {e}")
    
    def _create_example_input(self):
        """创建示例输入用于模型追踪"""
        batch_size = 1
        num_points = 1024
        
        # PointNeXt正确的输入格式
        pos = torch.randn(batch_size, num_points, 3)  # (B, N, 3)
        
        return {
            'pos': pos
        }
        

    def prepare_data(self) -> torch.utils.data.DataLoader:
        """准备校准数据"""
        print("📊 准备校准数据...")
        
        # 构建数据加载器
        try:
            # 修改配置以获取小批量数据用于校准
            cal_cfg = self.cfg.copy()
            cal_cfg.dataset.common.train.batch_size = 8  # 小批量
            cal_cfg.dataset.common.train.num_workers = 2
            
            # 构建校准数据加载器
            dataloader = build_dataloader_from_cfg(cal_cfg.get('dataset', {}))
            cal_loader = dataloader['train'] if 'train' in dataloader else dataloader
            
            print(f"✅ 校准数据准备完成")
            return cal_loader
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("🔄 使用合成数据进行校准...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """创建合成数据用于校准"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        class SyntheticDataset:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                self.device = device
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # 创建合成点云数据 - 正确的PointNeXt输入格式
                num_points = 1024
                # PointNeXt期望的输入格式: (num_points, 3) for pos
                # 始终在CPU上创建数据，避免设备冲突
                pos = torch.randn(num_points, 3)  
                
                # 根据任务类型返回不同格式
                data = {
                    'pos': pos,
                    'y': torch.randint(0, 40, ())  # 标量形式
                }
                return data
        
        dataset = SyntheticDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        return dataloader
    
    def quantize_model(self, model: nn.Module, calibration_loader) -> nn.Module:
        """
        使用PyTorch FX进行模型量化 - 兼容不同版本
        """
        print("🔥 开始模型量化...")
        
        # 设置量化配置
        model.eval()
        
        # 1. 尝试FX量化
        try:
            return self._fx_quantize(model, calibration_loader)
        except Exception as e:
            print(f"❌ FX量化失败: {e}")
            print("� 尝试传统量化方法...")
            return self._manual_quantize(model, calibration_loader)
    
    def _fx_quantize(self, model: nn.Module, calibration_loader) -> nn.Module:
        """FX量化方法"""
        print("📈 尝试FX量化...")
        
        # 获取示例输入
        sample_data = next(iter(calibration_loader))
        if isinstance(sample_data, dict):
            example_inputs = sample_data
        else:
            example_inputs = sample_data[0] if isinstance(sample_data, (list, tuple)) else sample_data
        
        # 确保输入在CPU上
        if isinstance(example_inputs, dict):
            example_inputs = {k: v.cpu() if torch.is_tensor(v) else v 
                            for k, v in example_inputs.items()}
        
        # 符号化追踪
        traced_model = symbolic_trace(model)
        print("✅ 符号化追踪成功")
        
        # 准备量化 - 兼容不同版本
        try:
            # 新版本API
            qconfig_mapping = self.qconfig_mapping
            if callable(qconfig_mapping):
                # 如果是函数，说明是兼容性包装
                qconfig_dict = {"": qconfig_mapping}
            else:
                qconfig_dict = qconfig_mapping
                
            prepared_model = prepare_fx(traced_model, qconfig_dict, example_inputs)
        except Exception as e:
            print(f"新版FX API失败: {e}, 尝试旧版API...")
            # 旧版本API
            qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
            prepared_model = prepare_fx(traced_model, qconfig_dict, example_inputs)
        
        print("✅ 量化准备完成")
        
        # 校准
        print("📊 开始校准...")
        prepared_model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(calibration_loader):
                if i >= 10:  # 只使用少量数据进行校准
                    break
                    
                try:
                    if isinstance(data, dict):
                        inputs = data
                    else:
                        inputs = data[0] if isinstance(data, (list, tuple)) else data
                    
                    # 确保输入在CPU上
                    if isinstance(inputs, dict):
                        inputs = {k: v.cpu() if torch.is_tensor(v) else v 
                                for k, v in inputs.items()}
                    
                    _ = prepared_model(inputs)
                    
                except Exception as e:
                    print(f"⚠️  校准批次 {i} 失败: {e}")
                    continue
        
        print("✅ 校准完成")
        
        # 转换为量化模型
        print("🔄 转换为量化模型...")
        quantized_model = convert_fx(prepared_model)
        print("✅ FX量化转换成功")
        return quantized_model
    
    def _manual_quantize(self, model: nn.Module, calibration_loader) -> nn.Module:
        """
        手动量化方法（备用方案）
        """
        print("🔧 使用手动量化方法...")
        
        # 确保模型在CPU上进行量化
        model = model.cpu()
        print("🔄 将模型移至CPU进行量化...")
        
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        torch.quantization.prepare(model, inplace=True)
        
        # 校准
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(calibration_loader):
                if i >= 10:  # 少量校准数据
                    break
                try:
                    if isinstance(data, dict):
                        # 将数据移到CPU
                        inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in data.items()}
                    else:
                        inputs = data[0] if isinstance(data, (list, tuple)) else data
                        if torch.is_tensor(inputs):
                            inputs = inputs.cpu()
                        elif isinstance(inputs, dict):
                            inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
                    
                    _ = model(inputs)
                except:
                    continue
        
        # 转换为量化模型
        torch.quantization.convert(model, inplace=True)
        
        print("✅ 手动量化完成")
        return model
    
    def evaluate_model(self, model: nn.Module, test_loader, model_name: str = "模型"):
        """评估模型性能"""
        print(f"📊 评估{model_name}性能...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检测是否为量化模型
        is_quantized = any(hasattr(m, '_weight_bias') or 'quantized' in str(type(m)).lower() 
                          for m in model.modules())
        
        if is_quantized:
            # 量化模型只能在CPU上运行
            print("🔄 检测到量化模型，将在CPU上评估...")
            model = model.cpu()
            eval_device = torch.device('cpu')
        else:
            # 原始模型可以在GPU上运行
            model = model.to(device)
            eval_device = device
            
        model.eval()
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= 20:  # 只测试少量批次
                    break
                
                try:
                    if isinstance(data, dict):
                        inputs = {k: v.to(eval_device) if torch.is_tensor(v) else v for k, v in data.items()}
                    else:
                        inputs = data[0] if isinstance(data, (list, tuple)) else data
                        if torch.is_tensor(inputs):
                            inputs = inputs.to(eval_device)
                        elif isinstance(inputs, dict):
                            inputs = {k: v.to(eval_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    
                    # 计时
                    if eval_device.type == 'cuda':
                        start_time = torch.cuda.Event(enable_timing=True)
                        end_time = torch.cuda.Event(enable_timing=True)
                        start_time.record()
                    else:
                        import time
                        start = time.time()
                    
                    _ = model(inputs)
                    
                    if eval_device.type == 'cuda':
                        end_time.record()
                        torch.cuda.synchronize()
                        batch_time = start_time.elapsed_time(end_time)
                    else:
                        batch_time = (time.time() - start) * 1000  # 转换为毫秒
                    
                    total_time += batch_time
                    num_batches += 1
                    
                except Exception as e:
                    print(f"⚠️  评估批次 {i} 失败: {e}")
                    continue
        
        avg_time = total_time / num_batches if num_batches > 0 else 0
        print(f"📈 {model_name}平均推理时间: {avg_time:.2f} ms (设备: {eval_device})")
        return avg_time
    
    def compare_models(self, original_model: nn.Module, quantized_model: nn.Module, 
                      test_loader):
        """比较原始模型和量化模型"""
        print("\n" + "="*50)
        print("📊 模型性能对比")
        print("="*50)
        
        # 评估原始模型
        original_time = self.evaluate_model(original_model, test_loader, "原始模型")
        
        # 评估量化模型
        quantized_time = self.evaluate_model(quantized_model, test_loader, "量化模型")
        
        # 计算模型大小
        def get_model_size(model):
            total_params = sum(p.numel() * p.element_size() for p in model.parameters())
            total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
            return (total_params + total_buffers) / 1024 / 1024  # MB
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        # 输出对比结果
        print(f"\n📈 性能对比:")
        print(f"  原始模型推理时间: {original_time:.2f} ms")
        print(f"  量化模型推理时间: {quantized_time:.2f} ms")
        print(f"  速度提升: {original_time/quantized_time:.2f}x" if quantized_time > 0 else "  速度提升: N/A")
        
        print(f"\n💾 模型大小对比:")
        print(f"  原始模型大小: {original_size:.2f} MB")
        print(f"  量化模型大小: {quantized_size:.2f} MB")
        print(f"  大小压缩: {original_size/quantized_size:.2f}x" if quantized_size > 0 else "  大小压缩: N/A")
        
        return {
            'original_time': original_time,
            'quantized_time': quantized_time,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'speed_up': original_time/quantized_time if quantized_time > 0 else 0,
            'compression': original_size/quantized_size if quantized_size > 0 else 0
        }
    
    def save_quantized_model(self, quantized_model: nn.Module, save_path: str):
        """保存量化模型"""
        print(f"💾 保存量化模型到: {save_path}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            # 尝试保存完整模型（包括量化信息）
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'quantization_info': 'QAT quantized model',
                'model_class': type(quantized_model).__name__
            }, save_path)
            print("✅ 量化模型保存成功")
        except Exception as e:
            print(f"⚠️  保存完整模型失败: {e}")
            try:
                # 只保存状态字典
                torch.save(quantized_model.state_dict(), save_path.replace('.pth', '_state_dict.pth'))
                print(f"💾 模型状态字典已保存到: {save_path.replace('.pth', '_state_dict.pth')}")
                print("✅ 量化模型状态字典保存成功")
            except Exception as e2:
                print(f"❌ 保存模型失败: {e2}")
    
    def run_quantization(self, save_path: str = None):
        """运行完整的量化流程"""
        print("\n" + "🚀 开始PointNeXt量化流程" + "\n" + "="*50)
        
        # 1. 构建模型
        original_model = self.build_model()
        
        # 2. 准备数据
        calibration_loader = self.prepare_data()
        
        # 3. 量化模型
        quantized_model = self.quantize_model(original_model.cpu(), calibration_loader)
        
        # 4. 性能对比
        results = self.compare_models(original_model, quantized_model, calibration_loader)
        
        # 5. 保存模型
        if save_path:
            self.save_quantized_model(quantized_model, save_path)
        
        print("\n" + "🎉 量化流程完成!" + "\n" + "="*50)
        
        return quantized_model, results

    def run_qat_quantization(self, save_path: str = None, num_epochs: int = 3):
        """运行完整的QAT量化流程"""
        print("\n" + "🚀 开始PointNeXt QAT量化流程" + "\n" + "="*50)
        
        # 1. 构建模型
        original_model = self.build_model()
        
        # 2. 准备训练数据
        train_loader = self.prepare_data()
        
        # 3. 运行QAT量化（包含训练过程）
        print("🔥 开始QAT量化训练...")
        quantized_model = self.qat_version_model(original_model, train_loader)
        
        # 4. 性能对比
        results = self.compare_models(original_model, quantized_model, train_loader)
        
        # 5. 保存模型
        if save_path:
            # 修改保存路径以区分QAT模型
            qat_save_path = save_path.replace('.pth', '_qat.pth')
            self.save_quantized_model(quantized_model, qat_save_path)
            print(f"💾 QAT量化模型保存到: {qat_save_path}")
        
        print("\n" + "🎉 QAT量化流程完成!" + "\n" + "="*50)
        
        return quantized_model, results

    def compare_quantization_methods(self, save_path: str = None):
        """对比静态量化和QAT量化的效果"""
        print("\n" + "🔬 开始量化方法对比" + "\n" + "="*50)
        
        # 1. 构建原始模型
        original_model = self.build_model()
        data_loader = self.prepare_data()
        
        # 2. 静态量化
        print("\n📊 运行静态量化...")
        static_quantized = self.quantize_model(original_model.cpu(), data_loader)
        static_results = self.compare_models(original_model, static_quantized, data_loader)
        
        # 3. QAT量化
        print("\n🎯 运行QAT量化...")
        qat_quantized = self.qat_version_model(copy.deepcopy(original_model), data_loader)
        qat_results = self.compare_models(original_model, qat_quantized, data_loader)
        
        # 4. 对比结果
        print("\n" + "="*60)
        print("📊 量化方法对比结果")
        print("="*60)
        
        print(f"🔹 原始模型:")
        print(f"  推理时间: {static_results['original_time']:.2f} ms")
        print(f"  模型大小: {static_results['original_size']:.2f} MB")
        
        print(f"\n🔹 静态量化:")
        print(f"  推理时间: {static_results['quantized_time']:.2f} ms")
        print(f"  模型大小: {static_results['quantized_size']:.2f} MB")
        print(f"  速度提升: {static_results['speed_up']:.2f}x")
        print(f"  大小压缩: {static_results['compression']:.2f}x")
        
        print(f"\n🔹 QAT量化:")
        print(f"  推理时间: {qat_results['quantized_time']:.2f} ms")
        print(f"  模型大小: {qat_results['quantized_size']:.2f} MB")
        print(f"  速度提升: {qat_results['speed_up']:.2f}x")
        print(f"  大小压缩: {qat_results['compression']:.2f}x")
        
        # 对比静态量化和QAT
        speed_diff = qat_results['speed_up'] / static_results['speed_up']
        size_diff = qat_results['compression'] / static_results['compression']
        
        print(f"\n🔸 QAT vs 静态量化:")
        print(f"  速度对比: {speed_diff:.2f}x {'(QAT更快)' if speed_diff > 1 else '(静态更快)'}")
        print(f"  压缩对比: {size_diff:.2f}x {'(QAT压缩更好)' if size_diff > 1 else '(静态压缩更好)'}")
        
        # 5. 保存两个模型
        if save_path:
            static_path = save_path.replace('.pth', '_static.pth')
            qat_path = save_path.replace('.pth', '_qat.pth')
            
            self.save_quantized_model(static_quantized, static_path)
            self.save_quantized_model(qat_quantized, qat_path)
            
            print(f"\n💾 模型保存:")
            print(f"  静态量化: {static_path}")
            print(f"  QAT量化: {qat_path}")
        
        return {
            'static': static_results,
            'qat': qat_results,
            'comparison': {
                'speed_ratio': speed_diff,
                'compression_ratio': size_diff
            }
        }


def main():
    parser = argparse.ArgumentParser(description='PointNeXt PyTorch FX 量化')
    parser.add_argument('--cfg', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='预训练模型路径')
    parser.add_argument('--save_path', type=str, default='quantized_models/quantized_model.pth',
                       help='量化模型保存路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备 (cpu/cuda)')
    parser.add_argument('--method', type=str, default='static', 
                       choices=['static', 'qat', 'compare'],
                       help='量化方法: static(静态量化), qat(QAT量化), compare(对比两种方法)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='QAT训练轮数')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.cfg):
        print(f"❌ 配置文件不存在: {args.cfg}")
        return
    
    # 检查预训练模型
    if args.pretrained and not os.path.exists(args.pretrained):
        print(f"⚠️  预训练模型不存在: {args.pretrained}")
        print("🔄 将使用随机初始化的模型进行量化")
        args.pretrained = None
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = 'cpu'
    elif args.device == 'cpu' and torch.cuda.is_available():
        print("💡 检测到CUDA可用，建议使用 --device cuda 以获得更好性能")
    
    print(f"🖥️  使用设备: {args.device}")
    print(f"⚙️  量化方法: {args.method}")
    
    # 设置默认设备
    if args.device == 'cuda':
        torch.cuda.set_device(0)  # 使用第一个GPU
    
    # 开始量化
    try:
        quantizer = SimplePointNeXtQuantizer(args.cfg, args.pretrained)
        
        if args.method == 'static':
            # 静态量化
            quantized_model, results = quantizer.run_quantization(args.save_path)
            print(f"\n🎯 静态量化总结:")
            print(f"  速度提升: {results['speed_up']:.2f}x")
            print(f"  模型压缩: {results['compression']:.2f}x")
            
        elif args.method == 'qat':
            # QAT量化
            quantized_model, results = quantizer.run_qat_quantization(args.save_path, args.epochs)
            print(f"\n🎯 QAT量化总结:")
            print(f"  速度提升: {results['speed_up']:.2f}x")
            print(f"  模型压缩: {results['compression']:.2f}x")
            
        elif args.method == 'compare':
            # 对比两种方法
            comparison_results = quantizer.compare_quantization_methods(args.save_path)
            print(f"\n🏆 最佳量化方法推荐:")
            
            static_score = comparison_results['static']['speed_up'] + comparison_results['static']['compression']
            qat_score = comparison_results['qat']['speed_up'] + comparison_results['qat']['compression']
            
            if qat_score > static_score:
                print(f"  🥇 推荐QAT量化 (综合得分: {qat_score:.2f})")
                print(f"  🥈 静态量化 (综合得分: {static_score:.2f})")
            else:
                print(f"  🥇 推荐静态量化 (综合得分: {static_score:.2f})")
                print(f"  🥈 QAT量化 (综合得分: {qat_score:.2f})")
        
    except Exception as e:
        print(f"❌ 量化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
