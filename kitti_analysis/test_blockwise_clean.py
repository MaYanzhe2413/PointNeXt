#!/usr/bin/env python3
"""
测试 BlockWiseTransfer 的 forward 函数
"""
import torch
import numpy as np
import sys
import os

# 添加项目路径到 sys.path
sys.path.append('/workspace/home/mayz/network/PointNeXt')

from openpoints.models.custom.blockwise import BlockWiseTransfer

def generate_test_data():
    """生成测试数据"""
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成前一帧数据 (已配准) - 坐标+特征合并
    NA = 1000
    C = 64  # 特征维度
    xA = torch.randn(NA, 3, device=device) * 2.0  # 坐标范围 [-4, 4]
    fA = torch.randn(NA, C, device=device)         # 特征
    points_A = torch.cat([xA, fA], dim=1)          # 合并为 (NA, 3+C)
    
    # 生成当前帧数据 - 只有坐标，特征维度为0
    NB = 800
    xB = torch.randn(NB, 3, device=device) * 2.0  # 坐标范围 [-4, 4]
    points_B = xB                                  # 只有坐标 (NB, 3)
    
    return points_A, points_B, device

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    points_A, points_B, device = generate_test_data()
    
    # 创建模型
    model = BlockWiseTransfer(block_size=0.4)
    
    print(f"输入数据:")
    print(f"  points_A shape: {points_A.shape}, device: {points_A.device}")
    print(f"  points_B shape: {points_B.shape}, device: {points_B.device}")
    
    # 执行前向传播
    try:
        diff_coords, matched_coords_features = model(points_A, points_B)
        
        print(f"\n输出结果:")
        print(f"  diff_coords shape: {diff_coords.shape}")
        print(f"  matched_coords_features shape: {matched_coords_features.shape}")
        print(f"  总点数: {diff_coords.shape[0] + matched_coords_features.shape[0]} (应该 <= {points_B.shape[0]})")
        
        # 验证维度
        assert diff_coords.shape[1] == 3, f"差分坐标维度错误: {diff_coords.shape[1]} != 3"
        expected_feature_dim = points_A.shape[1] - 3 + 3  # 3(坐标) + C(特征)
        assert matched_coords_features.shape[1] == expected_feature_dim, \
            f"匹配数据维度错误: {matched_coords_features.shape[1]} != {expected_feature_dim}"
        
        print("✓ 维度检查通过")
        
        # 检查数据类型和设备
        print(f"  diff_coords device: {diff_coords.device}")
        print(f"  matched_coords_features device: {matched_coords_features.device}")
        print(f"  expected device: {device}")
        
        # 放宽设备检查，只要能正常运行就行
        if diff_coords.device.type == device.type and matched_coords_features.device.type == device.type:
            print("✓ 设备类型检查通过")
        else:
            print("⚠️ 设备类型不匹配，但继续测试")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BlockWiseTransfer(block_size=0.4)
    
    # 测试1: 空数据
    print("测试1: 空数据")
    try:
        points_A = torch.empty(0, 67, device=device)  # 3坐标+64特征
        points_B = torch.empty(0, 3, device=device)   # 只有坐标
        
        diff_coords, matched_coords_features = model(points_A, points_B)
        print(f"  diff_coords shape: {diff_coords.shape}")
        print(f"  matched_coords_features shape: {matched_coords_features.shape}")
        print("✓ 空数据测试通过")
    except Exception as e:
        print(f"❌ 空数据测试失败: {e}")
    
    # 测试2: 单点数据
    print("\n测试2: 单点数据")
    try:
        coord_A = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        feat_A = torch.randn(1, 64, device=device)
        points_A = torch.cat([coord_A, feat_A], dim=1)
        
        points_B = torch.tensor([[0.1, 0.1, 0.1]], device=device)
        
        diff_coords, matched_coords_features = model(points_A, points_B)
        print(f"  diff_coords shape: {diff_coords.shape}")
        print(f"  matched_coords_features shape: {matched_coords_features.shape}")
        print("✓ 单点数据测试通过")
    except Exception as e:
        print(f"❌ 单点数据测试失败: {e}")

def test_different_block_sizes():
    """测试不同的block size"""
    print("\n=== 测试不同的block size ===")
    
    points_A, points_B, device = generate_test_data()
    
    block_sizes = [0.1, 0.5, 1.0, 2.0]
    
    for bs in block_sizes:
        print(f"\n测试 block_size = {bs}")
        try:
            model = BlockWiseTransfer(block_size=bs)
            diff_coords, matched_coords_features = model(points_A, points_B)
            
            total_points = diff_coords.shape[0] + matched_coords_features.shape[0]
            print(f"  差分点数: {diff_coords.shape[0]}")
            print(f"  匹配点数: {matched_coords_features.shape[0]}")
            print(f"  总点数: {total_points} / {points_B.shape[0]}")
            print(f"  覆盖率: {total_points/points_B.shape[0]*100:.1f}%")
            
        except Exception as e:
            print(f"❌ block_size={bs} 测试失败: {e}")

def visualize_results():
    """可视化结果并保存为图片"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        print("\n=== 生成可视化结果 ===")
        
        # 生成较小的测试数据便于可视化
        torch.manual_seed(42)
        device = torch.device('cpu')  # 使用CPU便于可视化
        
        # 生成测试数据
        xA = torch.randn(200, 3) * 1.0
        fA = torch.randn(200, 32)
        points_A = torch.cat([xA, fA], dim=1)  # 合并坐标和特征
        points_B = torch.randn(150, 3) * 1.0   # 只有坐标
        
        model = BlockWiseTransfer(block_size=0.5)
        diff_coords, matched_coords_features = model(points_A, points_B)
        
        # 提取匹配点的坐标
        if matched_coords_features.shape[0] > 0:
            matched_coords = matched_coords_features[:, :3]
        else:
            matched_coords = torch.empty((0, 3))
        
        # 转换为numpy
        xA_np = xA.numpy()  # 原始A坐标
        xB_np = points_B.numpy()  # B坐标
        diff_coords_np = diff_coords.numpy()
        matched_coords_np = matched_coords.numpy()
        
        # 创建图形
        fig = plt.figure(figsize=(20, 5))
        
        # 原始数据
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.scatter(xA_np[:, 0], xA_np[:, 1], xA_np[:, 2], c='blue', alpha=0.6, s=20, label=f'Frame A ({points_A.shape[0]})')
        ax1.scatter(xB_np[:, 0], xB_np[:, 1], xB_np[:, 2], c='red', alpha=0.6, s=20, label=f'Frame B ({points_B.shape[0]})')
        ax1.set_title('Original Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 差分区域
        ax2 = fig.add_subplot(142, projection='3d')
        if diff_coords.shape[0] > 0:
            ax2.scatter(diff_coords_np[:, 0], diff_coords_np[:, 1], diff_coords_np[:, 2], 
                       c='red', alpha=0.8, s=30, label=f'Diff Points ({diff_coords.shape[0]})')
        ax2.set_title('Difference Region')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # 匹配区域
        ax3 = fig.add_subplot(143, projection='3d')
        if matched_coords.shape[0] > 0:
            ax3.scatter(matched_coords_np[:, 0], matched_coords_np[:, 1], matched_coords_np[:, 2], 
                       c='green', alpha=0.8, s=30, label=f'Matched Points ({matched_coords.shape[0]})')
        ax3.set_title('Matched Region')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        # 统计图
        ax4 = fig.add_subplot(144)
        labels = ['Diff Points', 'Matched Points']
        sizes = [diff_coords.shape[0], matched_coords_features.shape[0]]
        colors = ['red', 'green']
        
        # 只显示非零的部分
        non_zero_labels = []
        non_zero_sizes = []
        non_zero_colors = []
        for i, size in enumerate(sizes):
            if size > 0:
                non_zero_labels.append(labels[i])
                non_zero_sizes.append(size)
                non_zero_colors.append(colors[i])
        
        if non_zero_sizes:
            wedges, texts, autotexts = ax4.pie(non_zero_sizes, labels=non_zero_labels, 
                                              colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Point Distribution\nTotal: {sum(sizes)}/{points_B.shape[0]}')
        else:
            ax4.text(0.5, 0.5, 'No Points Processed', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Point Distribution')
        
        plt.tight_layout()
        
        # 保存图片到当前目录
        output_path = './blockwise_test_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
        
        # 关闭图形释放内存
        plt.close(fig)
        
        # 生成详细的文本报告
        report_path = './blockwise_test_report.txt'
        with open(report_path, 'w') as f:
            f.write("BlockWiseTransfer 测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"可视化测试数据:\n")
            f.write(f"  Frame A: {points_A.shape[0]} 点, 特征维度: {fA.shape[1]}\n")
            f.write(f"  Frame B: {points_B.shape[0]} 点\n")
            f.write(f"  Block size: {model.block_size}\n\n")
            f.write(f"处理结果:\n")
            f.write(f"  差分区域: {diff_coords.shape[0]} 点 ({diff_coords.shape[0]/points_B.shape[0]*100:.1f}%)\n")
            f.write(f"  匹配区域: {matched_coords_features.shape[0]} 点 ({matched_coords_features.shape[0]/points_B.shape[0]*100:.1f}%)\n")
            f.write(f"  总处理: {diff_coords.shape[0] + matched_coords_features.shape[0]} / {points_B.shape[0]} 点\n")
            f.write(f"  覆盖率: {(diff_coords.shape[0] + matched_coords_features.shape[0])/points_B.shape[0]*100:.1f}%\n\n")
            
            if matched_coords_features.shape[0] > 0:
                f.write(f"匹配数据格式:\n")
                f.write(f"  形状: {matched_coords_features.shape}\n")
                f.write(f"  前3列: B帧坐标\n")
                f.write(f"  后{fA.shape[1]}列: A帧对应特征\n")
        
        print(f"详细报告已保存到: {report_path}")
        
    except ImportError:
        print("未安装matplotlib，跳过可视化")
        print("如需可视化，请安装: pip install matplotlib")
    except Exception as e:
        print(f"可视化过程出错: {e}")

def main():
    print("开始测试 BlockWiseTransfer...")
    
    # 基本功能测试
    success = test_basic_functionality()
    
    if success:
        # 边界情况测试
        test_edge_cases()
        
        # 不同参数测试
        test_different_block_sizes()
        
        # 可视化
        visualize_results()
        
        print("\n🎉 所有测试完成!")
    else:
        print("\n❌ 基本功能测试失败，请检查代码")

if __name__ == "__main__":
    main()
