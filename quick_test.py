#!/usr/bin/env python3
"""
快速测试 BlockWiseTransfer 的核心功能
"""
import torch
import sys
import os

# 添加项目路径到 sys.path
sys.path.append('/workspace/home/mayz/network/PointNeXt')

from openpoints.models.custom.blockwise import BlockWiseTransfer

def quick_test():
    print("=== BlockWiseTransfer 快速测试 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成测试数据
    NA, NB, C = 100, 80, 32
    xA = torch.randn(NA, 3, device=device) * 2.0
    fA = torch.randn(NA, C, device=device)
    xB = torch.randn(NB, 3, device=device) * 2.0
    
    print(f"输入数据: xA{xA.shape}, fA{fA.shape}, xB{xB.shape}")
    
    # 创建模型并测试
    model = BlockWiseTransfer(block_size=0.5)
    
    try:
        diff_coords, matched_coords_features = model(xA, fA, xB)
        
        total_processed = diff_coords.shape[0] + matched_coords_features.shape[0]
        print(f"输出结果:")
        print(f"  差分区域: {diff_coords.shape[0]} 点")
        print(f"  匹配区域: {matched_coords_features.shape[0]} 点")
        print(f"  总处理点数: {total_processed} / {NB}")
        print(f"  覆盖率: {total_processed/NB*100:.1f}%")
        
        # 基本验证
        assert total_processed <= NB, f"处理点数超过输入点数"
        assert diff_coords.shape[1] == 3, f"差分坐标维度错误"
        if matched_coords_features.shape[0] > 0:
            assert matched_coords_features.shape[1] == 3 + C, f"匹配数据维度错误"
        
        print("✅ 快速测试通过!")
        
        # 显示点数分布是否合理
        if total_processed < NB * 0.8:  # 如果覆盖率低于80%
            print(f"⚠️  警告: 覆盖率较低 ({total_processed/NB*100:.1f}%)，可能需要调整block_size")
        
        # 可视化结果
        visualize_results(xA, fA, xB, diff_coords, matched_coords_features)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_results(xA, fA, xB, diff_coords, matched_coords_features):
    """可视化结果并保存为图片"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        print("\n=== 生成可视化结果 ===")
        
        # 转换到CPU便于可视化
        xA_cpu = xA.cpu().numpy()
        xB_cpu = xB.cpu().numpy()
        diff_coords_cpu = diff_coords.cpu().numpy()
        
        if matched_coords_features.shape[0] > 0:
            matched_coords_cpu = matched_coords_features[:, :3].cpu().numpy()
        else:
            matched_coords_cpu = np.empty((0, 3))
        
        # 创建图形
        fig = plt.figure(figsize=(20, 5))
        
        # 子图1: 原始数据
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.scatter(xA_cpu[:, 0], xA_cpu[:, 1], xA_cpu[:, 2], 
                   c='blue', alpha=0.6, s=20, label=f'Frame A ({xA.shape[0]})')
        ax1.scatter(xB_cpu[:, 0], xB_cpu[:, 1], xB_cpu[:, 2], 
                   c='red', alpha=0.6, s=20, label=f'Frame B ({xB.shape[0]})')
        ax1.set_title('Original Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 子图2: 差分区域
        ax2 = fig.add_subplot(142, projection='3d')
        if diff_coords.shape[0] > 0:
            ax2.scatter(diff_coords_cpu[:, 0], diff_coords_cpu[:, 1], diff_coords_cpu[:, 2], 
                       c='red', alpha=0.8, s=30, label=f'Diff Points ({diff_coords.shape[0]})')
        ax2.set_title('Difference Region')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # 子图3: 匹配区域
        ax3 = fig.add_subplot(143, projection='3d')
        if matched_coords_cpu.shape[0] > 0:
            ax3.scatter(matched_coords_cpu[:, 0], matched_coords_cpu[:, 1], matched_coords_cpu[:, 2], 
                       c='green', alpha=0.8, s=30, label=f'Matched Points ({matched_coords_cpu.shape[0]})')
        ax3.set_title('Matched Region')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        # 子图4: 结果统计
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
            ax4.set_title(f'Point Distribution\nTotal: {sum(sizes)}/{xB.shape[0]}')
        else:
            ax4.text(0.5, 0.5, 'No Points Processed', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Point Distribution')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = '/workspace/home/mayz/network/PointNeXt/blockwise_test_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
        
        # 关闭图形释放内存
        plt.close(fig)
        
        # 生成详细的文本报告
        report_path = '/workspace/home/mayz/network/PointNeXt/blockwise_test_report.txt'
        with open(report_path, 'w') as f:
            f.write("BlockWiseTransfer 测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"输入数据:\n")
            f.write(f"  Frame A: {xA.shape[0]} 点, 特征维度: {fA.shape[1]}\n")
            f.write(f"  Frame B: {xB.shape[0]} 点\n\n")
            f.write(f"处理结果:\n")
            f.write(f"  差分区域: {diff_coords.shape[0]} 点 ({diff_coords.shape[0]/xB.shape[0]*100:.1f}%)\n")
            f.write(f"  匹配区域: {matched_coords_features.shape[0]} 点 ({matched_coords_features.shape[0]/xB.shape[0]*100:.1f}%)\n")
            f.write(f"  总处理: {diff_coords.shape[0] + matched_coords_features.shape[0]} / {xB.shape[0]} 点\n")
            f.write(f"  覆盖率: {(diff_coords.shape[0] + matched_coords_features.shape[0])/xB.shape[0]*100:.1f}%\n\n")
            
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

if __name__ == "__main__":
    quick_test()
