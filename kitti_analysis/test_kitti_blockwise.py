#!/usr/bin/env python3
"""
使用KITTI数据集测试 BlockWiseTransfer 的 forward 函数
"""
import torch
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 尝试导入配准相关库
try:
    import open3d as o3d
    HAS_OPEN3D = True
    print("✅ Open3D已加载")
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D未安装，将跳过高级配准功能")
    print("💡 安装建议: pip install open3d")

try:
    import pcl
    HAS_PCL = True
except ImportError:
    HAS_PCL = False

# 导入scipy用于简单配准
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    HAS_SCIPY = True
    print("✅ SciPy已加载，可使用简单配准")
except ImportError:
    HAS_SCIPY = False
    print("⚠️  SciPy未安装，配准功能受限")
    print("💡 安装建议: pip install scipy")

# 添加项目路径到 sys.path
sys.path.append('/workspace/PointNeXt')

from openpoints.models.custom.blockwise import BlockWiseTransfer

def load_kitti_bin(file_path):
    """
    加载KITTI .bin点云文件
    KITTI格式: [x, y, z, intensity] (N, 4)
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def load_kitti_txt(file_path):
    """
    加载KITTI .txt点云文件 (如果有的话)
    """
    points = np.loadtxt(file_path, dtype=np.float32)
    return points

def preprocess_kitti_data(points, max_points=50000, coord_range=None):
    """
    预处理KITTI点云数据
    
    Args:
        points: (N, 4) numpy array [x, y, z, intensity]
        max_points: 最大点数限制
        coord_range: 坐标范围限制 [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    
    Returns:
        processed_points: (M, 4) torch tensor [x, y, z, intensity]
    """
    # 移除无效点
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    
    # 坐标范围过滤 (去除过远的点)
    if coord_range is None:
        # 全局KITTI坐标范围 - 不限制范围，保留所有有效点
        coord_range = [(-1000, 1000), (-1000, 1000), (-100, 100)]
    
    x_min, x_max = coord_range[0]
    y_min, y_max = coord_range[1]
    z_min, z_max = coord_range[2]
    
    range_mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    points = points[range_mask]
    
    # 随机下采样到指定点数
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    # 转换为torch tensor
    points_tensor = torch.from_numpy(points).float()
    
    return points_tensor

def create_features_from_kitti(points, feature_dim=64):
    """
    从KITTI点云创建特征
    
    Args:
        points: (N, 4) torch tensor [x, y, z, intensity]
        feature_dim: 特征维度
    
    Returns:
        points_with_features: (N, 3+feature_dim) torch tensor
    """
    coords = points[:, :3]  # [x, y, z]
    intensity = points[:, 3:4]  # intensity
    
    # 计算基础几何特征
    # 1. 距离特征
    distance = torch.norm(coords, dim=1, keepdim=True)
    
    # 2. 高度特征
    height = coords[:, 2:3]
    
    # 3. 强度特征
    intensity_norm = (intensity - intensity.mean()) / (intensity.std() + 1e-8)
    
    # 4. 局部密度特征 (简化版)
    # 这里使用随机特征代替复杂的密度计算
    remaining_features = torch.randn(len(points), feature_dim - 3)
    
    # 组合所有特征
    features = torch.cat([
        distance,           # 1维
        height,            # 1维  
        intensity_norm,    # 1维
        remaining_features # feature_dim-3 维
    ], dim=1)
    
    # 合并坐标和特征
    points_with_features = torch.cat([coords, features], dim=1)
    
    return points_with_features

def test_with_kitti_data(data_dir, sequence_id="00", frame_ids=[0, 1], device="cuda", enable_registration=True, registration_method='open3d_icp'):
    """
    使用KITTI数据测试BlockWiseTransfer
    
    Args:
        data_dir: KITTI数据根目录
        sequence_id: 序列ID (如 "00", "01", ...)
        frame_ids: 要测试的帧ID列表 [frame_A, frame_B]
        device: 设备
        enable_registration: 是否启用配准
        registration_method: 配准方法
    """
    print(f"=== 使用KITTI数据测试 (序列:{sequence_id}, 帧:{frame_ids}) ===")
    
    # 构建文件路径
    velodyne_dir = os.path.join(data_dir, "sequences", sequence_id, "velodyne")
    
    frame_A_file = os.path.join(velodyne_dir, f"{frame_ids[0]:06d}.bin")
    frame_B_file = os.path.join(velodyne_dir, f"{frame_ids[1]:06d}.bin")
    
    # 检查文件是否存在
    if not os.path.exists(frame_A_file):
        print(f"❌ 文件不存在: {frame_A_file}")
        return False
    if not os.path.exists(frame_B_file):
        print(f"❌ 文件不存在: {frame_B_file}")
        return False
    
    print(f"加载数据:")
    print(f"  Frame A: {frame_A_file}")
    print(f"  Frame B: {frame_B_file}")
    
    try:
        # 加载原始点云数据
        points_A_raw = load_kitti_bin(frame_A_file)
        points_B_raw = load_kitti_bin(frame_B_file)
        
        print(f"原始数据:")
        print(f"  Frame A: {points_A_raw.shape}")
        print(f"  Frame B: {points_B_raw.shape}")
        
        # 预处理数据 - 使用全局范围
        points_A_processed = preprocess_kitti_data(points_A_raw, max_points=20000)
        points_B_processed = preprocess_kitti_data(points_B_raw, max_points=15000)
        
        print(f"预处理后:")
        print(f"  Frame A: {points_A_processed.shape}")
        print(f"  Frame B: {points_B_processed.shape}")
        
        # 点云配准（如果启用）
        if enable_registration:
            print(f"\n--- 执行点云配准 (方法: {registration_method}) ---")
            coords_A = points_A_processed[:, :3].numpy()
            coords_B = points_B_processed[:, :3].numpy()
            
            transformation, registered_coords_A, reg_info = register_point_clouds(
                coords_A, coords_B, method=registration_method
            )
            
            print(f"配准结果:")
            print(f"  方法: {reg_info['method']}")
            print(f"  适应度: {reg_info.get('fitness', 'N/A'):.4f}")
            print(f"  RMSE: {reg_info.get('inlier_rmse', 'N/A')}")
            if 'correspondence_set' in reg_info:
                print(f"  对应点数: {reg_info['correspondence_set']}")
            
            # 更新Frame A的坐标
            points_A_processed[:, :3] = torch.from_numpy(registered_coords_A).float()
            print(f"  Frame A已配准到Frame B坐标系")
        else:
            print("⚠️  跳过点云配准")
        
        # 创建特征 (Frame A有特征，Frame B只有坐标)
        points_A_with_features = create_features_from_kitti(points_A_processed, feature_dim=64)
        points_B_coords = points_B_processed[:, :3]  # 只取坐标
        
        # 转移到指定设备
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        points_A_with_features = points_A_with_features.to(device)
        points_B_coords = points_B_coords.to(device)
        
        print(f"\n最终输入:")
        print(f"  Frame A (with features): {points_A_with_features.shape}, device: {points_A_with_features.device}")
        print(f"  Frame B (coords only): {points_B_coords.shape}, device: {points_B_coords.device}")
        
        # === 快速坐标分布统计 ===
        coords_A_for_stats = points_A_with_features[:, :3].cpu().numpy()
        coords_B_for_stats = points_B_coords.cpu().numpy()
        
        # 使用新的分析函数
        coord_stats = analyze_coordinate_distribution(coords_A_for_stats, coords_B_for_stats)
        print_coordinate_stats(coord_stats)
        
        # === 开始测试不同block_size ===
        # 测试不同的block_size
        block_sizes = [1.0, 2.0, 5.0, 10.0]
        
        for block_size in block_sizes:
            print(f"\n--- 测试 block_size = {block_size}m ---")
            
            model = BlockWiseTransfer(block_size=block_size)
            
            # 执行前向传播
            diff_coords, matched_coords_features = model(points_A_with_features, points_B_coords)
            
            total_processed = diff_coords.shape[0] + matched_coords_features.shape[0]
            
            print(f"结果:")
            print(f"  差分区域: {diff_coords.shape[0]} 点 ({diff_coords.shape[0]/points_B_coords.shape[0]*100:.1f}%)")
            print(f"  匹配区域: {matched_coords_features.shape[0]} 点 ({matched_coords_features.shape[0]/points_B_coords.shape[0]*100:.1f}%)")
            print(f"  总处理: {total_processed} / {points_B_coords.shape[0]} 点")
            print(f"  覆盖率: {total_processed/points_B_coords.shape[0]*100:.1f}%")
            
            # 验证输出格式
            assert diff_coords.shape[1] == 3, f"差分坐标维度错误: {diff_coords.shape[1]} != 3"
            assert matched_coords_features.shape[1] == 67, f"匹配数据维度错误: {matched_coords_features.shape[1]} != 67"
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_kitti_sequences(data_dir):
    """查找可用的KITTI序列"""
    sequences_dir = os.path.join(data_dir, "sequences")
    if not os.path.exists(sequences_dir):
        return []
    
    sequences = []
    for item in os.listdir(sequences_dir):
        seq_path = os.path.join(sequences_dir, item)
        if os.path.isdir(seq_path):
            velodyne_path = os.path.join(seq_path, "velodyne")
            if os.path.exists(velodyne_path):
                # 检查是否有.bin文件
                bin_files = glob.glob(os.path.join(velodyne_path, "*.bin"))
                if len(bin_files) > 1:  # 至少需要2帧
                    sequences.append(item)
    
    return sorted(sequences)

def visualize_kitti_results(data_dir, sequence_id="00", frame_ids=[0, 1], device="cuda", enable_registration=True, registration_method='open3d_icp'):
    """
    可视化KITTI数据的BlockWiseTransfer结果
    """
    try:
        print(f"\n=== 生成KITTI数据可视化结果 ===")
        
        # 构建文件路径
        velodyne_dir = os.path.join(data_dir, "sequences", sequence_id, "velodyne")
        frame_A_file = os.path.join(velodyne_dir, f"{frame_ids[0]:06d}.bin")
        frame_B_file = os.path.join(velodyne_dir, f"{frame_ids[1]:06d}.bin")
        
        # 检查文件是否存在
        if not os.path.exists(frame_A_file) or not os.path.exists(frame_B_file):
            print(f"❌ 文件不存在，跳过可视化")
            return
        
        # 加载和预处理数据（使用较小的点数便于可视化）
        points_A_raw = load_kitti_bin(frame_A_file)
        points_B_raw = load_kitti_bin(frame_B_file)
        
        # 为了可视化效果，使用全局范围但适当减少点数
        points_A_processed = preprocess_kitti_data(points_A_raw, max_points=8000, 
                                                 coord_range=[(-200, 200), (-200, 200), (-10, 10)])
        points_B_processed = preprocess_kitti_data(points_B_raw, max_points=6000,
                                                 coord_range=[(-200, 200), (-200, 200), (-10, 10)])
        
        # 创建特征
        points_A_with_features = create_features_from_kitti(points_A_processed, feature_dim=64)
        points_B_coords = points_B_processed[:, :3]
        
        # 点云配准（如果启用）
        if enable_registration:
            print(f"执行配准 (方法: {registration_method})")
            coords_A = points_A_processed[:, :3].numpy()
            coords_B = points_B_coords.numpy()
            
            transformation, registered_coords_A, reg_info = register_point_clouds(
                coords_A, coords_B, method=registration_method
            )
            
            print(f"配准适应度: {reg_info.get('fitness', 'N/A'):.4f}")
            
            # 更新Frame A的坐标
            points_A_processed[:, :3] = torch.from_numpy(registered_coords_A).float()
            points_A_with_features = create_features_from_kitti(points_A_processed, feature_dim=64)
        
        # 转移到CPU便于可视化
        device = torch.device('cpu')
        points_A_with_features = points_A_with_features.to(device)
        points_B_coords = points_B_coords.to(device)
        
        # 提取坐标用于可视化
        coords_A = points_A_with_features[:, :3]
        
        # 测试四种不同的block_size
        block_sizes = [1.0, 2.0, 5.0, 10.0]
        results = []
        
        for block_size in block_sizes:
            model = BlockWiseTransfer(block_size=block_size)
            diff_coords, matched_coords_features = model(points_A_with_features, points_B_coords)
            
            # 提取匹配点的坐标
            if matched_coords_features.shape[0] > 0:
                matched_coords = matched_coords_features[:, :3]
            else:
                matched_coords = torch.empty((0, 3))
            
            results.append({
                'block_size': block_size,
                'diff_coords': diff_coords,
                'matched_coords': matched_coords,
                'diff_count': diff_coords.shape[0],
                'matched_count': matched_coords.shape[0]
            })
        
        # 创建大图：2x2子图布局，每个子图显示一种block_size的结果
        fig = plt.figure(figsize=(24, 20))
        
        # 为每种block_size创建子图
        for i, result in enumerate(results):
            # 创建子图 (2x2布局)
            ax_main = fig.add_subplot(2, 2, i+1, projection='3d')
            
            # 转换为numpy便于绘图
            coords_A_np = coords_A.numpy()
            coords_B_np = points_B_coords.numpy()
            diff_coords_np = result['diff_coords'].numpy()
            matched_coords_np = result['matched_coords'].numpy()
            
            # 绘制原始Frame A (蓝色，较小点)
            ax_main.scatter(coords_A_np[:, 0], coords_A_np[:, 1], coords_A_np[:, 2], 
                          c='lightblue', alpha=0.3, s=1, label=f'Frame A ({coords_A.shape[0]})')
            
            # 绘制原始Frame B (浅灰色，较小点)
            ax_main.scatter(coords_B_np[:, 0], coords_B_np[:, 1], coords_B_np[:, 2], 
                          c='lightgray', alpha=0.3, s=1, label=f'Frame B ({points_B_coords.shape[0]})')
            
            # 绘制差分区域 (红色，较大点)
            if result['diff_count'] > 0:
                ax_main.scatter(diff_coords_np[:, 0], diff_coords_np[:, 1], diff_coords_np[:, 2], 
                              c='red', alpha=0.8, s=15, label=f'Diff ({result["diff_count"]})')
            
            # 绘制匹配区域 (绿色，较大点)
            if result['matched_count'] > 0:
                ax_main.scatter(matched_coords_np[:, 0], matched_coords_np[:, 1], matched_coords_np[:, 2], 
                              c='green', alpha=0.8, s=15, label=f'Matched ({result["matched_count"]})')
            
            # 设置标题和标签
            total_processed = result['diff_count'] + result['matched_count']
            coverage = total_processed / points_B_coords.shape[0] * 100
            ax_main.set_title(f'Block Size = {result["block_size"]}m\n'
                            f'Coverage: {coverage:.1f}% ({total_processed}/{points_B_coords.shape[0]})',
                            fontsize=12)
            ax_main.set_xlabel('X (m)', fontsize=10)
            ax_main.set_ylabel('Y (m)', fontsize=10)
            ax_main.set_zlabel('Z (m)', fontsize=10)
            ax_main.legend(fontsize=8)
            
            # 设置动态坐标范围 - 基于数据自适应调整
            # 计算实际数据范围
            all_coords = np.concatenate([coords_A_np, coords_B_np], axis=0)
            x_range = [all_coords[:, 0].min() - 5, all_coords[:, 0].max() + 5]
            y_range = [all_coords[:, 1].min() - 5, all_coords[:, 1].max() + 5]
            z_range = [all_coords[:, 2].min() - 1, all_coords[:, 2].max() + 1]
            
            ax_main.set_xlim(x_range)
            ax_main.set_ylim(y_range)
            ax_main.set_zlim(z_range)
            
            # 调整视角
            ax_main.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = f'/workspace/PointNeXt/kitti_blockwise_seq{sequence_id}_frames{frame_ids[0]}-{frame_ids[1]}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
        
        # 关闭图形释放内存
        plt.close(fig)
        
        # === 新增：点云xyz轴分布统计和可视化 ===
        print(f"\n=== 点云xyz轴分布统计 ===")
        
        # 统计Frame A和Frame B的坐标分布
        coords_A_stats = {
            'x': {'min': coords_A_np[:, 0].min(), 'max': coords_A_np[:, 0].max(), 
                  'mean': coords_A_np[:, 0].mean(), 'std': coords_A_np[:, 0].std()},
            'y': {'min': coords_A_np[:, 1].min(), 'max': coords_A_np[:, 1].max(), 
                  'mean': coords_A_np[:, 1].mean(), 'std': coords_A_np[:, 1].std()},
            'z': {'min': coords_A_np[:, 2].min(), 'max': coords_A_np[:, 2].max(), 
                  'mean': coords_A_np[:, 2].mean(), 'std': coords_A_np[:, 2].std()}
        }
        
        coords_B_stats = {
            'x': {'min': coords_B_np[:, 0].min(), 'max': coords_B_np[:, 0].max(), 
                  'mean': coords_B_np[:, 0].mean(), 'std': coords_B_np[:, 0].std()},
            'y': {'min': coords_B_np[:, 1].min(), 'max': coords_B_np[:, 1].max(), 
                  'mean': coords_B_np[:, 1].mean(), 'std': coords_B_np[:, 1].std()},
            'z': {'min': coords_B_np[:, 2].min(), 'max': coords_B_np[:, 2].max(), 
                  'mean': coords_B_np[:, 2].mean(), 'std': coords_B_np[:, 2].std()}
        }
        
        # 打印统计信息
        print("Frame A 坐标分布:")
        for axis in ['x', 'y', 'z']:
            stats = coords_A_stats[axis]
            print(f"  {axis.upper()}轴: 范围[{stats['min']:.2f}, {stats['max']:.2f}], "
                  f"均值:{stats['mean']:.2f}, 标准差:{stats['std']:.2f}")
        
        print("Frame B 坐标分布:")
        for axis in ['x', 'y', 'z']:
            stats = coords_B_stats[axis]
            print(f"  {axis.upper()}轴: 范围[{stats['min']:.2f}, {stats['max']:.2f}], "
                  f"均值:{stats['mean']:.2f}, 标准差:{stats['std']:.2f}")
        
        # 创建坐标分布可视化图
        fig_dist, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig_dist.suptitle(f'Point Cloud Coordinate Distribution\nSequence {sequence_id}, Frames {frame_ids[0]}-{frame_ids[1]}', 
                         fontsize=16, fontweight='bold')
        
        # 为每个轴创建直方图
        axes_names = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        for i, (axis_name, color) in enumerate(zip(axes_names, colors)):
            # Frame A 分布
            ax_a = axes[i, 0]
            data_a = coords_A_np[:, i]
            ax_a.hist(data_a, bins=50, alpha=0.7, color=f'light{color}', edgecolor=color, linewidth=1)
            ax_a.axvline(data_a.mean(), color=color, linestyle='--', linewidth=2, label=f'Mean: {data_a.mean():.2f}')
            ax_a.axvline(data_a.mean() + data_a.std(), color=color, linestyle=':', alpha=0.7, 
                        label=f'±1σ: {data_a.std():.2f}')
            ax_a.axvline(data_a.mean() - data_a.std(), color=color, linestyle=':', alpha=0.7)
            ax_a.set_title(f'Frame A - {axis_name} Axis Distribution')
            ax_a.set_xlabel(f'{axis_name} Coordinate (m)')
            ax_a.set_ylabel('Point Count')
            ax_a.legend()
            ax_a.grid(True, alpha=0.3)
            
            # Frame B 分布
            ax_b = axes[i, 1]
            data_b = coords_B_np[:, i]
            ax_b.hist(data_b, bins=50, alpha=0.7, color=f'light{color}', edgecolor=color, linewidth=1)
            ax_b.axvline(data_b.mean(), color=color, linestyle='--', linewidth=2, label=f'Mean: {data_b.mean():.2f}')
            ax_b.axvline(data_b.mean() + data_b.std(), color=color, linestyle=':', alpha=0.7, 
                        label=f'±1σ: {data_b.std():.2f}')
            ax_b.axvline(data_b.mean() - data_b.std(), color=color, linestyle=':', alpha=0.7)
            ax_b.set_title(f'Frame B - {axis_name} Axis Distribution')
            ax_b.set_xlabel(f'{axis_name} Coordinate (m)')
            ax_b.set_ylabel('Point Count')
            ax_b.legend()
            ax_b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存坐标分布图
        dist_path = f'/workspace/PointNeXt/kitti_coord_distribution_seq{sequence_id}_frames{frame_ids[0]}-{frame_ids[1]}.png'
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        print(f"坐标分布图已保存到: {dist_path}")
        
        plt.close(fig_dist)
        
        # 创建坐标对比箱线图
        fig_box, axes_box = plt.subplots(1, 3, figsize=(18, 6))
        fig_box.suptitle(f'Coordinate Distribution Comparison (Box Plot)\nSequence {sequence_id}, Frames {frame_ids[0]}-{frame_ids[1]}', 
                        fontsize=14, fontweight='bold')
        
        for i, (axis_name, color) in enumerate(zip(axes_names, colors)):
            ax = axes_box[i]
            data_to_plot = [coords_A_np[:, i], coords_B_np[:, i]]
            labels = [f'Frame A\n({coords_A_np.shape[0]} pts)', f'Frame B\n({coords_B_np.shape[0]} pts)']
            
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor(f'light{color}')
            box_plot['boxes'][1].set_facecolor(f'light{color}')
            box_plot['boxes'][0].set_alpha(0.7)
            box_plot['boxes'][1].set_alpha(0.7)
            
            ax.set_title(f'{axis_name} Axis Distribution')
            ax.set_ylabel(f'{axis_name} Coordinate (m)')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息文本
            stats_text = f'Frame A: μ={coords_A_stats[axis_name.lower()]["mean"]:.2f}, σ={coords_A_stats[axis_name.lower()]["std"]:.2f}\n'
            stats_text += f'Frame B: μ={coords_B_stats[axis_name.lower()]["mean"]:.2f}, σ={coords_B_stats[axis_name.lower()]["std"]:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存箱线图
        box_path = f'/workspace/PointNeXt/kitti_coord_boxplot_seq{sequence_id}_frames{frame_ids[0]}-{frame_ids[1]}.png'
        plt.savefig(box_path, dpi=150, bbox_inches='tight')
        print(f"坐标箱线图已保存到: {box_path}")
        
        # === 原有的统计对比图部分 ===
        # 创建统计对比图
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取统计数据
        block_sizes_list = [r['block_size'] for r in results]
        diff_counts = [r['diff_count'] for r in results]
        matched_counts = [r['matched_count'] for r in results]
        total_counts = [d + m for d, m in zip(diff_counts, matched_counts)]
        coverage_rates = [t / points_B_coords.shape[0] * 100 for t in total_counts]
        
        # 图1: 各block_size的点数分布
        x = np.arange(len(block_sizes_list))
        width = 0.35
        
        ax1.bar(x - width/2, diff_counts, width, label='Diff Points', color='red', alpha=0.7)
        ax1.bar(x + width/2, matched_counts, width, label='Matched Points', color='green', alpha=0.7)
        ax1.set_xlabel('Block Size (m)')
        ax1.set_ylabel('Point Count')
        ax1.set_title('Point Distribution by Block Size')
        ax1.set_xticks(x)
        ax1.set_xticklabels(block_sizes_list)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 覆盖率对比
        ax2.plot(block_sizes_list, coverage_rates, 'b-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Block Size (m)')
        ax2.set_ylabel('Coverage Rate (%)')
        ax2.set_title('Coverage Rate vs Block Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # 图3: 饼图显示最优block_size的分布
        best_idx = np.argmax(coverage_rates)
        best_result = results[best_idx]
        labels = ['Diff Points', 'Matched Points']
        sizes = [best_result['diff_count'], best_result['matched_count']]
        colors = ['red', 'green']
        
        # 只显示非零的部分
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        if non_zero_data:
            non_zero_labels, non_zero_sizes, non_zero_colors = zip(*non_zero_data)
            wedges, texts, autotexts = ax3.pie(non_zero_sizes, labels=non_zero_labels, 
                                              colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Best Block Size: {best_result["block_size"]}m\n'
                         f'Coverage: {coverage_rates[best_idx]:.1f}%')
        
        # 图4: 处理效率对比
        efficiency = [m / (d + m) * 100 if (d + m) > 0 else 0 for d, m in zip(diff_counts, matched_counts)]
        ax4.bar(block_sizes_list, efficiency, color='orange', alpha=0.7)
        ax4.set_xlabel('Block Size (m)')
        ax4.set_ylabel('Matching Efficiency (%)')
        ax4.set_title('Matching Efficiency by Block Size')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])
        
        plt.tight_layout()
        
        # 保存统计图
        stats_path = f'/workspace/PointNeXt/kitti_blockwise_stats_seq{sequence_id}_frames{frame_ids[0]}-{frame_ids[1]}.png'
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        print(f"统计图已保存到: {stats_path}")
        
        plt.close(fig2)
        
        # 生成详细的文本报告
        report_path = f'/workspace/PointNeXt/kitti_blockwise_report_seq{sequence_id}_frames{frame_ids[0]}-{frame_ids[1]}.txt'
        with open(report_path, 'w') as f:
            f.write("KITTI BlockWiseTransfer 测试报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据集信息:\n")
            f.write(f"  序列ID: {sequence_id}\n")
            f.write(f"  测试帧: {frame_ids[0]} -> {frame_ids[1]}\n")
            f.write(f"  Frame A: {coords_A.shape[0]} 点, 特征维度: 64\n")
            f.write(f"  Frame B: {points_B_coords.shape[0]} 点\n\n")
            
            # 添加xyz轴分布统计
            f.write("点云坐标分布统计:\n")
            f.write("-" * 60 + "\n")
            f.write("Frame A 坐标分布:\n")
            for axis in ['x', 'y', 'z']:
                stats = coords_A_stats[axis]
                f.write(f"  {axis.upper()}轴: 范围[{stats['min']:.2f}, {stats['max']:.2f}], "
                       f"均值:{stats['mean']:.2f}, 标准差:{stats['std']:.2f}\n")
            
            f.write("\nFrame B 坐标分布:\n")
            for axis in ['x', 'y', 'z']:
                stats = coords_B_stats[axis]
                f.write(f"  {axis.upper()}轴: 范围[{stats['min']:.2f}, {stats['max']:.2f}], "
                       f"均值:{stats['mean']:.2f}, 标准差:{stats['std']:.2f}\n")
            
            # 添加坐标偏移分析
            f.write("\n坐标偏移分析:\n")
            for axis in ['x', 'y', 'z']:
                offset_mean = coords_B_stats[axis]['mean'] - coords_A_stats[axis]['mean']
                offset_std = abs(coords_B_stats[axis]['std'] - coords_A_stats[axis]['std'])
                f.write(f"  {axis.upper()}轴偏移: 均值差={offset_mean:.2f}m, 标准差差={offset_std:.2f}m\n")
            f.write("\n")
            
            f.write("各Block Size测试结果:\n")
            f.write("-" * 60 + "\n")
            for i, result in enumerate(results):
                total = result['diff_count'] + result['matched_count']
                coverage = total / points_B_coords.shape[0] * 100
                efficiency = result['matched_count'] / total * 100 if total > 0 else 0
                
                f.write(f"Block Size {result['block_size']}m:\n")
                f.write(f"  差分区域: {result['diff_count']} 点 ({result['diff_count']/points_B_coords.shape[0]*100:.1f}%)\n")
                f.write(f"  匹配区域: {result['matched_count']} 点 ({result['matched_count']/points_B_coords.shape[0]*100:.1f}%)\n")
                f.write(f"  总处理: {total} / {points_B_coords.shape[0]} 点\n")
                f.write(f"  覆盖率: {coverage:.1f}%\n")
                f.write(f"  匹配效率: {efficiency:.1f}%\n\n")
            
            # 推荐最佳参数
            best_coverage_idx = np.argmax(coverage_rates)
            best_efficiency_idx = np.argmax(efficiency)
            
            f.write("推荐参数:\n")
            f.write(f"  最佳覆盖率: Block Size {results[best_coverage_idx]['block_size']}m (覆盖率: {coverage_rates[best_coverage_idx]:.1f}%)\n")
            f.write(f"  最佳匹配效率: Block Size {results[best_efficiency_idx]['block_size']}m (匹配效率: {efficiency[best_efficiency_idx]:.1f}%)\n")
        
        print(f"详细报告已保存到: {report_path}")
        
    except ImportError:
        print("未安装matplotlib，跳过可视化")
        print("如需可视化，请安装: pip install matplotlib")
    except Exception as e:
        print(f"可视化过程出错: {e}")
        import traceback
        traceback.print_exc()

def register_point_clouds(source_points, target_points, method='open3d_icp', **kwargs):
    """
    点云配准统一接口
    
    Args:
        source_points: (N, 3) numpy array - 源点云坐标
        target_points: (M, 3) numpy array - 目标点云坐标
        method: 配准方法 ['open3d_icp', 'open3d_feature', 'simple_icp', 'none']
        **kwargs: 其他参数
    
    Returns:
        transformation_matrix: (4, 4) 变换矩阵
        registered_points: 配准后的源点云
        registration_info: 配准信息字典
    """
    if method == 'none':
        # 不进行配准，返回单位矩阵
        identity = np.eye(4)
        return identity, source_points.copy(), {'method': 'none', 'fitness': 1.0}
    
    elif method == 'open3d_icp' and HAS_OPEN3D:
        return register_with_open3d_icp(source_points, target_points, **kwargs)
    
    elif method == 'open3d_feature' and HAS_OPEN3D:
        return register_with_open3d_feature(source_points, target_points, **kwargs)
    
    elif method == 'simple_icp':
        return register_with_simple_icp(source_points, target_points, **kwargs)
    
    else:
        print(f"⚠️  配准方法 '{method}' 不可用，使用简单ICP")
        return register_with_simple_icp(source_points, target_points, **kwargs)

def register_with_open3d_icp(source_points, target_points, threshold=2.0, max_iteration=50):
    """使用Open3D ICP配准"""
    # 创建点云对象
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # 估计法向量
    source.estimate_normals()
    target.estimate_normals()
    
    # ICP配准
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    # 应用变换
    source.transform(reg_result.transformation)
    registered_points = np.asarray(source.points)
    
    info = {
        'method': 'open3d_icp',
        'fitness': reg_result.fitness,
        'inlier_rmse': reg_result.inlier_rmse,
        'correspondence_set': len(reg_result.correspondence_set)
    }
    
    return reg_result.transformation, registered_points, info

def register_with_open3d_feature(source_points, target_points, 
                                 voxel_size=1.0, distance_threshold=1.5):
    """使用Open3D基于特征的配准"""
    # 创建点云对象
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # 下采样
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # 计算法向量
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 计算FPFH特征
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC配准
    reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    # 精细配准
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, reg_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # 应用变换
    source.transform(reg_result.transformation)
    registered_points = np.asarray(source.points)
    
    info = {
        'method': 'open3d_feature',
        'fitness': reg_result.fitness,
        'inlier_rmse': reg_result.inlier_rmse,
        'correspondence_set': len(reg_result.correspondence_set)
    }
    
    return reg_result.transformation, registered_points, info

def register_with_simple_icp(source_points, target_points, max_iterations=20, tolerance=1e-6):
    """简单的ICP实现（备用方案）"""
    if not HAS_SCIPY:
        print("⚠️  SciPy未安装，使用最简单的配准方案")
        # 使用质心对齐作为最简单的配准
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        translation = target_center - source_center
        
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        
        # 应用变换
        registered_points = source_points + translation
        
        info = {
            'method': 'centroid_alignment',
            'fitness': 0.5,  # 假设适应度
            'inlier_rmse': np.linalg.norm(translation),
            'success': True
        }
        
        return transformation, registered_points, info
    
    def transformation_matrix_from_params(params):
        """从6DOF参数构建变换矩阵"""
        tx, ty, tz, rx, ry, rz = params
        
        # 旋转矩阵（欧拉角）
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        
        R = np.array([
            [cy*cz, -cy*sz, sy],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T
    
    def objective_function(params, source, target):
        """目标函数：最小化点到点距离"""
        T = transformation_matrix_from_params(params)
        source_transformed = (T[:3, :3] @ source.T + T[:3, 3:4]).T
        
        # 计算最近邻距离
        distances = cdist(source_transformed, target)
        min_distances = np.min(distances, axis=1)
        return np.mean(min_distances)
    
    # 初始参数估计
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    initial_translation = target_center - source_center
    initial_params = np.concatenate([initial_translation, [0, 0, 0]])
    
    # 优化
    result = minimize(objective_function, initial_params, 
                     args=(source_points, target_points),
                     method='L-BFGS-B')
    
    # 构建最终变换矩阵
    final_transformation = transformation_matrix_from_params(result.x)
    
    # 应用变换
    source_homo = np.hstack([source_points, np.ones((len(source_points), 1))])
    registered_points = (final_transformation @ source_homo.T).T[:, :3]
    
    info = {
        'method': 'simple_icp',
        'fitness': 1.0 / (1.0 + result.fun),  # 近似适应度
        'inlier_rmse': result.fun,
        'success': result.success
    }
    
    return final_transformation, registered_points, info

def get_best_registration_method():
    """
    根据可用库自动选择最佳配准方法
    """
    if HAS_OPEN3D:
        print("🎯 使用Open3D ICP配准（高精度）")
        return 'open3d_icp'
    elif HAS_SCIPY:
        print("🎯 使用简单ICP配准（中等精度）")
        return 'simple_icp'
    else:
        print("🎯 使用质心对齐配准（基础精度）")
        return 'simple_icp'  # 质心对齐在simple_icp中处理

def main():
    print("开始KITTI数据测试...")
    
    # KITTI数据路径配置
    # 请根据您的实际路径修改
    possible_data_dirs = [
        "/workspace/network/data/kitti",  # 用户提供的路径
        "/workspace/data/kitti",
        "/workspace/datasets/kitti", 
        "/data/kitti",
        "/datasets/kitti",
        "./kitti_data",
        "../kitti_data"
    ]
    
    data_dir = None
    for path in possible_data_dirs:
        if os.path.exists(path):
            data_dir = path
            break
    
    if data_dir is None:
        print("❌ 未找到KITTI数据集，请检查以下路径是否存在:")
        for path in possible_data_dirs:
            print(f"  {path}")
        print("\n请确保KITTI数据集结构如下:")
        print("kitti_data/")
        print("├── sequences/")
        print("│   ├── 00/")
        print("│   │   └── velodyne/")
        print("│   │       ├── 000000.bin")
        print("│   │       ├── 000001.bin")
        print("│   │       └── ...")
        print("│   ├── 01/")
        print("│   └── ...")
        return
    
    print(f"找到KITTI数据集: {data_dir}")
    
    # 查找可用序列
    sequences = find_kitti_sequences(data_dir)
    if not sequences:
        print(f"❌ 在 {data_dir} 中未找到有效的KITTI序列")
        return
    
    print(f"可用序列: {sequences}")
    
    # 自动选择最佳配准方法
    best_registration_method = get_best_registration_method()
    
    # 测试前几个序列
    test_sequences = sequences[:2]  # 测试前2个序列
    
    for seq_id in test_sequences:
        print(f"\n{'='*50}")
        
        # 测试连续帧
        frame_pairs = [(0, 1), (10, 11), (50, 51)]
        
        for frame_A, frame_B in frame_pairs:
            success = test_with_kitti_data(
                data_dir=data_dir,
                sequence_id=seq_id,
                frame_ids=[frame_A, frame_B],
                device="cuda",
                enable_registration=True,  # 启用配准
                registration_method=best_registration_method  # 自动选择最佳方法
            )
            
            if not success:
                print(f"跳过序列 {seq_id} 的帧 {frame_A}-{frame_B}")
                continue
            
            # 生成可视化
            visualize_kitti_results(
                data_dir=data_dir,
                sequence_id=seq_id,
                frame_ids=[frame_A, frame_B],
                device="cuda",
                enable_registration=True,
                registration_method=best_registration_method  # 使用相同的配准方法
            )
            
            break  # 成功测试一对帧后跳出
    
    print(f"\n🎉 KITTI数据测试完成!")

if __name__ == "__main__":
    main()

def analyze_coordinate_distribution(coords_A, coords_B, frame_names=['Frame A', 'Frame B']):
    """
    分析两个点云的坐标分布
    
    Args:
        coords_A: (N, 3) numpy array - 第一个点云的坐标
        coords_B: (M, 3) numpy array - 第二个点云的坐标
        frame_names: 两个点云的名称
    
    Returns:
        stats_dict: 包含统计信息的字典
    """
    stats = {}
    
    for i, (coords, name) in enumerate(zip([coords_A, coords_B], frame_names)):
        stats[name] = {}
        for j, axis in enumerate(['x', 'y', 'z']):
            stats[name][axis] = {
                'min': float(coords[:, j].min()),
                'max': float(coords[:, j].max()),
                'mean': float(coords[:, j].mean()),
                'std': float(coords[:, j].std()),
                'range': float(coords[:, j].max() - coords[:, j].min())
            }
    
    # 计算坐标偏移
    stats['offset'] = {}
    for axis in ['x', 'y', 'z']:
        stats['offset'][axis] = {
            'mean_diff': stats[frame_names[1]][axis]['mean'] - stats[frame_names[0]][axis]['mean'],
            'std_diff': stats[frame_names[1]][axis]['std'] - stats[frame_names[0]][axis]['std'],
            'range_diff': stats[frame_names[1]][axis]['range'] - stats[frame_names[0]][axis]['range']
        }
    
    return stats

def print_coordinate_stats(stats, frame_names=['Frame A', 'Frame B']):
    """
    打印坐标统计信息
    
    Args:
        stats: analyze_coordinate_distribution返回的统计字典
        frame_names: 帧名称列表
    """
    print("=== 坐标分布统计 ===")
    
    for name in frame_names:
        print(f"\n{name} 坐标分布:")
        for axis in ['x', 'y', 'z']:
            s = stats[name][axis]
            print(f"  {axis.upper()}轴: 范围[{s['min']:.2f}, {s['max']:.2f}] ({s['range']:.2f}m), "
                  f"均值:{s['mean']:.2f}, 标准差:{s['std']:.2f}")
    
    print("\n坐标偏移分析:")
    for axis in ['x', 'y', 'z']:
        offset = stats['offset'][axis]
        print(f"  {axis.upper()}轴偏移: 均值差={offset['mean_diff']:.2f}m, "
              f"标准差差={offset['std_diff']:.2f}m, 范围差={offset['range_diff']:.2f}m")
