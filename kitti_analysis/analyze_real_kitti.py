#!/usr/bin/env python3
"""
分析真实KITTI数据的坐标分布，包含点云配准前后对比
专门针对用户的KITTI数据路径：/workspace/data/kitti/sequences/00/velodyne
"""
import os
import sys
import glob
import struct
import math

# 添加项目路径
sys.path.append('/workspace/PointNeXt')

# 尝试导入Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
    print("✅ Open3D已加载")
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D未安装，将跳过配准功能")

# 尝试导入matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
    print("✅ matplotlib已加载 (非GUI模式)")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib未安装，将跳过图形可视化")

def load_kitti_bin_simple(file_path):
    """
    简单版本的KITTI .bin文件加载器，不依赖numpy
    KITTI格式: [x, y, z, intensity] (N, 4)
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 每个点4个float32值 = 16字节
        num_points = len(data) // 16
        points = []
        
        for i in range(num_points):
            offset = i * 16
            # 使用struct来解析二进制数据
            import struct
            x, y, z, intensity = struct.unpack('<ffff', data[offset:offset+16])
            points.append((x, y, z, intensity))
        
        return points
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []

def simple_stats(data):
    """计算简单统计信息"""
    if not data:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'count': 0}
    
    import math
    n = len(data)
    min_val = min(data)
    max_val = max(data)
    mean_val = sum(data) / n
    
    # 计算标准差
    variance = sum((x - mean_val) ** 2 for x in data) / n
    std_val = math.sqrt(variance)
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'range': max_val - min_val,
        'count': n
    }

def analyze_kitti_frame(file_path):
    """分析单个KITTI帧的坐标分布"""
    print(f"分析文件: {file_path}")
    
    # 加载点云数据
    points = load_kitti_bin_simple(file_path)
    
    if not points:
        print("❌ 无法加载数据")
        return None
    
    print(f"✅ 成功加载 {len(points)} 个点")
    
    # 提取坐标
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    intensity_values = [p[3] for p in points]
    
    # 计算统计信息
    stats = {
        'x': simple_stats(x_coords),
        'y': simple_stats(y_coords),
        'z': simple_stats(z_coords),
        'intensity': simple_stats(intensity_values)
    }
    
    # 打印结果
    print("\n=== 坐标分布统计 ===")
    for axis in ['x', 'y', 'z']:
        s = stats[axis]
        print(f"{axis.upper()}轴: 范围[{s['min']:.2f}, {s['max']:.2f}] ({s['range']:.2f}m), "
              f"均值:{s['mean']:.2f}, 标准差:{s['std']:.2f}")
    
    # 强度信息
    s = stats['intensity']
    print(f"强度: 范围[{s['min']:.2f}, {s['max']:.2f}], "
          f"均值:{s['mean']:.2f}, 标准差:{s['std']:.2f}")
    
    return stats

def compare_kitti_frames(file1, file2):
    """比较两个KITTI帧的坐标分布"""
    print(f"\n{'='*60}")
    print("=== 比较两个KITTI帧 ===")
    
    print(f"\n--- Frame A ---")
    stats_A = analyze_kitti_frame(file1)
    
    print(f"\n--- Frame B ---")
    stats_B = analyze_kitti_frame(file2)
    
    if not stats_A or not stats_B:
        print("❌ 无法比较，数据加载失败")
        return
    
    # 计算偏移
    print(f"\n--- 帧间偏移分析 ---")
    for axis in ['x', 'y', 'z']:
        mean_diff = stats_B[axis]['mean'] - stats_A[axis]['mean']
        std_diff = stats_B[axis]['std'] - stats_A[axis]['std']
        range_diff = stats_B[axis]['range'] - stats_A[axis]['range']
        
        print(f"{axis.upper()}轴偏移: 均值差={mean_diff:.2f}m, "
              f"标准差差={std_diff:.2f}m, 范围差={range_diff:.2f}m")
    
    # 强度偏移
    intensity_mean_diff = stats_B['intensity']['mean'] - stats_A['intensity']['mean']
    print(f"强度偏移: 均值差={intensity_mean_diff:.2f}")
    
    return stats_A, stats_B

def analyze_kitti_sequence(data_dir, sequence_id="00", num_frames=5):
    """分析KITTI序列中多个帧的坐标分布"""
    print(f"=== 分析KITTI序列 {sequence_id} ===")
    
    velodyne_dir = os.path.join(data_dir, "sequences", sequence_id, "velodyne")
    
    if not os.path.exists(velodyne_dir):
        print(f"❌ 目录不存在: {velodyne_dir}")
        return
    
    # 获取所有.bin文件
    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    
    if not bin_files:
        print(f"❌ 在目录中未找到.bin文件: {velodyne_dir}")
        return
    
    print(f"✅ 找到 {len(bin_files)} 个.bin文件")
    
    # 分析前几帧
    frames_to_analyze = min(num_frames, len(bin_files))
    all_stats = []
    
    for i in range(frames_to_analyze):
        print(f"\n{'='*40}")
        print(f"=== 帧 {i:06d} ===")
        
        stats = analyze_kitti_frame(bin_files[i])
        if stats:
            all_stats.append({
                'frame_id': i,
                'file_path': bin_files[i],
                'stats': stats
            })
    
    # 序列统计摘要
    if all_stats:
        print(f"\n{'='*60}")
        print("=== 序列统计摘要 ===")
        
        # 计算每个轴的整体统计
        for axis in ['x', 'y', 'z']:
            means = [frame['stats'][axis]['mean'] for frame in all_stats]
            ranges = [frame['stats'][axis]['range'] for frame in all_stats]
            
            mean_of_means = sum(means) / len(means)
            mean_of_ranges = sum(ranges) / len(ranges)
            
            print(f"{axis.upper()}轴 - 平均中心: {mean_of_means:.2f}m, 平均范围: {mean_of_ranges:.2f}m")
        
        # 分析帧间变化
        if len(all_stats) > 1:
            print(f"\n--- 帧间变化分析 ---")
            for i in range(1, len(all_stats)):
                prev_stats = all_stats[i-1]['stats']
                curr_stats = all_stats[i]['stats']
                
                print(f"帧 {i-1:06d} -> {i:06d}:")
                for axis in ['x', 'y', 'z']:
                    mean_shift = curr_stats[axis]['mean'] - prev_stats[axis]['mean']
                    print(f"  {axis.upper()}轴均值位移: {mean_shift:.2f}m")
    
    return all_stats

def load_kitti_to_open3d(file_path, max_points=None):
    """加载KITTI数据并转换为Open3D点云格式"""
    if not HAS_OPEN3D:
        print("❌ Open3D不可用")
        return None
    
    # 加载原始数据
    points = load_kitti_bin_simple(file_path)
    if not points:
        return None
    
    # 下采样（如果指定）
    if max_points and len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]
    
    # 提取坐标
    coords = [[p[0], p[1], p[2]] for p in points]
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    return pcd

def register_point_clouds_open3d(source_pcd, target_pcd, method='icp'):
    """使用Open3D进行点云配准"""
    if not HAS_OPEN3D:
        print("❌ Open3D不可用")
        return None, None, {}
    
    print(f"使用方法: {method}")
    
    # 估计法向量
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    if method == 'icp':
        # ICP配准
        threshold = 2.0
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    elif method == 'feature':
        # 基于特征的配准
        voxel_size = 1.0
        
        # 下采样
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        # 计算FPFH特征
        radius_normal = voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        # RANSAC配准
        distance_threshold = 1.5
        reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        # 精细ICP
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold, reg_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # 应用变换
    source_registered = source_pcd.transform(reg_result.transformation)
    
    # 配准信息
    reg_info = {
        'method': method,
        'fitness': reg_result.fitness,
        'inlier_rmse': reg_result.inlier_rmse,
        'transformation': reg_result.transformation,
        'correspondence_set': len(reg_result.correspondence_set)
    }
    
    return source_registered, reg_result.transformation, reg_info

def visualize_registration_result(source_original, source_registered, target, reg_info):
    """打印配准结果统计信息（不使用GUI可视化）"""
    print(f"\n=== 配准结果统计 ===")
    print(f"配准方法: {reg_info['method']}")
    print(f"适应度: {reg_info['fitness']:.4f}")
    print(f"RMSE: {reg_info['inlier_rmse']:.4f}")
    print(f"对应点数: {reg_info['correspondence_set']}")
    
    # 计算点云统计信息
    def compute_pcd_stats(pcd, name):
        points = pcd.points
        if len(points) > 0:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            z_range = max(z_coords) - min(z_coords)
            
            x_center = sum(x_coords) / len(x_coords)
            y_center = sum(y_coords) / len(y_coords)
            z_center = sum(z_coords) / len(z_coords)
            
            print(f"{name}:")
            print(f"  点数: {len(points)}")
            print(f"  中心: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f})")
            print(f"  范围: X={x_range:.2f}m, Y={y_range:.2f}m, Z={z_range:.2f}m")
    
    if HAS_OPEN3D:
        compute_pcd_stats(source_original, "原始Frame A")
        compute_pcd_stats(source_registered, "配准后Frame A")
        compute_pcd_stats(target, "Frame B (目标)")
    else:
        print("⚠️  Open3D不可用，跳过详细统计")
    
    print("💡 配准可视化已生成matplotlib图表，请查看保存的图片文件")

def create_matplotlib_comparison(points_A_original, points_A_registered, points_B, reg_info):
    """使用matplotlib创建配准前后对比图"""
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib不可用，跳过图形对比")
        return
    
    # 提取坐标
    x_orig = [p[0] for p in points_A_original]
    y_orig = [p[1] for p in points_A_original]
    z_orig = [p[2] for p in points_A_original]
    
    x_reg = [p[0] for p in points_A_registered]
    y_reg = [p[1] for p in points_A_registered]
    z_reg = [p[2] for p in points_A_registered]
    
    x_B = [p[0] for p in points_B]
    y_B = [p[1] for p in points_B]
    z_B = [p[2] for p in points_B]
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 配准前3D视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(x_orig, y_orig, z_orig, c='red', s=1, alpha=0.6, label='Frame A (原始)')
    ax1.scatter(x_B, y_B, z_B, c='blue', s=1, alpha=0.6, label='Frame B (目标)')
    ax1.set_title('配准前 - 3D视图')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    
    # 2. 配准后3D视图
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(x_reg, y_reg, z_reg, c='green', s=1, alpha=0.6, label='Frame A (配准后)')
    ax2.scatter(x_B, y_B, z_B, c='blue', s=1, alpha=0.6, label='Frame B (目标)')
    ax2.set_title('配准后 - 3D视图')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()
    
    # 3. 配准前俯视图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(x_orig, y_orig, c='red', s=1, alpha=0.6, label='Frame A (原始)')
    ax3.scatter(x_B, y_B, c='blue', s=1, alpha=0.6, label='Frame B (目标)')
    ax3.set_title('配准前 - 俯视图')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. 配准后俯视图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(x_reg, y_reg, c='green', s=1, alpha=0.6, label='Frame A (配准后)')
    ax4.scatter(x_B, y_B, c='blue', s=1, alpha=0.6, label='Frame B (目标)')
    ax4.set_title('配准后 - 俯视图')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. 变换矩阵可视化
    ax5 = fig.add_subplot(2, 3, 5)
    if 'transformation' in reg_info:
        T = reg_info['transformation']
        im = ax5.imshow(T, cmap='coolwarm', aspect='equal')
        ax5.set_title('变换矩阵')
        for i in range(4):
            for j in range(4):
                ax5.text(j, i, f'{T[i,j]:.3f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax5)
    
    # 6. 配准统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""配准统计信息:
    
方法: {reg_info.get('method', 'N/A')}
适应度: {reg_info.get('fitness', 0):.4f}
RMSE: {reg_info.get('inlier_rmse', 0):.4f}
对应点数: {reg_info.get('correspondence_set', 0)}

Frame A 原始点数: {len(points_A_original)}
Frame A 配准后点数: {len(points_A_registered)}
Frame B 点数: {len(points_B)}
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/home/mayz/network/PointNeXt/kitti_analysis/kitti_registration_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 配准对比图已保存到: {output_path}")
    
    # 不尝试显示GUI，直接关闭
    plt.close()
    print("💡 图表已保存为PNG文件，请查看文件进行可视化分析")

def analyze_registration_differences(points_A_original, points_A_registered, points_B):
    """分析配准前后的数据差异"""
    print(f"\n=== 配准效果数值分析 ===")
    
    # 计算质心
    def compute_centroid(points):
        x_mean = sum(p[0] for p in points) / len(points)
        y_mean = sum(p[1] for p in points) / len(points)
        z_mean = sum(p[2] for p in points) / len(points)
        return (x_mean, y_mean, z_mean)
    
    centroid_A_orig = compute_centroid(points_A_original)
    centroid_A_reg = compute_centroid(points_A_registered)
    centroid_B = compute_centroid(points_B)
    
    print(f"质心坐标:")
    print(f"  Frame A (原始): ({centroid_A_orig[0]:.2f}, {centroid_A_orig[1]:.2f}, {centroid_A_orig[2]:.2f})")
    print(f"  Frame A (配准后): ({centroid_A_reg[0]:.2f}, {centroid_A_reg[1]:.2f}, {centroid_A_reg[2]:.2f})")
    print(f"  Frame B (目标): ({centroid_B[0]:.2f}, {centroid_B[1]:.2f}, {centroid_B[2]:.2f})")
    
    # 计算距离
    def distance_3d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    
    dist_before = distance_3d(centroid_A_orig, centroid_B)
    dist_after = distance_3d(centroid_A_reg, centroid_B)
    improvement = dist_before - dist_after
    
    print(f"\n质心距离:")
    print(f"  配准前: {dist_before:.2f}m")
    print(f"  配准后: {dist_after:.2f}m")
    print(f"  改善: {improvement:.2f}m ({improvement/dist_before*100:.1f}%)")
    
    # 计算配准引起的变换
    transform_distance = distance_3d(centroid_A_orig, centroid_A_reg)
    print(f"  Frame A变换距离: {transform_distance:.2f}m")
    
    return {
        'centroid_A_original': centroid_A_orig,
        'centroid_A_registered': centroid_A_reg,
        'centroid_B': centroid_B,
        'distance_before': dist_before,
        'distance_after': dist_after,
        'improvement': improvement,
        'transform_distance': transform_distance
    }

def main():
    """主函数"""
    print("=== KITTI真实数据坐标分布分析 + 点云配准对比 ===")
    
    # 用户提供的数据路径
    kitti_data_dir = "/workspace/data/kitti"
    
    # 验证路径存在
    if not os.path.exists(kitti_data_dir):
        print(f"❌ KITTI数据目录不存在: {kitti_data_dir}")
        return
    
    sequence_dir = os.path.join(kitti_data_dir, "sequences", "00")
    if not os.path.exists(sequence_dir):
        print(f"❌ 序列00目录不存在: {sequence_dir}")
        return
    
    velodyne_dir = os.path.join(sequence_dir, "velodyne")
    if not os.path.exists(velodyne_dir):
        print(f"❌ velodyne目录不存在: {velodyne_dir}")
        return
    
    print(f"✅ 找到KITTI数据: {kitti_data_dir}")
    
    # 获取文件列表
    velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    if len(velodyne_files) < 2:
        print("❌ 需要至少2个文件进行配准对比")
        return
    
    print(f"✅ 找到 {len(velodyne_files)} 个文件")
    
    # === 原始分析 ===
    print(f"\n{'='*60}")
    print("=== 原始数据分析 ===")
    
    # 分析序列
    stats = analyze_kitti_sequence(kitti_data_dir, sequence_id="00", num_frames=5)
    
    # 比较连续帧
    print(f"\n{'='*60}")
    print("=== 连续帧对比分析 ===")
    compare_kitti_frames(velodyne_files[0], velodyne_files[1])
    
    # === 点云配准分析 ===
    if HAS_OPEN3D:
        print(f"\n{'='*60}")
        print("=== 点云配准分析 ===")
        
        # 选择要配准的帧
        frame_A_file = velodyne_files[0]
        frame_B_file = velodyne_files[1]
        
        print(f"配准帧:")
        print(f"  Frame A: {os.path.basename(frame_A_file)}")
        print(f"  Frame B: {os.path.basename(frame_B_file)}")
        
        # 加载点云数据
        print(f"\n--- 加载点云数据 ---")
        pcd_A = load_kitti_to_open3d(frame_A_file, max_points=20000)
        pcd_B = load_kitti_to_open3d(frame_B_file, max_points=20000)
        
        if pcd_A is None or pcd_B is None:
            print("❌ 点云加载失败")
            return
        
        print(f"Frame A: {len(pcd_A.points)} 点")
        print(f"Frame B: {len(pcd_B.points)} 点")
        
        # 保存原始A点云的副本
        pcd_A_original = pcd_A.__copy__()
        
        # 执行配准
        print(f"\n--- 执行ICP配准 ---")
        pcd_A_registered, transformation, reg_info = register_point_clouds_open3d(
            pcd_A, pcd_B, method='icp'
        )
        
        if pcd_A_registered is not None:
            # 转换为原始数据格式进行分析
            points_A_original = load_kitti_bin_simple(frame_A_file)
            points_A_registered = [[p[0], p[1], p[2], 0] for p in pcd_A_registered.points]
            points_B = load_kitti_bin_simple(frame_B_file)
            
            # 数值分析
            analysis_result = analyze_registration_differences(
                points_A_original, points_A_registered, points_B
            )
            
            # 创建matplotlib对比图
            create_matplotlib_comparison(
                points_A_original, points_A_registered, points_B, reg_info
            )
            
            # Open3D配准统计（无GUI可视化）
            print(f"\n--- 配准结果统计 ---")
            visualize_registration_result(
                pcd_A_original, pcd_A_registered, pcd_B, reg_info
            )
            
            # 尝试特征配准对比
            print(f"\n--- 尝试特征配准对比 ---")
            pcd_A_feature = pcd_A_original.__copy__()
            pcd_A_feature_reg, transformation_feat, reg_info_feat = register_point_clouds_open3d(
                pcd_A_feature, pcd_B, method='feature'
            )
            
            if pcd_A_feature_reg is not None:
                print(f"ICP配准适应度: {reg_info['fitness']:.4f}")
                print(f"特征配准适应度: {reg_info_feat['fitness']:.4f}")
                
                if reg_info_feat['fitness'] > reg_info['fitness']:
                    print("✅ 特征配准效果更好")
                else:
                    print("✅ ICP配准效果更好")
    else:
        print(f"\n⚠️  Open3D不可用，跳过配准分析")
    
    print(f"\n🎉 分析完成！")
    print(f"📊 数据路径: {kitti_data_dir}")
    print(f"📁 序列: 00")
    print(f"💾 总文件数: {len(velodyne_files)}")
    
    if HAS_OPEN3D:
        print(f"🎯 已完成点云配准对比分析")
        if HAS_MATPLOTLIB:
            print(f"📈 配准对比图已生成")

if __name__ == "__main__":
    main()
