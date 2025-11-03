#!/usr/bin/env python3
"""
Docker环境专用的KITTI点云配准分析脚本
完全避免GUI，只生成图片和文本输出
"""
import os
import sys
import glob
import struct
import math
import time

# 添加项目路径
sys.path.append('/workspace/PointNeXt')

# 强制matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 必须在import pyplot之前设置

# 尝试导入Open3D（用于配准计算，不用于可视化）
try:
    import open3d as o3d
    HAS_OPEN3D = True
    print("✅ Open3D已加载（仅用于计算，不使用GUI）")
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D未安装，将跳过配准功能")

# 导入matplotlib（非GUI模式）
try:
    import matplotlib
    matplotlib.use('Agg')  # 必须在import pyplot之前设置
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 设置字体以避免中文字符问题
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    HAS_MATPLOTLIB = True
    print("✅ matplotlib已加载（非GUI模式，英文字体）")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib未安装，将跳过图形可视化")

def load_kitti_bin_simple(file_path):
    """简单版本的KITTI .bin文件加载器"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        num_points = len(data) // 16
        points = []
        
        for i in range(num_points):
            offset = i * 16
            x, y, z, intensity = struct.unpack('<ffff', data[offset:offset+16])
            points.append((x, y, z, intensity))
        
        return points
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []

def load_kitti_to_open3d(file_path, max_points=None):
    """加载KITTI数据并转换为Open3D点云格式"""
    if not HAS_OPEN3D:
        return None
    
    points = load_kitti_bin_simple(file_path)
    if not points:
        return None
    
    # 下采样
    if max_points and len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]
    
    coords = [[p[0], p[1], p[2]] for p in points]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    return pcd

def register_point_clouds_open3d(source_pcd, target_pcd, method='icp'):
    """使用Open3D进行点云配准（无GUI）"""
    if not HAS_OPEN3D:
        return None, None, {}
    
    print(f"🔄 Executing {method} registration...")
    
    # 估计法向量
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    if method == 'icp':
        threshold = 2.0
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
    else:  # feature-based
        voxel_size = 1.0
        
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        radius_normal = voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
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
    
    source_registered = source_pcd.transform(reg_result.transformation)
    
    reg_info = {
        'method': method,
        'fitness': reg_result.fitness,
        'inlier_rmse': reg_result.inlier_rmse,
        'transformation': reg_result.transformation,
        'correspondence_set': len(reg_result.correspondence_set)
    }
    
    print(f"✅ Registration completed - Fitness: {reg_info['fitness']:.4f}, RMSE: {reg_info['inlier_rmse']:.4f}")
    
    return source_registered, reg_result.transformation, reg_info

def create_registration_comparison_plot(points_A_original, points_A_registered, points_B, reg_info, output_dir):
    """创建配准前后对比图（Docker友好版）"""
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib不可用，跳过图形对比")
        return
    
    print("📊 Generating registration comparison plot...")
    
    # 提取坐标（下采样以提高性能）
    def downsample_points(points, max_points=5000):
        if len(points) <= max_points:
            return points
        step = len(points) // max_points
        return points[::step]
    
    points_A_orig_down = downsample_points(points_A_original)
    points_A_reg_down = downsample_points(points_A_registered)
    points_B_down = downsample_points(points_B)
    
    x_orig = [p[0] for p in points_A_orig_down]
    y_orig = [p[1] for p in points_A_orig_down]
    z_orig = [p[2] for p in points_A_orig_down]
    
    x_reg = [p[0] for p in points_A_reg_down]
    y_reg = [p[1] for p in points_A_reg_down]
    z_reg = [p[2] for p in points_A_reg_down]
    
    x_B = [p[0] for p in points_B_down]
    y_B = [p[1] for p in points_B_down]
    z_B = [p[2] for p in points_B_down]
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 配准前3D视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(x_orig, y_orig, z_orig, c='red', s=0.5, alpha=0.6, label='Frame A (Original)')
    ax1.scatter(x_B, y_B, z_B, c='blue', s=0.5, alpha=0.6, label='Frame B (Target)')
    ax1.set_title('Before Registration - 3D View', fontsize=14)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    
    # 2. 配准后3D视图
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(x_reg, y_reg, z_reg, c='green', s=0.5, alpha=0.6, label='Frame A (Registered)')
    ax2.scatter(x_B, y_B, z_B, c='blue', s=0.5, alpha=0.6, label='Frame B (Target)')
    ax2.set_title('After Registration - 3D View', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()
    
    # 3. 配准前俯视图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(x_orig, y_orig, c='red', s=0.5, alpha=0.6, label='Frame A (Original)')
    ax3.scatter(x_B, y_B, c='blue', s=0.5, alpha=0.6, label='Frame B (Target)')
    ax3.set_title('Before Registration - Top View', fontsize=14)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. 配准后俯视图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(x_reg, y_reg, c='green', s=0.5, alpha=0.6, label='Frame A (Registered)')
    ax4.scatter(x_B, y_B, c='blue', s=0.5, alpha=0.6, label='Frame B (Target)')
    ax4.set_title('After Registration - Top View', fontsize=14)
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
        ax5.set_title('Transformation Matrix', fontsize=14)
        for i in range(4):
            for j in range(4):
                ax5.text(j, i, f'{T[i,j]:.3f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax5)
    
    # 6. 配准统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""Registration Statistics:

Method: {reg_info.get('method', 'N/A')}
Fitness: {reg_info.get('fitness', 0):.4f}
RMSE: {reg_info.get('inlier_rmse', 0):.4f}
Correspondences: {reg_info.get('correspondence_set', 0)}

Frame A Original Points: {len(points_A_original)}
Frame A Registered Points: {len(points_A_registered)}
Frame B Points: {len(points_B)}

Displayed Points (downsampled):
- Frame A Original: {len(points_A_orig_down)}
- Frame A Registered: {len(points_A_reg_down)}
- Frame B: {len(points_B_down)}
"""
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'kitti_registration_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Registration comparison plot saved to: {output_path}")
    
    plt.close()
    return output_path

def analyze_registration_differences(points_A_original, points_A_registered, points_B):
    """分析配准前后的数据差异"""
    print(f"=== Registration Effect Analysis ===")
    
    def compute_centroid(points):
        x_mean = sum(p[0] for p in points) / len(points)
        y_mean = sum(p[1] for p in points) / len(points)
        z_mean = sum(p[2] for p in points) / len(points)
        return (x_mean, y_mean, z_mean)
    
    centroid_A_orig = compute_centroid(points_A_original)
    centroid_A_reg = compute_centroid(points_A_registered)
    centroid_B = compute_centroid(points_B)
    
    print(f"Centroid Coordinates:")
    print(f"  Frame A (Original): ({centroid_A_orig[0]:.2f}, {centroid_A_orig[1]:.2f}, {centroid_A_orig[2]:.2f})")
    print(f"  Frame A (Registered): ({centroid_A_reg[0]:.2f}, {centroid_A_reg[1]:.2f}, {centroid_A_reg[2]:.2f})")
    print(f"  Frame B (Target): ({centroid_B[0]:.2f}, {centroid_B[1]:.2f}, {centroid_B[2]:.2f})")
    
    def distance_3d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    
    dist_before = distance_3d(centroid_A_orig, centroid_B)
    dist_after = distance_3d(centroid_A_reg, centroid_B)
    improvement = dist_before - dist_after
    transform_distance = distance_3d(centroid_A_orig, centroid_A_reg)
    
    print(f"\nCentroid Distance:")
    print(f"  Before Registration: {dist_before:.2f}m")
    print(f"  After Registration: {dist_after:.2f}m")
    print(f"  Improvement: {improvement:.2f}m ({improvement/dist_before*100:.1f}%)")
    print(f"  Frame A Transform Distance: {transform_distance:.2f}m")
    
    return {
        'centroid_A_original': centroid_A_orig,
        'centroid_A_registered': centroid_A_reg,
        'centroid_B': centroid_B,
        'distance_before': dist_before,
        'distance_after': dist_after,
        'improvement': improvement,
        'transform_distance': transform_distance
    }

def test_icp_convergence(source_pcd, target_pcd, max_iterations_list=None, threshold=2.0):
    """测试不同迭代次数下的ICP收敛效果"""
    if max_iterations_list is None:
        max_iterations_list = list(range(3, 21))  # 默认3-20次，步长1
    
    print(f"\n🔬 Testing Point-to-Point ICP Convergence...")
    print(f"Testing iterations: {max_iterations_list}")
    print(f"Total tests to run: {len(max_iterations_list)}")
    
    convergence_results = {}
    
    for max_iter in max_iterations_list:
        print(f"\n--- Testing {max_iter} iterations ---")
        
        # 创建点云副本
        source_copy = source_pcd.__copy__()
        target_copy = target_pcd.__copy__()
        
        # 估计法向量
        source_copy.estimate_normals()
        target_copy.estimate_normals()
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行ICP
        reg_result = o3d.pipelines.registration.registration_icp(
            source_copy, target_copy, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        
        # 记录执行时间
        execution_time = time.time() - start_time
        
        # 保存结果
        convergence_results[max_iter] = {
            'fitness': reg_result.fitness,
            'rmse': reg_result.inlier_rmse,
            'correspondences': len(reg_result.correspondence_set),
            'execution_time': execution_time
        }
        
        print(f"   Fitness: {reg_result.fitness:.6f}")
        print(f"   RMSE: {reg_result.inlier_rmse:.6f}")
        print(f"   Correspondences: {len(reg_result.correspondence_set)}")
        print(f"   Time: {execution_time:.3f}s")
    
    return convergence_results

def analyze_convergence_efficiency(convergence_results):
    """分析收敛效率 - 针对细粒度迭代优化"""
    print(f"\n📊 Fine-grained Convergence Efficiency Analysis:")
    print(f"{'Iter':<6} {'Fitness':<12} {'RMSE':<12} {'Time(s)':<8} {'Efficiency':<10} {'Quality':<8} {'Change':<8}")
    print("-" * 80)
    
    best_efficiency = 0
    best_efficiency_iter = 0
    best_quality = 0
    best_quality_iter = 0
    convergence_detected_at = None
    
    sorted_iters = sorted(convergence_results.keys())
    prev_fitness = None
    prev_rmse = None
    
    for i, max_iter in enumerate(sorted_iters):
        result = convergence_results[max_iter]
        
        # 计算效率指标：适应度/时间
        efficiency = result['fitness'] / result['execution_time']
        
        # 计算质量指标：适应度 - RMSE归一化
        quality = result['fitness'] - (result['rmse'] / 10.0)  # 简单归一化
        
        # 计算相对上次的变化
        if prev_fitness is not None:
            fitness_change = abs(result['fitness'] - prev_fitness)
            rmse_change = abs(result['rmse'] - prev_rmse)
            change_magnitude = fitness_change + rmse_change
        else:
            change_magnitude = float('inf')
        
        print(f"{max_iter:<6} {result['fitness']:<12.6f} {result['rmse']:<12.6f} "
              f"{result['execution_time']:<8.3f} {efficiency:<10.3f} {quality:<8.3f} {change_magnitude:<8.6f}")
        
        # 检测收敛（变化很小）
        if change_magnitude < 1e-5 and convergence_detected_at is None and i > 2:
            convergence_detected_at = max_iter
            print(f"    ⭐ Potential convergence detected!")
        
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_efficiency_iter = max_iter
        
        if quality > best_quality:
            best_quality = quality
            best_quality_iter = max_iter
        
        prev_fitness = result['fitness']
        prev_rmse = result['rmse']
    
    print(f"\n🏆 Best Efficiency: {best_efficiency:.3f} at {best_efficiency_iter} iterations")
    print(f"🎯 Best Quality: {best_quality:.3f} at {best_quality_iter} iterations")
    
    if convergence_detected_at:
        print(f"✅ Early convergence detected at: {convergence_detected_at} iterations")
        print(f"💡 Recommended minimum iterations: {convergence_detected_at}")
        print(f"🔧 Optimal iterations (with safety margin): {convergence_detected_at + 2}")
    else:
        print(f"⚠️  No clear convergence point detected in tested range")
        print(f"💡 Consider testing with higher iteration counts or the algorithm may need more iterations")
    
    # 分析收敛曲线的斜率变化
    fitness_values = [convergence_results[k]['fitness'] for k in sorted_iters]
    rmse_values = [convergence_results[k]['rmse'] for k in sorted_iters]
    
    # 计算连续三点的平均变化率
    if len(fitness_values) >= 5:
        recent_fitness_changes = []
        recent_rmse_changes = []
        
        for i in range(len(fitness_values) - 3):
            fitness_slope = abs(fitness_values[i+3] - fitness_values[i]) / 3
            rmse_slope = abs(rmse_values[i+3] - rmse_values[i]) / 3
            recent_fitness_changes.append(fitness_slope)
            recent_rmse_changes.append(rmse_slope)
        
        avg_recent_fitness_change = sum(recent_fitness_changes[-3:]) / 3 if len(recent_fitness_changes) >= 3 else 0
        avg_recent_rmse_change = sum(recent_rmse_changes[-3:]) / 3 if len(recent_rmse_changes) >= 3 else 0
        
        if avg_recent_fitness_change < 1e-6 and avg_recent_rmse_change < 1e-6:
            print(f"📈 Convergence curve analysis: STABLE (very low change rate)")
        elif avg_recent_fitness_change < 1e-4 and avg_recent_rmse_change < 1e-4:
            print(f"📈 Convergence curve analysis: CONVERGING (decreasing change rate)")
        else:
            print(f"� Convergence curve analysis: STILL_IMPROVING (significant changes)")
    
    return best_efficiency_iter, best_quality_iter

def create_convergence_plot(convergence_results, output_dir):
    """创建收敛分析图表"""
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib not available for plotting")
        return None
    
    print("📈 Creating convergence analysis plot...")
    
    iterations = sorted(convergence_results.keys())
    fitness_values = [convergence_results[k]['fitness'] for k in iterations]
    rmse_values = [convergence_results[k]['rmse'] for k in iterations]
    time_values = [convergence_results[k]['execution_time'] for k in iterations]
    efficiency_values = [convergence_results[k]['fitness'] / convergence_results[k]['execution_time'] for k in iterations]
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ICP Convergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Fitness vs Iterations
    ax1 = axes[0, 0]
    ax1.plot(iterations, fitness_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness vs Max Iterations')
    ax1.grid(True, alpha=0.3)
    
    # 标注数值
    for i, (x, y) in enumerate(zip(iterations, fitness_values)):
        if i % 2 == 0:  # 只标注部分点避免重叠
            ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # 2. RMSE vs Iterations
    ax2 = axes[0, 1]
    ax2.plot(iterations, rmse_values, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Max Iterations')
    ax2.grid(True, alpha=0.3)
    
    # 3. Execution Time vs Iterations
    ax3 = axes[1, 0]
    ax3.plot(iterations, time_values, 'g-^', linewidth=2, markersize=6)
    ax3.set_xlabel('Max Iterations')
    ax3.set_ylabel('Execution Time (s)')
    ax3.set_title('Execution Time vs Max Iterations')
    ax3.grid(True, alpha=0.3)
    
    # 添加线性拟合线
    import numpy as np
    z = np.polyfit(iterations, time_values, 1)
    p = np.poly1d(z)
    ax3.plot(iterations, p(iterations), "g--", alpha=0.8, 
             label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.4f}')
    ax3.legend()
    
    # 4. Efficiency Analysis
    ax4 = axes[1, 1]
    bars = ax4.bar(iterations, efficiency_values, color='orange', alpha=0.7)
    ax4.set_xlabel('Max Iterations')
    ax4.set_ylabel('Efficiency (Fitness/Time)')
    ax4.set_title('Registration Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # 标注最高效率
    max_eff_idx = efficiency_values.index(max(efficiency_values))
    max_eff_iter = iterations[max_eff_idx]
    ax4.annotate(f'Best: {max(efficiency_values):.2f}', 
                xy=(max_eff_iter, max(efficiency_values)),
                xytext=(max_eff_iter, max(efficiency_values) + max(efficiency_values)*0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'icp_convergence_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Convergence analysis saved to: {output_path}")
    
    plt.close()
    return output_path

def main():
    """主函数 - Docker环境专用"""
    print("=== Docker Environment KITTI Point Cloud Registration Analysis ===")
    print("🐳 Docker-friendly mode - No GUI visualization")
    
    # 数据路径
    kitti_data_dir = "/workspace/data/kitti"
    
    # 输出目录
    output_dir = "/workspace/PointNeXt/kitti_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证路径
    velodyne_dir = os.path.join(kitti_data_dir, "sequences", "00", "velodyne")
    if not os.path.exists(velodyne_dir):
        print(f"❌ velodyne directory not found: {velodyne_dir}")
        return
    
    velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    if len(velodyne_files) < 2:
        print("❌ Need at least 2 files for registration comparison")
        return
    
    print(f"✅ Found {len(velodyne_files)} files")
    
    if not HAS_OPEN3D:
        print("❌ Open3D not available, cannot perform registration analysis")
        return
    
    # 选择要配准的帧
    frame_A_file = velodyne_files[0]
    frame_B_file = velodyne_files[1]
    
    print(f"\nRegistration frames:")
    print(f"  Frame A: {os.path.basename(frame_A_file)}")
    print(f"  Frame B: {os.path.basename(frame_B_file)}")
    
    # 加载点云数据
    print(f"\n--- Loading point cloud data ---")
    pcd_A = load_kitti_to_open3d(frame_A_file, max_points=10000)  # 减少点数提高性能
    pcd_B = load_kitti_to_open3d(frame_B_file, max_points=10000)
    
    if pcd_A is None or pcd_B is None:
        print("❌ Point cloud loading failed")
        return
    
    print(f"Frame A: {len(pcd_A.points)} points")
    print(f"Frame B: {len(pcd_B.points)} points")
    
    # 保存原始A点云的副本
    pcd_A_original = pcd_A.__copy__()
    
    # 执行ICP配准
    print(f"\n--- Executing ICP Registration ---")
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
        plot_path = create_registration_comparison_plot(
            points_A_original, points_A_registered, points_B, reg_info, output_dir
        )
        
        # 配准统计
        print(f"\n--- Registration Results ---")
        print(f"Registration method: {reg_info['method']}")
        print(f"Fitness: {reg_info['fitness']:.4f}")
        print(f"RMSE: {reg_info['inlier_rmse']:.4f}")
        print(f"Correspondences: {reg_info['correspondence_set']}")
        
        # 收敛性测试 - 细粒度测试3-20次迭代
        print(f"\n🔬 Starting Fine-grained ICP Convergence Test...")
        print(f"🎯 Testing iterations 3-20 with step size 1")
        convergence_results = test_icp_convergence(
            pcd_A_original, pcd_B, 
            max_iterations_list=list(range(3, 21)),  # 3到20，步长为1
            threshold=2.0
        )
        best_efficiency_iter, best_quality_iter = analyze_convergence_efficiency(convergence_results)
        convergence_plot_path = create_convergence_plot(convergence_results, output_dir)
        
        print(f"\n🎉 Docker environment registration analysis completed!")
        print(f"\n" + "="*60)
        print("� FINAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"�📊 Data path: {kitti_data_dir}")
        print(f"� Results saved to: {output_dir}")
        print(f"🔄 Registration Fitness: {reg_info['fitness']:.6f}")
        print(f"📐 Registration RMSE: {reg_info['inlier_rmse']:.6f}")
        print(f"⚡ Best Efficiency at: {best_efficiency_iter} iterations")
        print(f"🎯 Best Quality at: {best_quality_iter} iterations")
        if plot_path:
            print(f"🖼️  Registration visualization: {plot_path}")
        if convergence_plot_path:
            print(f"📈 Convergence analysis: {convergence_plot_path}")
        print(f"💡 Please download the PNG files to view visualization results")
        print("="*60)
    else:
        print("❌ Registration failed")

if __name__ == "__main__":
    main()
