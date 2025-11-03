#!/usr/bin/env python3
"""
Point-to-Point ICP收敛性测试脚本
分析不同迭代次数下的收敛效果和性能
"""
import os
import sys
import glob
import struct
import math
import time
import numpy as np

# 强制matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 导入Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
    print("✅ Open3D loaded for convergence testing")
except ImportError:
    HAS_OPEN3D = False
    print("❌ Open3D not available")
    sys.exit(1)

def load_kitti_bin_simple(file_path):
    """加载KITTI .bin文件"""
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
        print(f"Failed to load file: {e}")
        return []

def load_kitti_to_open3d(file_path, max_points=None):
    """加载KITTI数据并转换为Open3D点云格式"""
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

def create_initial_transformation(translation_range=5.0, rotation_range=0.3):
    """创建初始变换来测试收敛性"""
    # 随机平移
    translation = np.random.uniform(-translation_range, translation_range, 3)
    
    # 随机旋转（欧拉角）
    rx = np.random.uniform(-rotation_range, rotation_range)
    ry = np.random.uniform(-rotation_range, rotation_range)
    rz = np.random.uniform(-rotation_range, rotation_range)
    
    # 构建旋转矩阵
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)
    
    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])
    
    R_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])
    
    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]])
    
    rotation = R_z @ R_y @ R_x
    
    # 构建4x4变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    
    return transformation

class ICPConvergenceTracker:
    """ICP收敛过程跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置跟踪数据"""
        self.iterations = []
        self.fitness_history = []
        self.rmse_history = []
        self.correspondence_history = []
        self.transformation_history = []
        self.execution_times = []
        self.converged = False
        self.convergence_iteration = -1
    
    def add_iteration_data(self, iteration, fitness, rmse, correspondences, transformation, exec_time):
        """添加迭代数据"""
        self.iterations.append(iteration)
        self.fitness_history.append(fitness)
        self.rmse_history.append(rmse)
        self.correspondence_history.append(correspondences)
        self.transformation_history.append(transformation.copy())
        self.execution_times.append(exec_time)
    
    def check_convergence(self, tolerance=1e-6):
        """检查是否收敛"""
        if len(self.rmse_history) < 2:
            return False
        
        # 检查RMSE变化是否小于阈值
        rmse_change = abs(self.rmse_history[-1] - self.rmse_history[-2])
        if rmse_change < tolerance and not self.converged:
            self.converged = True
            self.convergence_iteration = len(self.rmse_history) - 1
            return True
        
        return self.converged

def run_icp_with_max_iterations(source_pcd, target_pcd, max_iterations_list, threshold=2.0):
    """运行不同最大迭代次数的ICP测试"""
    print(f"🔬 Testing ICP convergence with different max iterations...")
    
    results = {}
    
    for max_iter in max_iterations_list:
        print(f"\n--- Testing with max_iterations = {max_iter} ---")
        
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
        
        # 记录结束时间
        execution_time = time.time() - start_time
        
        # 保存结果
        results[max_iter] = {
            'fitness': reg_result.fitness,
            'rmse': reg_result.inlier_rmse,
            'correspondences': len(reg_result.correspondence_set),
            'transformation': reg_result.transformation,
            'execution_time': execution_time,
            'reg_result': reg_result
        }
        
        print(f"   Fitness: {reg_result.fitness:.6f}")
        print(f"   RMSE: {reg_result.inlier_rmse:.6f}")
        print(f"   Correspondences: {len(reg_result.correspondence_set)}")
        print(f"   Execution time: {execution_time:.3f}s")
    
    return results

def run_custom_icp_iteration_tracking(source_pcd, target_pcd, max_iterations=100, threshold=2.0):
    """自定义ICP迭代跟踪（逐步执行）"""
    print(f"🔍 Running custom ICP with iteration tracking...")
    
    tracker = ICPConvergenceTracker()
    
    # 创建点云副本
    source_current = source_pcd.__copy__()
    target_copy = target_pcd.__copy__()
    
    # 估计法向量
    source_current.estimate_normals()
    target_copy.estimate_normals()
    
    # 逐步迭代
    for iteration in range(1, max_iterations + 1):
        start_time = time.time()
        
        # 执行单次迭代的ICP
        reg_result = o3d.pipelines.registration.registration_icp(
            source_current, target_copy, threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        
        exec_time = time.time() - start_time
        
        # 记录数据
        tracker.add_iteration_data(
            iteration, 
            reg_result.fitness,
            reg_result.inlier_rmse,
            len(reg_result.correspondence_set),
            reg_result.transformation,
            exec_time
        )
        
        # 应用变换到源点云
        source_current.transform(reg_result.transformation)
        
        # 检查收敛
        converged = tracker.check_convergence(tolerance=1e-6)
        
        if iteration % 10 == 0 or converged:
            print(f"   Iteration {iteration}: Fitness={reg_result.fitness:.6f}, "
                  f"RMSE={reg_result.inlier_rmse:.6f}, "
                  f"Correspondences={len(reg_result.correspondence_set)}")
        
        if converged:
            print(f"   ✅ Converged at iteration {tracker.convergence_iteration}")
            break
        
        # 如果RMSE不再显著改善，提前停止
        if iteration > 10:
            recent_rmse = tracker.rmse_history[-5:]
            if len(set([round(x, 8) for x in recent_rmse])) == 1:
                print(f"   🔄 RMSE stabilized at iteration {iteration}")
                break
    
    return tracker

def analyze_convergence_patterns(results, tracker, output_dir):
    """分析收敛模式"""
    print(f"\n🔍 Analyzing convergence patterns...")
    
    # 创建收敛分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ICP Convergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. 不同最大迭代次数的效果对比
    ax1 = axes[0, 0]
    max_iters = sorted(results.keys())
    fitness_values = [results[k]['fitness'] for k in max_iters]
    rmse_values = [results[k]['rmse'] for k in max_iters]
    exec_times = [results[k]['execution_time'] for k in max_iters]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(max_iters, fitness_values, 'b-o', label='Fitness')
    line2 = ax1_twin.plot(max_iters, rmse_values, 'r-s', label='RMSE')
    
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Fitness', color='b')
    ax1_twin.set_ylabel('RMSE', color='r')
    ax1.set_title('Effect of Max Iterations')
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # 2. 执行时间 vs 最大迭代次数
    ax2 = axes[0, 1]
    ax2.plot(max_iters, exec_times, 'g-^', linewidth=2, markersize=8)
    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Execution Time vs Max Iterations')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(max_iters, exec_times):
        ax2.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # 3. 迭代过程中的适应度变化
    ax3 = axes[0, 2]
    if tracker.fitness_history:
        ax3.plot(tracker.iterations, tracker.fitness_history, 'b-o', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Fitness')
        ax3.set_title('Fitness Evolution')
        ax3.grid(True, alpha=0.3)
        
        if tracker.converged:
            ax3.axvline(x=tracker.convergence_iteration, color='red', 
                       linestyle='--', label=f'Converged at {tracker.convergence_iteration}')
            ax3.legend()
    
    # 4. 迭代过程中的RMSE变化
    ax4 = axes[1, 0]
    if tracker.rmse_history:
        ax4.plot(tracker.iterations, tracker.rmse_history, 'r-s', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('RMSE')
        ax4.set_title('RMSE Evolution')
        ax4.grid(True, alpha=0.3)
        
        # 标注收敛点
        if tracker.converged:
            ax4.axvline(x=tracker.convergence_iteration, color='blue', 
                       linestyle='--', label=f'Converged at {tracker.convergence_iteration}')
            ax4.legend()
    
    # 5. 对应点数变化
    ax5 = axes[1, 1]
    if tracker.correspondence_history:
        ax5.plot(tracker.iterations, tracker.correspondence_history, 'g-^', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Number of Correspondences')
        ax5.set_title('Correspondences Evolution')
        ax5.grid(True, alpha=0.3)
    
    # 6. 收敛效率分析
    ax6 = axes[1, 2]
    
    # 计算收敛效率指标
    efficiency_data = []
    efficiency_labels = []
    
    for max_iter in max_iters:
        result = results[max_iter]
        # 效率 = 适应度 / 执行时间
        efficiency = result['fitness'] / result['execution_time']
        efficiency_data.append(efficiency)
        efficiency_labels.append(f'{max_iter}')
    
    bars = ax6.bar(efficiency_labels, efficiency_data, 
                   color=['red' if i == max_iters.index(max(max_iters, key=lambda x: results[x]['fitness']/results[x]['execution_time'])) 
                         else 'lightblue' for i in range(len(max_iters))])
    
    ax6.set_xlabel('Max Iterations')
    ax6.set_ylabel('Efficiency (Fitness/Time)')
    ax6.set_title('Registration Efficiency')
    ax6.grid(True, alpha=0.3)
    
    # 标注最佳效率
    best_idx = efficiency_data.index(max(efficiency_data))
    ax6.annotate(f'Best: {efficiency_data[best_idx]:.2f}', 
                xy=(best_idx, efficiency_data[best_idx]),
                xytext=(best_idx, efficiency_data[best_idx] + max(efficiency_data)*0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'icp_convergence_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Convergence analysis saved to: {output_path}")
    
    plt.close()
    return output_path

def print_convergence_summary(results, tracker):
    """打印收敛分析摘要"""
    print(f"\n{'='*60}")
    print("=== ICP CONVERGENCE ANALYSIS SUMMARY ===")
    print(f"{'='*60}")
    
    # 最大迭代次数测试结果
    print(f"\n1. Max Iterations Test Results:")
    print(f"{'Max Iter':<10} {'Fitness':<12} {'RMSE':<12} {'Corresp':<10} {'Time(s)':<10} {'Efficiency':<12}")
    print("-" * 70)
    
    best_fitness = -1
    best_iter_fitness = 0
    best_efficiency = -1
    best_iter_efficiency = 0
    
    for max_iter in sorted(results.keys()):
        result = results[max_iter]
        efficiency = result['fitness'] / result['execution_time']
        
        print(f"{max_iter:<10} {result['fitness']:<12.6f} {result['rmse']:<12.6f} "
              f"{result['correspondences']:<10} {result['execution_time']:<10.3f} {efficiency:<12.3f}")
        
        if result['fitness'] > best_fitness:
            best_fitness = result['fitness']
            best_iter_fitness = max_iter
        
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_iter_efficiency = max_iter
    
    print(f"\n🏆 Best Fitness: {best_fitness:.6f} (Max Iter: {best_iter_fitness})")
    print(f"⚡ Best Efficiency: {best_efficiency:.3f} (Max Iter: {best_iter_efficiency})")
    
    # 迭代跟踪结果
    if tracker.iterations:
        print(f"\n2. Iteration Tracking Results:")
        print(f"   Total iterations: {len(tracker.iterations)}")
        print(f"   Final fitness: {tracker.fitness_history[-1]:.6f}")
        print(f"   Final RMSE: {tracker.rmse_history[-1]:.6f}")
        print(f"   Final correspondences: {tracker.correspondence_history[-1]}")
        print(f"   Total execution time: {sum(tracker.execution_times):.3f}s")
        
        if tracker.converged:
            print(f"   ✅ Converged at iteration: {tracker.convergence_iteration}")
        else:
            print(f"   ⚠️  Did not converge within {len(tracker.iterations)} iterations")
        
        # 计算收敛速度
        if len(tracker.rmse_history) > 1:
            initial_rmse = tracker.rmse_history[0]
            final_rmse = tracker.rmse_history[-1]
            improvement_rate = (initial_rmse - final_rmse) / len(tracker.iterations)
            print(f"   📈 RMSE improvement rate: {improvement_rate:.6f} per iteration")
    
    # 推荐设置
    print(f"\n3. Recommendations:")
    
    # 基于效率推荐
    efficient_iter = max(results.keys(), key=lambda x: results[x]['fitness']/results[x]['execution_time'])
    print(f"   🎯 Recommended max_iterations for efficiency: {efficient_iter}")
    
    # 基于质量推荐
    quality_iter = max(results.keys(), key=lambda x: results[x]['fitness'])
    print(f"   🎯 Recommended max_iterations for quality: {quality_iter}")
    
    # 基于收敛分析推荐
    if tracker.converged:
        recommended_iter = max(10, tracker.convergence_iteration + 5)
        print(f"   🎯 Recommended based on convergence analysis: {recommended_iter}")
    
    print(f"\n💡 For real-time applications, use {efficient_iter} iterations")
    print(f"💡 For high-accuracy applications, use {quality_iter} iterations")

def main():
    """主函数"""
    print("=== Point-to-Point ICP Convergence Testing ===")
    
    # 数据路径
    kitti_data_dir = "/workspace/data/kitti"
    output_dir = "/workspace/PointNeXt/kitti_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证路径
    velodyne_dir = os.path.join(kitti_data_dir, "sequences", "00", "velodyne")
    if not os.path.exists(velodyne_dir):
        print(f"❌ velodyne directory not found: {velodyne_dir}")
        return
    
    velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    if len(velodyne_files) < 2:
        print("❌ Need at least 2 files for convergence testing")
        return
    
    print(f"✅ Found {len(velodyne_files)} files")
    
    # 选择测试帧
    frame_A_file = velodyne_files[0]
    frame_B_file = velodyne_files[1]
    
    print(f"\nTest frames:")
    print(f"  Frame A: {os.path.basename(frame_A_file)}")
    print(f"  Frame B: {os.path.basename(frame_B_file)}")
    
    # 加载点云数据
    print(f"\n--- Loading point cloud data ---")
    pcd_A = load_kitti_to_open3d(frame_A_file, max_points=8000)
    pcd_B = load_kitti_to_open3d(frame_B_file, max_points=8000)
    
    if pcd_A is None or pcd_B is None:
        print("❌ Point cloud loading failed")
        return
    
    print(f"Frame A: {len(pcd_A.points)} points")
    print(f"Frame B: {len(pcd_B.points)} points")
    
    # 可选：添加初始扰动来测试收敛性
    print(f"\n--- Adding initial perturbation for testing ---")
    initial_transform = create_initial_transformation(translation_range=3.0, rotation_range=0.2)
    pcd_A.transform(initial_transform)
    print(f"Applied initial transformation with ~3m translation and ~11° rotation")
    
    # 测试不同的最大迭代次数
    print(f"\n{'='*60}")
    print("=== TESTING DIFFERENT MAX ITERATIONS ===")
    max_iterations_list = [5, 10, 20, 30, 50, 100, 200]
    
    results = run_icp_with_max_iterations(pcd_A, pcd_B, max_iterations_list)
    
    # 执行迭代跟踪
    print(f"\n{'='*60}")
    print("=== DETAILED ITERATION TRACKING ===")
    tracker = run_custom_icp_iteration_tracking(pcd_A, pcd_B, max_iterations=100)
    
    # 分析收敛模式
    print(f"\n{'='*60}")
    print("=== CONVERGENCE PATTERN ANALYSIS ===")
    visualization_path = analyze_convergence_patterns(results, tracker, output_dir)
    
    # 打印摘要
    print_convergence_summary(results, tracker)
    
    print(f"\n🎉 ICP convergence testing completed!")
    print(f"📊 Results saved to: {output_dir}")
    if visualization_path:
        print(f"📈 Visualization: {visualization_path}")
    print(f"💡 Use the recommendations above to optimize your ICP settings")

if __name__ == "__main__":
    main()
