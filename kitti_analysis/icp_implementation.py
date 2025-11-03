import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置matplotlib支持中文显示的完整方案
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    import platform
    system = platform.system()
    
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'DejaVu Sans']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass

def load_bin_file(file_path):
    """
    加载.bin格式的点云文件
    假设文件格式为KITTI格式：每个点包含x,y,z,intensity (4个float32值)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在！")
    
    # 读取二进制文件
    points = np.fromfile(file_path, dtype=np.float32)
    
    # 重新整形为 (N, 4) 的数组，假设每个点有 x,y,z,intensity
    if len(points) % 4 != 0:
        print(f"警告: 文件 {file_path} 的数据长度不是4的倍数，可能格式不正确")
        # 尝试只使用x,y,z坐标（3个值一组）
        if len(points) % 3 == 0:
            points = points.reshape(-1, 3)
            print(f"尝试按照x,y,z格式读取，点数: {len(points)}")
        else:
            raise ValueError(f"无法解析文件 {file_path} 的格式")
    else:
        points = points.reshape(-1, 4)
        # 只取前3列作为xyz坐标
        points = points[:, :3]
        print(f"成功读取文件 {file_path}，点数: {len(points)}")
    
    return points

def preprocess_point_cloud(points, voxel_size=0.1, max_points=10000):
    """
    预处理点云：下采样和去除离群点
    """
    # 转换为Open3D点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 体素下采样
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"体素下采样后点数: {len(pcd.points)}")
    
    # 如果点数仍然太多，随机采样
    if len(pcd.points) > max_points:
        pcd = pcd.random_down_sample(max_points / len(pcd.points))
        print(f"随机采样后点数: {len(pcd.points)}")
    
    # 去除离群点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"去除离群点后点数: {len(pcd.points)}")
    
    return np.asarray(pcd.points)

class PointToPointICP:
    def __init__(self, max_iterations=50, tolerance=1e-6, max_correspondence_distance=0.5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        
    def find_correspondences(self, source_points, target_points):
        """寻找对应点对"""
        target_tree = KDTree(target_points)
        correspondences = []
        distances = []
        
        for i, source_point in enumerate(source_points):
            dist, idx = target_tree.query(source_point)
            if dist < self.max_correspondence_distance:
                correspondences.append([i, idx])
                distances.append(dist)
            # correspondences.append([i, idx])
            # distances.append(dist)
        
        return np.array(correspondences), np.array(distances)
    
    def compute_transformation_svd(self, source_corr, target_corr):
        """使用SVD方法计算变换矩阵"""
        source_center = np.mean(source_corr, axis=0)
        target_center = np.mean(target_corr, axis=0)
        
        source_centered = source_corr - source_center
        target_centered = target_corr - target_center
        
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = target_center - R @ source_center
        
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        return transformation
    
    def apply_transformation(self, points, transformation):
        """应用变换矩阵到点云"""
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack([points, ones])
        transformed = (transformation @ homogeneous_points.T).T
        return transformed[:, :3]
    
    def compute_rmse(self, source_corr, target_corr):
        """计算均方根误差"""
        return np.sqrt(np.mean(np.sum((source_corr - target_corr)**2, axis=1)))
    
    def register(self, source_points, target_points, init_transformation=None):
        """执行ICP配准"""
        if init_transformation is None:
            current_transformation = np.eye(4)
        else:
            current_transformation = init_transformation.copy()
        
        current_source = source_points.copy()
        rmse_history = []
        correspondence_count_history = []
        transformation_history = []
        
        for iteration in range(self.max_iterations):
            # 寻找对应点对
            correspondences, distances = self.find_correspondences(current_source, target_points)
            
            if len(correspondences) < 3:
                print(f"迭代 {iteration}: 对应点数量不足，停止迭代")
                break
            
            source_corr = current_source[correspondences[:, 0]]
            target_corr = target_points[correspondences[:, 1]]
            
            # 计算和应用变换
            delta_transformation = self.compute_transformation_svd(source_corr, target_corr)
            current_transformation = delta_transformation @ current_transformation
            transformation_history.append(current_transformation.copy())
            current_source = self.apply_transformation(current_source, delta_transformation)
            
            # 重新计算变换后的RMSE
            correspondences_after, _ = self.find_correspondences(current_source, target_points)
            if len(correspondences_after) >= 3:
                source_corr_after = current_source[correspondences_after[:, 0]]
                target_corr_after = target_points[correspondences_after[:, 1]]
                current_rmse = self.compute_rmse(source_corr_after, target_corr_after)
            else:
                current_rmse = float('inf')
            
            rmse_history.append(current_rmse)
            correspondence_count_history.append(len(correspondences_after) if len(correspondences_after) >= 3 else len(correspondences))
            
            print(f"迭代 {iteration}: RMSE = {current_rmse:.6f}, 对应点数 = {correspondence_count_history[-1]}")
            
            # 收敛检查
            if iteration > 0 and abs(rmse_history[-2] - current_rmse) < self.tolerance:
                print(f"在第 {iteration} 次迭代收敛")
                break
        
        return {
            'transformation': current_transformation,
            'rmse_history': rmse_history,
            'correspondence_count_history': correspondence_count_history,
            'transformation_history': transformation_history,
            'final_rmse': rmse_history[-1] if rmse_history else float('inf'),
            'iterations': len(rmse_history)
        }

def evaluate_with_open3d_iterative(source_points, target_points, max_correspondence_distance, max_iterations):
    """使用Open3D的ICP进行逐步迭代对比"""
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_points)
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_points)
    
    # 存储每次迭代的结果
    o3d_rmse_history = []
    o3d_transformation_history = []
    
    current_transformation = np.eye(4)
    
    print("\nOpen3D逐步迭代:")
    for i in range(max_iterations):
        # 每次只迭代1步
        result_o3d = o3d.pipelines.registration.registration_icp(
            source_o3d, target_o3d, max_correspondence_distance, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)  # 只迭代1次
        )
        
        current_transformation = result_o3d.transformation
        o3d_rmse_history.append(result_o3d.inlier_rmse)
        o3d_transformation_history.append(current_transformation.copy())
        
        print(f"Open3D迭代 {i}: RMSE = {result_o3d.inlier_rmse:.6f}, 适应度 = {result_o3d.fitness:.6f}")
        
        # 检查收敛
        if i > 0 and abs(o3d_rmse_history[-2] - o3d_rmse_history[-1]) < 1e-6:
            print(f"Open3D在第 {i} 次迭代收敛")
            break
    
    return {
        'rmse_history': o3d_rmse_history,
        'transformation_history': o3d_transformation_history,
        'final_transformation': current_transformation,
        'final_rmse': o3d_rmse_history[-1] if o3d_rmse_history else float('inf'),
        'iterations': len(o3d_rmse_history)
    }

def compare_iterations(my_result, o3d_result):
    """对比每次迭代的结果"""
    print("\n=== 逐步迭代对比 ===")
    print(f"{'迭代':<6} {'我们的RMSE':<12} {'Open3D RMSE':<12} {'RMSE差异':<10} {'变换差异':<10}")
    print("-" * 60)
    
    min_iterations = min(len(my_result['rmse_history']), len(o3d_result['rmse_history']))
    
    for i in range(min_iterations):
        my_rmse = my_result['rmse_history'][i]
        o3d_rmse = o3d_result['rmse_history'][i]
        rmse_diff = abs(my_rmse - o3d_rmse)
        
        # 计算变换矩阵差异
        if i < len(my_result['transformation_history']) and i < len(o3d_result['transformation_history']):
            trans_diff = np.linalg.norm(
                my_result['transformation_history'][i] - o3d_result['transformation_history'][i]
            )
        else:
            trans_diff = 0.0
        
        print(f"{i:<6} {my_rmse:<12.6f} {o3d_rmse:<12.6f} {rmse_diff:<10.6f} {trans_diff:<10.6f}")

def visualize_comparison(my_result, o3d_result):
    """可视化对比结果"""
    plt.figure(figsize=(15, 5))
    
    # 子图1: RMSE对比
    plt.subplot(1, 3, 1)
    iterations_my = range(len(my_result['rmse_history']))
    iterations_o3d = range(len(o3d_result['rmse_history']))
    
    plt.plot(iterations_my, my_result['rmse_history'], 'b-o', label='Our ICP', linewidth=2)
    plt.plot(iterations_o3d, o3d_result['rmse_history'], 'r-s', label='Open3D ICP', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('RMSE Convergence Comparison')
    plt.legend()
    plt.grid(True)
    
    # 子图2: RMSE差异
    plt.subplot(1, 3, 2)
    min_iterations = min(len(my_result['rmse_history']), len(o3d_result['rmse_history']))
    rmse_differences = [
        abs(my_result['rmse_history'][i] - o3d_result['rmse_history'][i])
        for i in range(min_iterations)
    ]
    plt.plot(range(min_iterations), rmse_differences, 'g-^', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE Difference')
    plt.title('RMSE Difference per Iteration')
    plt.grid(True)
    
    # 子图3: 对应点数量
    plt.subplot(1, 3, 3)
    if 'correspondence_count_history' in my_result:
        plt.plot(iterations_my, my_result['correspondence_count_history'], 'purple', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Correspondence Count')
        plt.title('Correspondence Count (Our ICP)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_results(source_points, target_points, my_transformation, o3d_transformation):
    """可视化结果"""
    my_icp = PointToPointICP()
    my_transformed = my_icp.apply_transformation(source_points, my_transformation)
    
    o3d_icp = PointToPointICP()
    o3d_transformed = o3d_icp.apply_transformation(source_points, o3d_transformation)
    
    # 创建Open3D点云用于可视化
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_points)
    source_o3d.paint_uniform_color([1, 0, 0])  # 红色：原始源点云
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_points)
    target_o3d.paint_uniform_color([0, 1, 0])  # 绿色：目标点云
    
    my_result_o3d = o3d.geometry.PointCloud()
    my_result_o3d.points = o3d.utility.Vector3dVector(my_transformed)
    my_result_o3d.paint_uniform_color([0, 0, 1])  # 蓝色：我们的ICP结果
    
    o3d_result_o3d = o3d.geometry.PointCloud()
    o3d_result_o3d.points = o3d.utility.Vector3dVector(o3d_transformed)
    o3d_result_o3d.paint_uniform_color([1, 1, 0])  # 黄色：Open3D的ICP结果
    
    print("可视化说明:")
    print("红色: 原始源点云")
    print("绿色: 目标点云")
    print("蓝色: 我们的ICP结果")
    print("黄色: Open3D的ICP结果")
    
    o3d.visualization.draw_geometries([source_o3d, target_o3d, my_result_o3d, o3d_result_o3d])

def main():
    """主函数"""
    print("=== 自实现ICP算法测试与评价 ===\n")
    
    # 设置中文字体支持
    setup_chinese_font()
    
    # 文件路径
    source_file = "000001.bin"
    target_file = "000002.bin"
    
    try:
        # 1. 加载点云文件
        print("1. 加载点云文件...")
        source_points_raw = load_bin_file(source_file)
        target_points_raw = load_bin_file(target_file)
        
        print(f"源点云原始点数: {len(source_points_raw)}")
        print(f"目标点云原始点数: {len(target_points_raw)}")
        
        # 2. 预处理点云
        print("\n2. 预处理点云...")
        voxel_size = 0.1  # 根据点云尺度调整
        source_points = preprocess_point_cloud(source_points_raw, voxel_size)
        target_points = preprocess_point_cloud(target_points_raw, voxel_size)
        
        print(f"预处理后源点云点数: {len(source_points)}")
        print(f"预处理后目标点云点数: {len(target_points)}")
        
        # 3. 计算合适的距离阈值
        source_scale = np.std(source_points)
        target_scale = np.std(target_points)
        avg_scale = (source_scale + target_scale) / 2
        max_correspondence_distance = avg_scale * 0.1
        
        print(f"\n自动计算的距离阈值: {max_correspondence_distance:.4f}")
        
        # 4. 设置相同的迭代次数
        max_iterations = 20
        print(f"设置最大迭代次数: {max_iterations}")
        
        # 5. 使用我们的ICP算法
        print("\n3. 运行自实现的ICP算法...")
        my_icp = PointToPointICP(
            max_iterations=max_iterations,
            max_correspondence_distance=max_correspondence_distance
        )
        my_result = my_icp.register(source_points, target_points)
        
        # 6. 使用Open3D的ICP进行逐步对比
        print("\n4. 运行Open3D的ICP算法（逐步迭代）...")
        o3d_result = evaluate_with_open3d_iterative(
            source_points, target_points, max_correspondence_distance, max_iterations
        )
        
        # 7. 显示最终结果
        print(f"\n=== 最终结果对比 ===")
        print(f"我们的ICP - 最终RMSE: {my_result['final_rmse']:.6f}, 迭代次数: {my_result['iterations']}")
        print(f"Open3D ICP - 最终RMSE: {o3d_result['final_rmse']:.6f}, 迭代次数: {o3d_result['iterations']}")
        
        # 8. 逐步迭代对比
        compare_iterations(my_result, o3d_result)
        
        # 9. 可视化对比
        print("\n5. 可视化对比结果...")
        visualize_comparison(my_result, o3d_result)
        
        # 10. 最终精度评价
        final_rmse_diff = abs(my_result['final_rmse'] - o3d_result['final_rmse'])
        final_transformation_diff = np.linalg.norm(
            my_result['transformation'] - o3d_result['final_transformation']
        )
        
        print(f"\n=== 最终精度评价 ===")
        print(f"最终RMSE差异: {final_rmse_diff:.6f}")
        print(f"最终变换矩阵差异: {final_transformation_diff:.6f}")
        
        if final_rmse_diff < 0.001 and final_transformation_diff < 0.1:
            print("✅ 算法实现正确！与Open3D结果高度一致")
        elif final_rmse_diff < 0.01:
            print("⚠️ 算法基本正确，存在小幅差异")
        else:
            print("❌ 算法可能存在问题，差异较大")
        
        # 11. 可视化最终结果
        print("\n6. 可视化最终配准结果...")
        visualize_results(source_points, target_points, 
                         my_result['transformation'], 
                         o3d_result['final_transformation'])
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 000001.bin 和 000002.bin 文件在当前目录下")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()