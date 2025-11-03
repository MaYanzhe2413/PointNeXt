import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class PointToPointICP:
    def __init__(self, max_iterations=50, tolerance=1e-6, max_correspondence_distance=0.05):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        
    def find_correspondences(self, source_points, target_points):
        """寻找对应点对"""
        # 构建目标点云的KD树
        target_tree = KDTree(target_points)
        
        correspondences = []
        distances = []
        
        for i, source_point in enumerate(source_points):
            # 寻找最近邻
            dist, idx = target_tree.query(source_point)
            
            # 距离过滤
            if dist < self.max_correspondence_distance:
                correspondences.append([i, idx])
                distances.append(dist)
        
        return np.array(correspondences), np.array(distances)
    
    def compute_transformation_svd(self, source_corr, target_corr):
        """使用SVD方法计算变换矩阵"""
        # 计算质心
        source_center = np.mean(source_corr, axis=0)
        target_center = np.mean(target_corr, axis=0)
        
        # 去质心
        source_centered = source_corr - source_center
        target_centered = target_corr - target_center
        
        # 计算协方差矩阵
        H = source_centered.T @ target_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保旋转矩阵的行列式为正（右手坐标系）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移向量
        t = target_center - R @ source_center
        
        # 构建4x4变换矩阵
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
        
        for iteration in range(self.max_iterations):
            # 步骤1: 寻找对应点对
            correspondences, distances = self.find_correspondences(current_source, target_points)
            
            if len(correspondences) < 3:
                print(f"迭代 {iteration}: 对应点数量不足，停止迭代")
                break
            
            # 步骤2: 提取对应点
            source_corr = current_source[correspondences[:, 0]]
            target_corr = target_points[correspondences[:, 1]]
            
            # 步骤3: 计算RMSE
            current_rmse = self.compute_rmse(source_corr, target_corr)
            rmse_history.append(current_rmse)
            correspondence_count_history.append(len(correspondences))
            
            print(f"迭代 {iteration}: RMSE = {current_rmse:.6f}, 对应点数 = {len(correspondences)}")
            
            # 步骤4: 检查收敛
            if iteration > 0 and abs(rmse_history[-2] - current_rmse) < self.tolerance:
                print(f"在第 {iteration} 次迭代收敛")
                break
            
            # 步骤5: 计算变换矩阵
            delta_transformation = self.compute_transformation_svd(source_corr, target_corr)
            
            # 步骤6: 更新变换和点云
            current_transformation = delta_transformation @ current_transformation
            current_source = self.apply_transformation(current_source, delta_transformation)
        
        return {
            'transformation': current_transformation,
            'rmse_history': rmse_history,
            'correspondence_count_history': correspondence_count_history,
            'final_rmse': rmse_history[-1] if rmse_history else float('inf'),
            'iterations': len(rmse_history)
        }

def create_test_data():
    """创建测试数据"""
    # 创建一个简单的点云（立方体的顶点）
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [0.5, 0.5, 0.5]  # 中心点
    ], dtype=np.float64)
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.01, points.shape)
    source_points = points + noise
    
    # 创建变换后的目标点云
    angle = np.pi / 6  # 30度
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_true = np.array([0.5, 0.3, 0.2])
    
    target_points = (R_true @ source_points.T).T + t_true
    
    # 真实变换矩阵
    true_transformation = np.eye(4)
    true_transformation[:3, :3] = R_true
    true_transformation[:3, 3] = t_true
    
    return source_points, target_points, true_transformation

def evaluate_with_open3d(source_points, target_points, max_correspondence_distance):
    """使用Open3D的ICP进行对比评价"""
    # 转换为Open3D点云格式
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_points)
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_points)
    
    # 使用Open3D的ICP
    result_o3d = o3d.pipelines.registration.registration_icp(
        source_o3d, target_o3d, max_correspondence_distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    return result_o3d

def visualize_results(source_points, target_points, my_transformation, o3d_transformation):
    """可视化结果"""
    # 应用我们的变换
    my_icp = PointToPointICP()
    my_transformed = my_icp.apply_transformation(source_points, my_transformation)
    
    # 应用Open3D的变换
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
    
    # 创建测试数据
    source_points, target_points, true_transformation = create_test_data()
    
    print("1. 测试数据创建完成")
    print(f"源点云形状: {source_points.shape}")
    print(f"目标点云形状: {target_points.shape}")
    print(f"真实变换矩阵:\n{true_transformation}\n")
    
    # 使用我们的ICP算法
    print("2. 运行自实现的ICP算法...")
    my_icp = PointToPointICP(max_correspondence_distance=0.1)
    my_result = my_icp.register(source_points, target_points)
    
    print(f"\n我们的ICP结果:")
    print(f"最终RMSE: {my_result['final_rmse']:.6f}")
    print(f"迭代次数: {my_result['iterations']}")
    print(f"变换矩阵:\n{my_result['transformation']}\n")
    
    # 使用Open3D的ICP进行对比
    print("3. 运行Open3D的ICP算法...")
    o3d_result = evaluate_with_open3d(source_points, target_points, 0.1)
    
    print(f"\nOpen3D的ICP结果:")
    print(f"适应度: {o3d_result.fitness:.6f}")
    print(f"RMSE: {o3d_result.inlier_rmse:.6f}")
    print(f"变换矩阵:\n{o3d_result.transformation}\n")
    
    # 计算与真实变换的误差
    print("4. 精度评价:")
    my_translation_error = np.linalg.norm(my_result['transformation'][:3, 3] - true_transformation[:3, 3])
    o3d_translation_error = np.linalg.norm(o3d_result.transformation[:3, 3] - true_transformation[:3, 3])
    
    my_rotation_error = np.linalg.norm(my_result['transformation'][:3, :3] - true_transformation[:3, :3], 'fro')
    o3d_rotation_error = np.linalg.norm(o3d_result.transformation[:3, :3] - true_transformation[:3, :3], 'fro')
    
    print(f"我们的ICP - 平移误差: {my_translation_error:.6f}, 旋转误差: {my_rotation_error:.6f}")
    print(f"Open3D ICP - 平移误差: {o3d_translation_error:.6f}, 旋转误差: {o3d_rotation_error:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(my_result['rmse_history'], 'b-o', label='我们的ICP')
    plt.axhline(y=o3d_result.inlier_rmse, color='r', linestyle='--', label='Open3D ICP最终RMSE')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.title('RMSE收敛曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(my_result['correspondence_count_history'], 'g-s')
    plt.xlabel('迭代次数')
    plt.ylabel('对应点数量')
    plt.title('对应点数量变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 可视化结果
    print("\n5. 可视化结果...")
    visualize_results(source_points, target_points, 
                     my_result['transformation'], 
                     o3d_result.transformation)

if __name__ == "__main__":
    main()