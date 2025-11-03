#!/usr/bin/env python3
"""
简化版坐标分布分析功能演示
不依赖numpy和torch，使用纯Python实现
"""
import math
import random

def simple_stats(data):
    """计算简单统计信息"""
    if not data:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
    
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
        'range': max_val - min_val
    }

def analyze_coordinate_distribution_simple(coords_A, coords_B, frame_names=['Frame A', 'Frame B']):
    """
    简化版坐标分布分析
    
    Args:
        coords_A: [(x, y, z), ...] - 第一个点云的坐标列表
        coords_B: [(x, y, z), ...] - 第二个点云的坐标列表
        frame_names: 两个点云的名称
    
    Returns:
        stats_dict: 包含统计信息的字典
    """
    stats = {}
    
    # 分析每个帧的坐标分布
    for coords, name in zip([coords_A, coords_B], frame_names):
        stats[name] = {}
        
        # 提取x, y, z坐标
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        z_coords = [point[2] for point in coords]
        
        # 计算每个轴的统计信息
        stats[name]['x'] = simple_stats(x_coords)
        stats[name]['y'] = simple_stats(y_coords)
        stats[name]['z'] = simple_stats(z_coords)
    
    # 计算坐标偏移
    stats['offset'] = {}
    for axis in ['x', 'y', 'z']:
        stats['offset'][axis] = {
            'mean_diff': stats[frame_names[1]][axis]['mean'] - stats[frame_names[0]][axis]['mean'],
            'std_diff': stats[frame_names[1]][axis]['std'] - stats[frame_names[0]][axis]['std'],
            'range_diff': stats[frame_names[1]][axis]['range'] - stats[frame_names[0]][axis]['range']
        }
    
    return stats

def print_coordinate_stats_simple(stats, frame_names=['Frame A', 'Frame B']):
    """
    打印坐标统计信息
    
    Args:
        stats: analyze_coordinate_distribution_simple返回的统计字典
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

def generate_sample_data():
    """生成示例点云数据"""
    random.seed(42)
    
    # Frame A: 以原点为中心的点云
    coords_A = []
    for _ in range(1000):
        x = random.gauss(0, 10)    # 均值0，标准差10
        y = random.gauss(0, 10)
        z = random.gauss(0, 3)     # Z轴压缩，模拟地面场景
        coords_A.append((x, y, z))
    
    # Frame B: 有偏移的点云
    coords_B = []
    for _ in range(800):
        x = random.gauss(5, 12)    # X轴偏移5米，标准差12
        y = random.gauss(2, 12)    # Y轴偏移2米
        z = random.gauss(1, 4)     # Z轴偏移1米，标准差4
        coords_B.append((x, y, z))
    
    return coords_A, coords_B

def test_coordinate_analysis_simple():
    """测试简化版坐标分布分析功能"""
    print("=== 测试简化版坐标分布分析功能 ===")
    print("(使用纯Python实现，不依赖numpy/torch)")
    
    # 生成示例数据
    coords_A, coords_B = generate_sample_data()
    
    print(f"\n模拟数据:")
    print(f"  Frame A: {len(coords_A)} 点")
    print(f"  Frame B: {len(coords_B)} 点")
    
    # 分析坐标分布
    stats = analyze_coordinate_distribution_simple(coords_A, coords_B, ['模拟Frame A', '模拟Frame B'])
    
    # 打印统计结果
    print_coordinate_stats_simple(stats, ['模拟Frame A', '模拟Frame B'])
    
    # 验证结果
    print("\n=== 验证结果 ===")
    expected_x_offset = 5.0
    actual_x_offset = stats['offset']['x']['mean_diff']
    print(f"X轴偏移 - 期望: {expected_x_offset:.2f}m, 实际: {actual_x_offset:.2f}m")
    
    expected_y_offset = 2.0
    actual_y_offset = stats['offset']['y']['mean_diff']
    print(f"Y轴偏移 - 期望: {expected_y_offset:.2f}m, 实际: {actual_y_offset:.2f}m")
    
    expected_z_offset = 1.0
    actual_z_offset = stats['offset']['z']['mean_diff']
    print(f"Z轴偏移 - 期望: {expected_z_offset:.2f}m, 实际: {actual_z_offset:.2f}m")
    
    # 检查偏差是否在合理范围内
    tolerance = 1.0  # 由于随机性，放宽容差
    x_ok = abs(actual_x_offset - expected_x_offset) < tolerance
    y_ok = abs(actual_y_offset - expected_y_offset) < tolerance
    z_ok = abs(actual_z_offset - expected_z_offset) < tolerance
    
    print(f"\n测试结果:")
    print(f"  X轴偏移检测: {'✅ 通过' if x_ok else '❌ 失败'} (误差: {abs(actual_x_offset - expected_x_offset):.2f}m)")
    print(f"  Y轴偏移检测: {'✅ 通过' if y_ok else '❌ 失败'} (误差: {abs(actual_y_offset - expected_y_offset):.2f}m)")
    print(f"  Z轴偏移检测: {'✅ 通过' if z_ok else '❌ 失败'} (误差: {abs(actual_z_offset - expected_z_offset):.2f}m)")
    
    if x_ok and y_ok and z_ok:
        print("🎉 所有测试通过！坐标分布分析功能正常。")
    else:
        print("⚠️  部分测试失败，但这可能是由于随机数导致的正常波动。")
    
    return stats

def demo_kitti_like_analysis():
    """演示类似KITTI数据的分析"""
    print("\n" + "="*60)
    print("=== KITTI风格点云分析演示 ===")
    
    # 模拟KITTI点云数据特征
    random.seed(123)
    
    # Frame A: 车辆在t时刻的观测
    coords_A = []
    for _ in range(5000):
        # 前方扇形区域的点云
        distance = random.uniform(5, 80)  # 5-80米范围
        angle = random.uniform(-math.pi/3, math.pi/3)  # 左右各60度
        
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        z = random.gauss(0, 2)  # 地面附近，偶有建筑物和车辆
        
        coords_A.append((x, y, z))
    
    # Frame B: 车辆在t+1时刻的观测（车辆向前移动了3米）
    coords_B = []
    for _ in range(4800):
        distance = random.uniform(5, 80)
        angle = random.uniform(-math.pi/3, math.pi/3)
        
        # 模拟车辆前进3米的效果
        x = distance * math.cos(angle) - 3  # 相对位移
        y = distance * math.sin(angle) + random.gauss(0, 0.5)  # 轻微侧向漂移
        z = random.gauss(0, 2)
        
        coords_B.append((x, y, z))
    
    print(f"KITTI风格数据:")
    print(f"  Frame A (t时刻): {len(coords_A)} 点")
    print(f"  Frame B (t+1时刻): {len(coords_B)} 点")
    print(f"  车辆预期前进: 3.0米")
    
    # 分析坐标分布
    stats = analyze_coordinate_distribution_simple(coords_A, coords_B, ['KITTI Frame A', 'KITTI Frame B'])
    
    # 打印统计结果
    print_coordinate_stats_simple(stats, ['KITTI Frame A', 'KITTI Frame B'])
    
    # 分析车辆运动
    vehicle_forward = -stats['offset']['x']['mean_diff']  # X轴负向表示前进
    vehicle_lateral = stats['offset']['y']['mean_diff']
    
    print(f"\n=== 车辆运动分析 ===")
    print(f"车辆前进距离: {vehicle_forward:.2f}m (预期: 3.0m)")
    print(f"车辆侧向偏移: {vehicle_lateral:.2f}m (预期: ~0m)")
    
    forward_error = abs(vehicle_forward - 3.0)
    lateral_error = abs(vehicle_lateral)
    
    print(f"前进距离误差: {forward_error:.2f}m")
    print(f"侧向偏移误差: {lateral_error:.2f}m")
    
    if forward_error < 1.0 and lateral_error < 1.0:
        print("✅ 车辆运动估计准确！")
    else:
        print("⚠️  车辆运动估计存在误差，可能需要更精确的配准。")

if __name__ == "__main__":
    # 运行基础测试
    test_coordinate_analysis_simple()
    
    # 运行KITTI风格演示
    demo_kitti_like_analysis()
    
    print("\n" + "="*60)
    print("📋 总结:")
    print("1. 成功实现了纯Python版本的坐标分布分析")
    print("2. 可以检测两个点云间的空间偏移")
    print("3. 适用于KITTI等自动驾驶场景的分析")
    print("4. 当numpy/torch可用时，可以用更高效的向量化实现")
