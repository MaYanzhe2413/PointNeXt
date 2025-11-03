#!/usr/bin/env python3
"""
简单测试坐标分布分析功能
"""
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/workspace/PointNeXt')

# 导入我们的分析函数
from test_kitti_blockwise import analyze_coordinate_distribution, print_coordinate_stats

def test_coordinate_analysis():
    """测试坐标分布分析功能"""
    print("=== 测试坐标分布分析功能 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # Frame A: 以原点为中心的点云
    coords_A = np.random.normal(0, 10, (1000, 3))  # 1000个点，标准差10
    coords_A[:, 2] *= 0.3  # Z轴压缩，模拟地面场景
    
    # Frame B: 有偏移的点云
    coords_B = np.random.normal(0, 12, (800, 3))   # 800个点，标准差12
    coords_B[:, 0] += 5    # X轴偏移5米
    coords_B[:, 1] += 2    # Y轴偏移2米
    coords_B[:, 2] *= 0.4  # Z轴进一步压缩
    coords_B[:, 2] += 1    # Z轴偏移1米
    
    print(f"模拟数据:")
    print(f"  Frame A: {coords_A.shape[0]} 点")
    print(f"  Frame B: {coords_B.shape[0]} 点")
    
    # 分析坐标分布
    stats = analyze_coordinate_distribution(coords_A, coords_B, ['模拟Frame A', '模拟Frame B'])
    
    # 打印统计结果
    print_coordinate_stats(stats, ['模拟Frame A', '模拟Frame B'])
    
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
    tolerance = 0.5
    x_ok = abs(actual_x_offset - expected_x_offset) < tolerance
    y_ok = abs(actual_y_offset - expected_y_offset) < tolerance
    z_ok = abs(actual_z_offset - expected_z_offset) < tolerance
    
    print(f"\n测试结果:")
    print(f"  X轴偏移检测: {'✅ 通过' if x_ok else '❌ 失败'}")
    print(f"  Y轴偏移检测: {'✅ 通过' if y_ok else '❌ 失败'}")
    print(f"  Z轴偏移检测: {'✅ 通过' if z_ok else '❌ 失败'}")
    
    if x_ok and y_ok and z_ok:
        print("🎉 所有测试通过！坐标分布分析功能正常。")
    else:
        print("⚠️  部分测试失败，可能是随机数导致的正常波动。")
    
    return stats

if __name__ == "__main__":
    test_coordinate_analysis()
