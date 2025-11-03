#!/usr/bin/env python3
"""
分析KDTree的结构信息：叶节点点数分布和树深度
"""

import numpy as np
import torch
from sklearn.neighbors import KDTree
from typing import List, Tuple, Dict
import sys
sys.path.append('/workspace/PointNeXt')

# 模拟点云数据（类似ModelNet40的点云）
def generate_sample_pointcloud(n_points=1024, distribution='random'):
    """生成示例点云数据"""
    if distribution == 'random':
        # 随机分布
        points = np.random.randn(n_points, 3)
    elif distribution == 'sphere':
        # 球面分布
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0.8, 1.2, n_points)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.column_stack([x, y, z])
    elif distribution == 'clustered':
        # 聚类分布
        centers = np.random.randn(5, 3) * 2
        points = []
        points_per_cluster = n_points // 5
        for center in centers:
            cluster_points = np.random.randn(points_per_cluster, 3) * 0.3 + center
            points.append(cluster_points)
        points = np.vstack(points)
        if len(points) < n_points:
            # 补充剩余点
            extra_points = np.random.randn(n_points - len(points), 3)
            points = np.vstack([points, extra_points])
    
    return points[:n_points]


def analyze_kdtree_structure(points: np.ndarray, leaf_size: int = 32) -> Dict:
    """
    分析KDTree结构信息
    
    Args:
        points: 点云数组 (N, 3)
        leaf_size: 叶节点最大大小
    
    Returns:
        analysis: Dict - 包含树结构分析信息
    """
    N = len(points)
    print(f"\\n=== KDTree结构分析 ===")
    print(f"总点数: {N}")
    print(f"叶节点最大大小: {leaf_size}")
    
    # 构建KDTree
    tree = KDTree(points, leaf_size=leaf_size)
    
    # 递归分析树结构
    leaf_nodes = []
    tree_info = {'max_depth': 0, 'leaf_count': 0, 'leaf_sizes': []}
    
    def recursive_analyze(indices: List[int], depth: int = 0):
        """递归分析树结构"""
        tree_info['max_depth'] = max(tree_info['max_depth'], depth)
        
        if len(indices) <= leaf_size:
            # 叶节点
            tree_info['leaf_count'] += 1
            tree_info['leaf_sizes'].append(len(indices))
            leaf_nodes.append(indices)
            print(f"  叶节点 {tree_info['leaf_count']}: {len(indices)} 个点, 深度 {depth}")
            return
        
        # 内部节点，继续分割
        split_dim = depth % 3
        sorted_indices = sorted(indices, key=lambda i: points[i][split_dim])
        mid = len(sorted_indices) // 2
        
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]
        
        recursive_analyze(left_indices, depth + 1)
        recursive_analyze(right_indices, depth + 1)
    
    # 开始递归分析
    all_indices = list(range(N))
    recursive_analyze(all_indices)
    
    # 统计信息
    leaf_sizes = tree_info['leaf_sizes']
    
    analysis = {
        'total_points': N,
        'leaf_size_limit': leaf_size,
        'num_leaf_nodes': tree_info['leaf_count'],
        'max_depth': tree_info['max_depth'],
        'leaf_sizes': leaf_sizes,
        'avg_leaf_size': np.mean(leaf_sizes),
        'min_leaf_size': np.min(leaf_sizes),
        'max_leaf_size': np.max(leaf_sizes),
        'leaf_size_std': np.std(leaf_sizes),
        'leaf_nodes': leaf_nodes
    }
    
    print(f"\\n=== 统计信息 ===")
    print(f"叶节点数量: {analysis['num_leaf_nodes']}")
    print(f"树最大深度: {analysis['max_depth']}")
    print(f"叶节点平均大小: {analysis['avg_leaf_size']:.2f}")
    print(f"叶节点大小范围: [{analysis['min_leaf_size']}, {analysis['max_leaf_size']}]")
    print(f"叶节点大小标准差: {analysis['leaf_size_std']:.2f}")
    
    # 叶节点大小分布
    unique_sizes, counts = np.unique(leaf_sizes, return_counts=True)
    print(f"\\n=== 叶节点大小分布 ===")
    for size, count in zip(unique_sizes, counts):
        print(f"  {size} 个点: {count} 个叶节点")
    
    return analysis


def compare_different_leaf_sizes(points: np.ndarray, leaf_sizes: List[int] = [16, 32, 64, 128]):
    """比较不同叶节点大小设置的效果"""
    print(f"\\n{'='*50}")
    print(f"比较不同叶节点大小设置")
    print(f"{'='*50}")
    
    results = {}
    for leaf_size in leaf_sizes:
        print(f"\\n--- 叶节点大小: {leaf_size} ---")
        analysis = analyze_kdtree_structure(points, leaf_size)
        results[leaf_size] = analysis
    
    # 汇总比较
    print(f"\\n{'='*20} 汇总比较 {'='*20}")
    print(f"{'叶节点大小':<10} {'叶节点数':<10} {'最大深度':<10} {'平均大小':<10}")
    print(f"{'-'*40}")
    for leaf_size in leaf_sizes:
        analysis = results[leaf_size]
        print(f"{leaf_size:<10} {analysis['num_leaf_nodes']:<10} {analysis['max_depth']:<10} {analysis['avg_leaf_size']:<10.2f}")
    
    return results


def analyze_sampling_efficiency(points: np.ndarray, npoint: int = 512, leaf_size: int = 32):
    """分析采样效率"""
    print(f"\\n{'='*20} 采样效率分析 {'='*20}")
    
    # 分析树结构
    analysis = analyze_kdtree_structure(points, leaf_size)
    
    # 计算采样比例
    sampling_ratio = npoint / len(points)
    expected_points_per_leaf = analysis['avg_leaf_size'] * sampling_ratio
    
    print(f"\\n=== 采样参数 ===")
    print(f"目标采样点数: {npoint}")
    print(f"总点数: {len(points)}")
    print(f"采样比例: {sampling_ratio:.3f}")
    print(f"每个叶节点预期采样点数: {expected_points_per_leaf:.2f}")
    
    # 模拟每个叶节点的采样
    actual_sampled_counts = []
    for leaf_size_actual in analysis['leaf_sizes']:
        leaf_sample_count = max(1, int(leaf_size_actual * sampling_ratio))
        actual_sampled_counts.append(leaf_sample_count)
    
    total_sampled = sum(actual_sampled_counts)
    print(f"实际采样总点数: {total_sampled}")
    print(f"采样效率: {total_sampled/npoint:.3f}")
    
    return {
        'target_npoint': npoint,
        'actual_sampled': total_sampled,
        'efficiency': total_sampled/npoint,
        'per_leaf_samples': actual_sampled_counts
    }


if __name__ == "__main__":
    print("KDTree结构分析工具")
    print("="*50)
    
    # 测试不同类型的点云分布
    distributions = ['random', 'sphere', 'clustered']
    n_points = 1024  # ModelNet40常用点数
    
    for dist in distributions:
        print(f"\\n{'#'*60}")
        print(f"测试点云分布: {dist.upper()}")
        print(f"{'#'*60}")
        
        # 生成点云
        points = generate_sample_pointcloud(n_points, dist)
        
        # 分析默认参数下的KDTree结构
        analysis = analyze_kdtree_structure(points, leaf_size=32)
        
        # 分析采样效率
        sampling_analysis = analyze_sampling_efficiency(points, npoint=512, leaf_size=32)
        
        # 比较不同叶节点大小
        if dist == 'random':  # 只在随机分布下详细比较
            compare_results = compare_different_leaf_sizes(points, [16, 32, 64, 128])
    
    print(f"\\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")