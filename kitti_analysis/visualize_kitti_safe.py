#!/usr/bin/env python3
"""
简化版KITTI点云可视化
完全不依赖matplotlib，纯文本输出
专门解决GLIBCXX版本问题
"""
import os
import glob
import struct
import math

def load_kitti_bin_safe(file_path):
    """安全加载KITTI .bin文件"""
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
        print(f"加载失败: {e}")
        return []

def analyze_kitti_data():
    """分析KITTI数据"""
    print("=== 安全模式KITTI数据分析 ===")
    
    # 数据路径
    data_path = "/workspace/home/mayz/network/data/kitti/sequences/00/velodyne"
    
    if not os.path.exists(data_path):
        print(f"❌ 路径不存在: {data_path}")
        return
    
    # 获取文件列表
    files = sorted(glob.glob(os.path.join(data_path, "*.bin")))
    if not files:
        print("❌ 未找到.bin文件")
        return
    
    print(f"✅ 找到 {len(files)} 个文件")
    
    # 分析第一个文件
    print(f"\n分析文件: {os.path.basename(files[0])}")
    points = load_kitti_bin_safe(files[0])
    
    if not points:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 加载 {len(points)} 个点")
    
    # 提取坐标
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    intensities = [p[3] for p in points]
    
    # 基本统计
    print(f"\n=== 基本统计 ===")
    print(f"X轴: 范围[{min(x_coords):.1f}, {max(x_coords):.1f}]m, 均值={sum(x_coords)/len(x_coords):.1f}m")
    print(f"Y轴: 范围[{min(y_coords):.1f}, {max(y_coords):.1f}]m, 均值={sum(y_coords)/len(y_coords):.1f}m")
    print(f"Z轴: 范围[{min(z_coords):.1f}, {max(z_coords):.1f}]m, 均值={sum(z_coords)/len(z_coords):.1f}m")
    print(f"强度: 范围[{min(intensities):.1f}, {max(intensities):.1f}], 均值={sum(intensities)/len(intensities):.1f}")
    
    # 距离分析
    distances = [math.sqrt(x*x + y*y) for x, y in zip(x_coords, y_coords)]
    print(f"距离: 范围[{min(distances):.1f}, {max(distances):.1f}]m, 均值={sum(distances)/len(distances):.1f}m")
    
    # 过滤近距离点进行可视化
    nearby_points = [(x, y, z, i) for x, y, z, i in points 
                     if -20 <= x <= 20 and -20 <= y <= 20 and -3 <= z <= 5]
    
    print(f"\n过滤后(±20m范围): {len(nearby_points)} 个点")
    
    if nearby_points:
        # 下采样
        step = max(1, len(nearby_points) // 500)  # 最多500个点
        sampled = nearby_points[::step]
        print(f"下采样后: {len(sampled)} 个点")
        
        # 创建简单的俯视图
        create_simple_top_view(sampled)
        
        # 高度分布
        z_values = [p[2] for p in sampled]
        create_simple_histogram(z_values, "高度分布", "米")
    
    # 如果有多个文件，比较前两个
    if len(files) >= 2:
        print(f"\n{'='*50}")
        print(f"比较前两帧")
        
        points2 = load_kitti_bin_safe(files[1])
        if points2:
            compare_two_frames(points, points2)

def create_simple_top_view(points):
    """创建简单俯视图"""
    print(f"\n=== 俯视图 (X-Y平面) ===")
    
    if not points:
        print("无数据")
        return
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 网格设置
    width, height = 30, 15
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    if x_max == x_min or y_max == y_min:
        print("数据范围太小，无法绘制")
        return
    
    # 创建网格
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # 映射点到网格
    for x, y, z, intensity in points:
        grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
        
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        
        grid[height - 1 - grid_y][grid_x] += 1
    
    # 显示
    max_count = max(max(row) for row in grid) if any(any(row) for row in grid) else 1
    
    print(f"Y({y_max:.0f}) ↑")
    for row in grid:
        line = ""
        for count in row:
            if count == 0:
                line += " "
            elif count <= max_count * 0.3:
                line += "·"
            elif count <= max_count * 0.6:
                line += "o"
            else:
                line += "#"
        print(line)
    
    print("+" + "-" * (width - 2) + f"+ → X({x_max:.0f})")
    print(f"({x_min:.0f})" + " " * (width - 8) + f"({y_min:.0f})")

def create_simple_histogram(data, title, unit):
    """创建简单直方图"""
    print(f"\n=== {title} ===")
    
    if not data:
        print("无数据")
        return
    
    data_min, data_max = min(data), max(data)
    if data_max == data_min:
        print(f"所有值相同: {data_min:.2f} {unit}")
        return
    
    # 分组
    bins = 8
    bin_width = (data_max - data_min) / bins
    counts = [0] * bins
    
    for value in data:
        bin_idx = int((value - data_min) / bin_width)
        bin_idx = max(0, min(bins - 1, bin_idx))
        counts[bin_idx] += 1
    
    # 显示
    max_count = max(counts)
    for i in range(bins):
        bin_start = data_min + i * bin_width
        bin_end = data_min + (i + 1) * bin_width
        count = counts[i]
        
        bar_len = int((count / max_count) * 20) if max_count > 0 else 0
        bar = "█" * bar_len
        
        print(f"[{bin_start:5.1f}-{bin_end:5.1f}] {unit}: {bar} ({count})")

def compare_two_frames(frame1, frame2):
    """比较两帧数据"""
    print(f"Frame 1: {len(frame1)} 点")
    print(f"Frame 2: {len(frame2)} 点")
    
    # 计算质心
    def centroid(points):
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        z = sum(p[2] for p in points) / len(points)
        return x, y, z
    
    c1 = centroid(frame1)
    c2 = centroid(frame2)
    
    print(f"Frame 1 质心: ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})")
    print(f"Frame 2 质心: ({c2[0]:.2f}, {c2[1]:.2f}, {c2[2]:.2f})")
    
    # 计算偏移
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1] 
    dz = c2[2] - c1[2]
    
    print(f"质心偏移: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
    print(f"水平移动: {math.sqrt(dx*dx + dy*dy):.2f}m")
    print(f"垂直移动: {abs(dz):.2f}m")

if __name__ == "__main__":
    try:
        analyze_kitti_data()
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
