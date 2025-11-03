#!/usr/bin/env python3
"""
KITTI点云数据可视化脚本
使用matplotlib进行2D和3D可视化
"""
import os
import sys
import glob
import struct
import math

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
            x, y, z, intensity = struct.unpack('<ffff', data[offset:offset+16])
            points.append((x, y, z, intensity))
        
        return points
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []

def filter_points_by_range(points, x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 10)):
    """过滤点云范围，便于可视化"""
    filtered = []
    for x, y, z, intensity in points:
        if (x_range[0] <= x <= x_range[1] and 
            y_range[0] <= y <= y_range[1] and 
            z_range[0] <= z <= z_range[1]):
            filtered.append((x, y, z, intensity))
    return filtered

def downsample_points(points, step=10):
    """下采样点云，减少显示点数"""
    return points[::step]

def create_kitti_visualization():
    """创建KITTI数据可视化"""
    
    # 数据路径
    kitti_data_dir = "/workspace/home/mayz/network/data/kitti"
    velodyne_dir = os.path.join(kitti_data_dir, "sequences", "00", "velodyne")
    
    if not os.path.exists(velodyne_dir):
        print(f"❌ 数据目录不存在: {velodyne_dir}")
        return
    
    # 获取.bin文件
    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    if not bin_files:
        print(f"❌ 未找到.bin文件")
        return
    
    print(f"✅ 找到 {len(bin_files)} 个点云文件")
    
    # 尝试导入matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        HAS_MATPLOTLIB = True
        print("✅ matplotlib可用")
    except ImportError:
        HAS_MATPLOTLIB = False
        print("❌ matplotlib不可用，将生成纯文本可视化")
    
    # 加载第一帧数据
    frame_0 = load_kitti_bin_simple(bin_files[0])
    print(f"加载第一帧: {len(frame_0)} 个点")
    
    # 过滤和下采样
    frame_0_filtered = filter_points_by_range(frame_0, x_range=(-30, 30), y_range=(-30, 30), z_range=(-3, 5))
    frame_0_sampled = downsample_points(frame_0_filtered, step=20)
    print(f"过滤和下采样后: {len(frame_0_sampled)} 个点")
    
    if HAS_MATPLOTLIB:
        # 创建matplotlib可视化
        create_matplotlib_plots(frame_0_sampled, bin_files)
    else:
        # 创建文本可视化
        create_text_visualization(frame_0_sampled)
    
    # 如果有多帧，比较前两帧
    if len(bin_files) >= 2:
        frame_1 = load_kitti_bin_simple(bin_files[1])
        frame_1_filtered = filter_points_by_range(frame_1, x_range=(-30, 30), y_range=(-30, 30), z_range=(-3, 5))
        frame_1_sampled = downsample_points(frame_1_filtered, step=20)
        
        if HAS_MATPLOTLIB:
            create_comparison_plot(frame_0_sampled, frame_1_sampled)
        else:
            print(f"\n加载第二帧: {len(frame_1_sampled)} 个点")
            compare_frames_text(frame_0_sampled, frame_1_sampled)

def create_matplotlib_plots(points, bin_files):
    """使用matplotlib创建可视化图"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 提取坐标
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    intensities = [p[3] for p in points]
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D散点图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(x_coords, y_coords, z_coords, c=intensities, 
                         cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Point Cloud (Colored by Intensity)')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # 2. 俯视图 (X-Y)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(x_coords, y_coords, c=intensities, cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 侧视图 (X-Z)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(x_coords, z_coords, c=intensities, cmap='viridis', s=1, alpha=0.6)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z Plane)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 强度分布直方图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(intensities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Count')
    ax4.set_title('Intensity Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. 高度分布直方图
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(z_coords, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_xlabel('Height Z (m)')
    ax5.set_ylabel('Count')
    ax5.set_title('Height Distribution')
    ax5.grid(True, alpha=0.3)
    
    # 6. 距离分布
    distances = [math.sqrt(x*x + y*y) for x, y in zip(x_coords, y_coords)]
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(distances, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Distance from Origin (m)')
    ax6.set_ylabel('Count')
    ax6.set_title('Distance Distribution')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = '/workspace/PointNeXt/kitti_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"🎨 可视化图已保存到: {output_path}")
    
    plt.close()

def create_comparison_plot(frame_0, frame_1):
    """创建两帧对比图"""
    import matplotlib.pyplot as plt
    
    # 提取坐标
    x0 = [p[0] for p in frame_0]
    y0 = [p[1] for p in frame_0]
    z0 = [p[2] for p in frame_0]
    
    x1 = [p[0] for p in frame_1]
    y1 = [p[1] for p in frame_1]
    z1 = [p[2] for p in frame_1]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Frame 0 俯视图
    axes[0, 0].scatter(x0, y0, c='blue', s=1, alpha=0.6, label='Frame 0')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Frame 0 - Top View')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Frame 1 俯视图
    axes[0, 1].scatter(x1, y1, c='red', s=1, alpha=0.6, label='Frame 1')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('Frame 1 - Top View')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # 重叠俯视图
    axes[1, 0].scatter(x0, y0, c='blue', s=1, alpha=0.4, label='Frame 0')
    axes[1, 0].scatter(x1, y1, c='red', s=1, alpha=0.4, label='Frame 1')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Overlapped View')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # 高度对比
    axes[1, 1].hist(z0, bins=30, alpha=0.5, color='blue', label='Frame 0', density=True)
    axes[1, 1].hist(z1, bins=30, alpha=0.5, color='red', label='Frame 1', density=True)
    axes[1, 1].set_xlabel('Height Z (m)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Height Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图
    output_path = '/workspace/PointNeXt/kitti_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"🎨 对比图已保存到: {output_path}")
    
    plt.close()

def create_text_visualization(points):
    """创建文本形式的可视化"""
    print("\n" + "="*60)
    print("=== KITTI点云数据文本可视化 ===")
    
    # 基本统计
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    intensities = [p[3] for p in points]
    
    print(f"点云统计:")
    print(f"  总点数: {len(points)}")
    print(f"  X范围: [{min(x_coords):.2f}, {max(x_coords):.2f}] (跨度: {max(x_coords)-min(x_coords):.2f}m)")
    print(f"  Y范围: [{min(y_coords):.2f}, {max(y_coords):.2f}] (跨度: {max(y_coords)-min(y_coords):.2f}m)")
    print(f"  Z范围: [{min(z_coords):.2f}, {max(z_coords):.2f}] (跨度: {max(z_coords)-min(z_coords):.2f}m)")
    print(f"  强度范围: [{min(intensities):.2f}, {max(intensities):.2f}]")
    
    # 计算均值和标准差
    x_mean = sum(x_coords) / len(x_coords)
    y_mean = sum(y_coords) / len(y_coords)
    z_mean = sum(z_coords) / len(z_coords)
    intensity_mean = sum(intensities) / len(intensities)
    
    print(f"\n中心位置:")
    print(f"  X均值: {x_mean:.2f}m")
    print(f"  Y均值: {y_mean:.2f}m") 
    print(f"  Z均值: {z_mean:.2f}m")
    print(f"  强度均值: {intensity_mean:.2f}")
    
    # 距离分析
    distances = [math.sqrt(x*x + y*y) for x, y in zip(x_coords, y_coords)]
    dist_mean = sum(distances) / len(distances)
    print(f"  平均距离: {dist_mean:.2f}m")
    
    # 简单的俯视图ASCII艺术
    print(f"\n=== 俯视图 (X-Y平面) ===")
    create_ascii_plot(x_coords, y_coords, 'X', 'Y')
    
    # 侧视图
    print(f"\n=== 侧视图 (X-Z平面) ===")
    create_ascii_plot(x_coords, z_coords, 'X', 'Z')
    
    # 强度分布
    print(f"\n=== 强度分布 ===")
    create_histogram_text(intensities, "强度", 10)
    
    # 高度分布
    print(f"\n=== 高度分布 ===")
    create_histogram_text(z_coords, "高度(m)", 10)

def create_ascii_plot(x_data, y_data, x_label, y_label):
    """创建ASCII散点图"""
    if not x_data or not y_data:
        print("无数据")
        return
    
    # 网格大小
    width, height = 40, 20
    
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)
    
    # 避免除零
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    
    # 创建网格
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # 将点映射到网格
    for x, y in zip(x_data, y_data):
        grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
        
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        
        grid[height - 1 - grid_y][grid_x] += 1  # 翻转Y轴
    
    # 显示网格
    max_count = max(max(row) for row in grid) if any(any(row) for row in grid) else 1
    
    print(f"{y_label} ↑")
    for row in grid:
        line = ""
        for count in row:
            if count == 0:
                line += " "
            elif count < max_count * 0.2:
                line += "."
            elif count < max_count * 0.5:
                line += "o"
            elif count < max_count * 0.8:
                line += "O"
            else:
                line += "#"
        print(line)
    
    # 添加X轴标签
    x_axis = "+" + "-" * (width - 2) + "+"
    print(x_axis + f" → {x_label}")
    print(f"{x_min:.1f}" + " " * (width - 10) + f"{x_max:.1f}")
    
    print(f"\n密度图例: 空格=无点, .=稀疏, o=中等, O=密集, #=非常密集")

def create_histogram_text(data, label, bins=10):
    """创建文本直方图"""
    if not data:
        print("无数据")
        return
    
    data_min, data_max = min(data), max(data)
    if data_max == data_min:
        print(f"所有值相同: {data_min:.2f}")
        return
    
    # 计算直方图
    bin_width = (data_max - data_min) / bins
    hist_counts = [0] * bins
    
    for value in data:
        bin_idx = int((value - data_min) / bin_width)
        bin_idx = max(0, min(bins - 1, bin_idx))
        hist_counts[bin_idx] += 1
    
    # 显示直方图
    max_count = max(hist_counts)
    bar_width = 50
    
    for i in range(bins):
        bin_start = data_min + i * bin_width
        bin_end = data_min + (i + 1) * bin_width
        count = hist_counts[i]
        
        # 计算条形长度
        if max_count > 0:
            bar_len = int((count / max_count) * bar_width)
        else:
            bar_len = 0
        
        bar = "#" * bar_len
        
        print(f"[{bin_start:6.1f}-{bin_end:6.1f}]: {bar} ({count})")
    
    # 统计信息
    mean_val = sum(data) / len(data)
    sorted_data = sorted(data)
    median_val = sorted_data[len(sorted_data) // 2]
    
    print(f"统计: 均值={mean_val:.2f}, 中位数={median_val:.2f}, 范围=[{data_min:.2f}, {data_max:.2f}]")

def compare_frames_text(frame_0, frame_1):
    """文本形式的帧间对比"""
    print(f"\n=== 帧间对比分析 ===")
    
    # 计算质心
    def compute_centroid(points):
        x_mean = sum(p[0] for p in points) / len(points)
        y_mean = sum(p[1] for p in points) / len(points)
        z_mean = sum(p[2] for p in points) / len(points)
        return x_mean, y_mean, z_mean
    
    c0 = compute_centroid(frame_0)
    c1 = compute_centroid(frame_1)
    
    print(f"Frame 0 质心: ({c0[0]:.2f}, {c0[1]:.2f}, {c0[2]:.2f})")
    print(f"Frame 1 质心: ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})")
    
    # 计算偏移
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    dz = c1[2] - c0[2]
    
    print(f"质心偏移: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
    print(f"水平移动距离: {math.sqrt(dx*dx + dy*dy):.2f}m")
    print(f"垂直移动距离: {abs(dz):.2f}m")

def main():
    """主函数"""
    print("=== KITTI点云数据可视化 ===")
    create_kitti_visualization()

if __name__ == "__main__":
    main()
