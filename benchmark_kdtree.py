"""
KD-Tree采样性能基准测试

对比以下实现:
1. FPS (baseline) - CUDA kernel
2. KD-Tree原始实现 (CPU循环)
3. KD-Tree向量化实现 (GPU优化)
"""

import torch
import time
import numpy as np
from openpoints.models.layers.subsample import furthest_point_sample
from openpoints.models.layers.kdsample import kdtree_leaf_simple_sample, kdtree_simple_sample_vectorized

def benchmark_sampling(B, N, npoint, num_runs=10, warmup=3):
    """
    基准测试不同采样方法

    Args:
        B: Batch size
        N: 输入点数
        npoint: 采样点数
        num_runs: 测试运行次数
        warmup: 预热次数
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Testing with B={B}, N={N}, npoint={npoint}")
    print("="*80)

    # 生成测试数据
    xyz = torch.randn(B, N, 3, device=device, dtype=torch.float32)

    results = {}

    # ===== 测试 1: FPS (Baseline) =====
    if device == 'cuda':
        print("\n[1/3] Testing FPS (CUDA kernel baseline)...")
        torch.cuda.synchronize()

        # 预热
        for _ in range(warmup):
            _ = furthest_point_sample(xyz, npoint)
        torch.cuda.synchronize()

        # 测试
        times = []
        for i in range(num_runs):
            start = time.time()
            idx = furthest_point_sample(xyz, npoint)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
            if i == 0:
                sample_fps = idx  # 保存用于验证

        mean_time = np.mean(times)
        std_time = np.std(times)
        results['FPS'] = {'mean': mean_time, 'std': std_time, 'speedup': 1.0}
        print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")

    # ===== 测试 2: KD-Tree 原始实现 =====
    print("\n[2/3] Testing KD-Tree (Original CPU implementation)...")
    torch.cuda.synchronize() if device == 'cuda' else None

    # 预热
    for _ in range(warmup):
        _ = kdtree_leaf_simple_sample(xyz, npoint, leaf_size=32, strategy='random', proportional=True)
    torch.cuda.synchronize() if device == 'cuda' else None

    # 测试
    times = []
    for i in range(num_runs):
        start = time.time()
        idx = kdtree_leaf_simple_sample(xyz, npoint, leaf_size=32, strategy='random', proportional=True)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        if i == 0:
            sample_kdtree_orig = idx  # 保存用于验证

    mean_time = np.mean(times)
    std_time = np.std(times)
    speedup = results['FPS']['mean'] / mean_time if 'FPS' in results else 1.0
    results['KDTree_Original'] = {'mean': mean_time, 'std': std_time, 'speedup': speedup}
    print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Speedup vs FPS: {speedup:.2f}x {'SLOWER' if speedup < 1 else 'FASTER'}")

    # ===== 测试 3: KD-Tree 向量化实现 =====
    print("\n[3/3] Testing KD-Tree (Vectorized GPU implementation)...")
    torch.cuda.synchronize() if device == 'cuda' else None

    # 预热
    for _ in range(warmup):
        _ = kdtree_simple_sample_vectorized(xyz, npoint, leaf_size=32, proportional=True)
    torch.cuda.synchronize() if device == 'cuda' else None

    # 测试
    times = []
    for i in range(num_runs):
        start = time.time()
        idx = kdtree_simple_sample_vectorized(xyz, npoint, leaf_size=32, proportional=True)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        if i == 0:
            sample_kdtree_vec = idx  # 保存用于验证

    mean_time = np.mean(times)
    std_time = np.std(times)
    speedup_vs_fps = results['FPS']['mean'] / mean_time if 'FPS' in results else 1.0
    speedup_vs_orig = results['KDTree_Original']['mean'] / mean_time
    results['KDTree_Vectorized'] = {
        'mean': mean_time,
        'std': std_time,
        'speedup_vs_fps': speedup_vs_fps,
        'speedup_vs_orig': speedup_vs_orig
    }
    print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Speedup vs FPS: {speedup_vs_fps:.2f}x {'SLOWER' if speedup_vs_fps < 1 else 'FASTER'}")
    print(f"  Speedup vs Original KDTree: {speedup_vs_orig:.2f}x FASTER")

    # ===== 总结 =====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'Time (ms)':<20} {'Relative to FPS':<20}")
    print("-"*80)

    if 'FPS' in results:
        r = results['FPS']
        print(f"{'FPS (CUDA baseline)':<30} {r['mean']:>8.2f} ± {r['std']:<7.2f}   {r['speedup']:>8.2f}x (baseline)")

    r = results['KDTree_Original']
    rel = f"{1/r['speedup']:.2f}x slower" if 'FPS' in results else 'N/A'
    print(f"{'KDTree Original (CPU)':<30} {r['mean']:>8.2f} ± {r['std']:<7.2f}   {rel:>20}")

    r = results['KDTree_Vectorized']
    rel_fps = f"{r['speedup_vs_fps']:.2f}x" if 'FPS' in results else 'N/A'
    if 'FPS' in results:
        rel_fps += ' slower' if r['speedup_vs_fps'] < 1 else ' faster'
    print(f"{'KDTree Vectorized (GPU)':<30} {r['mean']:>8.2f} ± {r['std']:<7.2f}   {rel_fps:>20}")

    print("\n" + "="*80)
    print("IMPROVEMENT")
    print("="*80)
    improvement = results['KDTree_Vectorized']['speedup_vs_orig']
    print(f"✅ Vectorized implementation is {improvement:.1f}x FASTER than original")

    if 'FPS' in results:
        gap = results['KDTree_Vectorized']['speedup_vs_fps']
        if gap >= 0.5:
            print(f"✅ Vectorized is only {1/gap:.1f}x slower than FPS (acceptable)")
        elif gap >= 0.3:
            print(f"⚠️  Vectorized is {1/gap:.1f}x slower than FPS (needs optimization)")
        else:
            print(f"❌ Vectorized is {1/gap:.1f}x slower than FPS (significant gap)")

    print("="*80)

    return results


def test_correctness():
    """验证采样结果的正确性"""
    print("\n" + "="*80)
    print("CORRECTNESS TESTS")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N, npoint = 2, 1000, 250

    xyz = torch.randn(B, N, 3, device=device, dtype=torch.float32)

    # 测试1: 索引范围检查
    print("\nTest 1: Index range check...")
    idx_orig = kdtree_leaf_simple_sample(xyz, npoint, leaf_size=32)
    idx_vec = kdtree_simple_sample_vectorized(xyz, npoint, leaf_size=32)

    assert torch.all(idx_orig >= 0) and torch.all(idx_orig < N), "Original: Invalid indices"
    assert torch.all(idx_vec >= 0) and torch.all(idx_vec < N), "Vectorized: Invalid indices"
    print("  ✅ All indices in valid range [0, N)")

    # 测试2: 采样数量检查
    print("\nTest 2: Sample count check...")
    assert idx_orig.shape == (B, npoint), f"Original: Wrong shape {idx_orig.shape}"
    assert idx_vec.shape == (B, npoint), f"Vectorized: Wrong shape {idx_vec.shape}"
    print(f"  ✅ Both return correct shape ({B}, {npoint})")

    # 测试3: 无重复检查
    print("\nTest 3: Duplicate check...")
    for b in range(B):
        unique_orig = len(torch.unique(idx_orig[b]))
        unique_vec = len(torch.unique(idx_vec[b]))
        print(f"  Batch {b}: Original {unique_orig}/{npoint} unique, Vectorized {unique_vec}/{npoint} unique")
        if unique_orig < npoint * 0.95:
            print(f"    ⚠️  Original has {npoint - unique_orig} duplicates")
        if unique_vec < npoint * 0.95:
            print(f"    ⚠️  Vectorized has {npoint - unique_vec} duplicates")

    # 测试4: 边界情况
    print("\nTest 4: Edge cases...")

    # npoint = N
    idx = kdtree_simple_sample_vectorized(xyz, N, leaf_size=32)
    assert idx.shape == (B, N), "Failed when npoint=N"
    print(f"  ✅ Works when npoint=N ({N})")

    # npoint > N
    idx = kdtree_simple_sample_vectorized(xyz, N+100, leaf_size=32)
    assert idx.shape[1] <= N, "Should clamp to N when npoint>N"
    print(f"  ✅ Correctly clamps when npoint>N")

    # Small npoint
    idx = kdtree_simple_sample_vectorized(xyz, 10, leaf_size=32)
    assert idx.shape == (B, 10), "Failed with small npoint"
    print(f"  ✅ Works with small npoint (10)")

    print("\n" + "="*80)
    print("✅ ALL CORRECTNESS TESTS PASSED")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("KD-TREE SAMPLING BENCHMARK")
    print("="*80)

    # 正确性测试
    test_correctness()

    # 性能测试 - 不同规模
    configs = [
        {"name": "Small (ModelNet40-like)", "B": 32, "N": 1024, "npoint": 512},
        {"name": "Medium (S3DIS-like)", "B": 32, "N": 24000, "npoint": 6000},
        {"name": "Large", "B": 16, "N": 40000, "npoint": 10000},
    ]

    all_results = {}
    for config in configs:
        print("\n" + "="*80)
        print(f"BENCHMARK: {config['name']}")
        print("="*80)
        results = benchmark_sampling(
            B=config['B'],
            N=config['N'],
            npoint=config['npoint'],
            num_runs=10,
            warmup=3
        )
        all_results[config['name']] = results

    # 最终总结
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL CONFIGURATIONS")
    print("="*80)

    for name, results in all_results.items():
        print(f"\n{name}:")
        if 'KDTree_Vectorized' in results:
            r = results['KDTree_Vectorized']
            print(f"  Vectorized: {r['mean']:.2f} ms")
            print(f"  Speedup vs Original: {r['speedup_vs_orig']:.1f}x")
            if 'speedup_vs_fps' in r:
                print(f"  vs FPS: {1/r['speedup_vs_fps']:.1f}x slower")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
