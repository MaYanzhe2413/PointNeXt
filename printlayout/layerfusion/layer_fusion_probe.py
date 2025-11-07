#!/usr/bin/env python3
"""
Layer fusion probe for PointNet++/PointNeXt

Goals (per single forward pass):
 1) Capture per-stage center coordinates and neighborhood indices (idx)
 2) Build inter-layer neighborhood mapping (deep -> shallow)
 3) Compute overlap degree and aggregation factor
 4) Optionally visualize and dump stats for fusion decisions

Notes:
 - We hook QueryAndGroup/KNNGroup to recompute and record idx via their inputs
 - We wrap SetAbstraction.sample_fn to capture FPS/KDTree indices
 - We can project deep neighborhoods down to a target shallow stage by composing idx sets
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.models.layers.group import ball_query
import warnings
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


@dataclass
class StageCapture:
    name: str
    stage_idx: int
    # Centers of this stage (query_xyz): (B, S, 3)
    query_xyz: Optional[torch.Tensor] = None
    # Support points (previous stage points) used for grouping: (B, Nprev, 3)
    support_xyz: Optional[torch.Tensor] = None
    # Group indices into support: (B, S, K)
    idx_group: Optional[torch.Tensor] = None
    # Optional sampling indices used to form centers from previous points: (B, S)
    idx_sample: Optional[torch.Tensor] = None
    # K (nsample) metadata and radius if available
    K: Optional[int] = None
    radius: Optional[float] = None


def _build_sample_inputs(cfg: EasyConfig, batch_size: int, num_points: int, feature_dim: int, device: torch.device):
    feature_last = cfg.model.get("feature_last_dim", False)
    encoder_args = cfg.model.get("encoder_args", {})
    inferred_in_channels = encoder_args.get("in_channels", cfg.model.get("in_channels", feature_dim))
    in_channels = inferred_in_channels if inferred_in_channels is not None else feature_dim

    points = torch.randn(batch_size, num_points, 3, device=device)
    if feature_last:
        features = torch.randn(batch_size, num_points, in_channels, device=device)
    else:
        features = torch.randn(batch_size, in_channels, num_points, device=device)

    return {"pos": points, "x": features}


def _is_grouper_module(m: nn.Module) -> bool:
    name = m.__class__.__name__.lower()
    return ("queryandgroup" in name) or ("knngroup" in name) or ("groupall" in name)


def _is_set_abstraction(m: nn.Module) -> bool:
    # PointNeXt SetAbstraction class name
    return m.__class__.__name__ == 'SetAbstraction'


def capture_neighborhoods(model: nn.Module, sample_input: Dict[str, torch.Tensor]) -> List[StageCapture]:
    stages: List[StageCapture] = []
    hooks = []
    stage_counter = 0

    # Wrap sample_fn in SetAbstraction to capture FPS/KDTree indices
    def wrap_sampling(module: nn.Module, module_name: str):
        if hasattr(module, 'sample_fn'):
            orig_fn = module.sample_fn
            def wrapped_fn(xyz, npoint, *_args, **_kwargs):
                idx = orig_fn(xyz, npoint)
                module._last_sample_idx = idx
                return idx
            module.sample_fn = wrapped_fn

    for name, m in model.named_modules():
        if _is_set_abstraction(m):
            wrap_sampling(m, name)

    # Hook groupers to get idx via inputs
    def make_grouper_hook(name: str):
        nonlocal stage_counter
        def _hook(m: nn.Module, inputs: tuple, output: Any):
            nonlocal stage_counter
            # inputs: (query_xyz, support_xyz, features)
            if not inputs:
                return
            query_xyz = inputs[0]
            support_xyz = inputs[1] if len(inputs) > 1 else inputs[0]
            # Infer params
            K = getattr(m, 'nsample', None)
            radius = getattr(m, 'radius', None)
            # Recompute idx (safe and deterministic)
            idx = None
            try:
                if hasattr(m, 'knn') and m.__class__.__name__.lower() == 'knngroup':
                    # KNNGroup: use internal KNN
                    _, idx_knn = m.knn(support_xyz, query_xyz)
                    idx = idx_knn.int()
                else:
                    # QueryAndGroup or GroupAll via BallQuery
                    if K is None:
                        # GroupAll: one group covering all support points
                        B = support_xyz.shape[0]
                        npoint = query_xyz.shape[1]
                        N = support_xyz.shape[1]
                        idx = torch.arange(N, device=support_xyz.device).view(1, 1, N).repeat(B, npoint, 1)
                    else:
                        idx = ball_query(radius, K, support_xyz.contiguous(), query_xyz.contiguous())
            except Exception:
                pass

            cap = StageCapture(name=name, stage_idx=stage_counter,
                               query_xyz=query_xyz.detach(), support_xyz=support_xyz.detach(),
                               idx_group=idx.detach() if idx is not None else None,
                               K=K, radius=float(radius) if isinstance(radius, (int, float)) else None)

            # Try to attach sample idx if available on nearest SetAbstraction parent
            # Heuristic: walk up parents to find SetAbstraction with _last_sample_idx
            parent_sa = None
            parts = name.split('.')
            for i in range(len(parts), 0, -1):
                parent_name = '.'.join(parts[:i])
                try:
                    parent_module = dict(model.named_modules()).get(parent_name, None)
                    if parent_module is not None and _is_set_abstraction(parent_module):
                        parent_sa = parent_module
                        break
                except Exception:
                    continue
            if parent_sa is not None and hasattr(parent_sa, '_last_sample_idx'):
                cap.idx_sample = getattr(parent_sa, '_last_sample_idx')

            stages.append(cap)
            stage_counter += 1
        return _hook

    for name, m in model.named_modules():
        if _is_grouper_module(m):
            hooks.append(m.register_forward_hook(make_grouper_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    return stages


def compose_mapping(idx_list: List[torch.Tensor], start_stage: int, target_stage: int) -> List[List[int]]:
    """
    Compose neighborhood indices from start_stage down to target_stage (inclusive of target).
    idx_list[s] has shape (B, S_{s}, K_{s}) and indexes into points at stage s-1.
    Returns per-center neighbor sets at target stage for batch 0.
    """
    assert start_stage > target_stage >= 0
    # For simplicity, operate on batch 0
    neighbors_per_center: List[List[int]] = []
    idx_curr = idx_list[start_stage][0]  # (S_s, K_s)
    for j in range(idx_curr.shape[0]):
        curr_set = set(idx_curr[j].tolist())  # indices into stage s-1
        # Walk down s-1 ... target_stage+1
        for s in range(start_stage - 1, target_stage - 1, -1):
            next_set: set = set()
            idx_s = idx_list[s][0]  # (S_s-1, K_s-1) indexes into stage s-2
            for i in curr_set:
                if i < 0 or i >= idx_s.shape[0]:
                    continue
                next_set.update(idx_s[i].tolist())
            curr_set = next_set
        neighbors_per_center.append(sorted(set(curr_set)))
    return neighbors_per_center


def jaccard_overlap(sets: List[List[int]]) -> np.ndarray:
    n = len(sets)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = set(sets[i])
        for j in range(i, n):
            b = set(sets[j])
            inter = len(a & b)
            uni = len(a | b)
            mat[i, j] = mat[j, i] = (inter / uni) if uni > 0 else 0.0
    return mat


def run_probe(cfg_path: str, device: str, num_points: int, feature_dim: int, project_to: int, out_dir: Optional[str], viz: bool = False, viz_k: int = 3) -> Dict:
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    dev = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(cfg.model).to(dev)
    model.eval()

    sample = _build_sample_inputs(cfg, batch_size=1, num_points=num_points, feature_dim=feature_dim, device=dev)
    stages = capture_neighborhoods(model, sample)

    # Gather idx list in call order; filter stages with idx_group
    idx_list = [s.idx_group for s in stages if s.idx_group is not None]
    # Projection sanity
    max_stage = len(idx_list) - 1
    if project_to < 0:
        project_to = 0
    if max_stage <= 0:
        raise RuntimeError("Not enough grouping stages found to compose mappings.")
    if project_to >= max_stage:
        project_to = max_stage - 1

    # Compose mapping from deepest stage down to target
    start_stage = max_stage
    proj_sets = compose_mapping(idx_list, start_stage=start_stage, target_stage=project_to)
    overlaps = jaccard_overlap(proj_sets)
    agg_sizes = [len(s) for s in proj_sets]

    summary = {
        'num_stages': len(stages),
        'group_stages': len(idx_list),
        'start_stage': start_stage,
        'project_to': project_to,
        'agg_sizes_mean': float(np.mean(agg_sizes)),
        'agg_sizes_std': float(np.std(agg_sizes)),
        'overlap_mean': float(np.mean(overlaps)),
        'overlap_std': float(np.std(overlaps)),
        'S_deep': int(idx_list[start_stage].shape[1]) if idx_list[start_stage] is not None else None,
        'K_chain': [int(t.shape[-1]) for t in idx_list],
    }

    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        # Save minimal tensors to npz (CPU)
        to_cpu = lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else None
        np.savez_compressed(
            od / 'stages.npz',
            **{f'query_xyz_{i}': to_cpu(s.query_xyz) for i, s in enumerate(stages)},
            **{f'support_xyz_{i}': to_cpu(s.support_xyz) for i, s in enumerate(stages)},
            **{f'idx_group_{i}': to_cpu(s.idx_group) for i, s in enumerate(stages)},
            **{f'idx_sample_{i}': to_cpu(s.idx_sample) for i, s in enumerate(stages)},
        )
        with open(od / 'summary.txt', 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        # Save inter-layer projection artifacts to CSV
        # 1) proj_sets as ragged CSV (each row: shallow indices for a deep center)
        proj_csv = od / f"proj_deep{start_stage}_to_stage{project_to}.csv"
        with open(proj_csv, 'w') as f:
            f.write(f"shallow_indices_stage_{project_to}\n")
            for row in proj_sets:
                f.write(','.join(map(str, row)) + '\n')
        # 2) edge list (deep_idx, shallow_idx)
        edge_csv = od / f"edge_list_deep{start_stage}_to_stage{project_to}.csv"
        with open(edge_csv, 'w') as f:
            f.write('deep_idx,shallow_idx\n')
            for i, row in enumerate(proj_sets):
                for j in row:
                    f.write(f"{i},{j}\n")
        # 3) overlap matrix and agg sizes
        cols = overlaps.shape[1]
        col_header = ','.join([f'c{j}' for j in range(cols)])
        np.savetxt(od / f"overlap_deep{start_stage}_to_stage{project_to}.csv", overlaps, delimiter=',', header=col_header, comments='')
        np.savetxt(od / f"agg_sizes_deep{start_stage}_to_stage{project_to}.csv", np.array(agg_sizes, dtype=np.int32), delimiter=',', fmt='%d', header='agg_size', comments='')

        # Optional visualization
        if viz:
            if not _HAS_MPL:
                warnings.warn("matplotlib not available; skipping visualization")
            else:
                try:
                    shallow_xyz = stages[project_to].query_xyz[0].detach().cpu().numpy()  # (S_t, 3)
                    deep_xyz = stages[start_stage].query_xyz[0].detach().cpu().numpy()    # (S_s, 3)
                    S_deep = len(proj_sets)
                    pick = np.random.choice(S_deep, size=min(viz_k, S_deep), replace=False)
                    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
                    plt.figure(figsize=(6, 6))
                    # Plot shallow as light grey (use first two dims)
                    plt.scatter(shallow_xyz[:, 0], shallow_xyz[:, 1], s=4, c='#cccccc', label='stage_%d points' % project_to)
                    for i, j in enumerate(pick):
                        idxs = np.array(proj_sets[j], dtype=int)
                        if idxs.size > 0:
                            plt.scatter(shallow_xyz[idxs, 0], shallow_xyz[idxs, 1], s=8, c=colors[i % len(colors)], label=f'deep{j} proj')
                    plt.title(f'Projection of deep stage {start_stage} neighborhoods to stage {project_to}')
                    plt.legend(loc='best', fontsize=8)
                    plt.tight_layout()
                    plt.savefig(od / 'projection_scatter.png', dpi=150)
                    plt.close()
                except Exception as e:
                    warnings.warn(f"Visualization failed: {e}")

    return summary


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr.reshape(arr.shape[1:])
    return arr


def export_npz_to_csv(npz_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(npz_path, allow_pickle=True)
    index_lines: List[str] = []
    for key in sorted(npz.files):
        arr = npz[key]
        # Skip None/object entries
        if arr.dtype == object:
            try:
                item = arr.item()
                if item is None:
                    continue
            except Exception:
                continue
        data = _squeeze_batch(np.array(arr))
        fname = f"{key}.csv"
        fpath = out_dir / fname
        if data.ndim == 1:
            np.savetxt(fpath, data, delimiter=",")
        elif data.ndim == 2:
            np.savetxt(fpath, data, delimiter=",")
        else:
            rows = data.shape[0]
            cols = int(np.prod(data.shape[1:]))
            flat = data.reshape(rows, cols)
            np.savetxt(fpath, flat, delimiter=",")
        index_lines.append(f"{fname}\tshape={tuple(data.shape)}\tdtype={data.dtype}")
    with open(out_dir / "csv_index.txt", "w") as f:
        f.write("\n".join(index_lines))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='Model YAML cfg path')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--num-points', type=int, default=1024)
    ap.add_argument('--feature-dim', type=int, default=3)
    ap.add_argument('--project-to', type=int, default=1, help='Project deep neighborhoods down to this shallow stage index')
    ap.add_argument('--out-dir', type=str, help='Optional output directory to dump npz and summary')
    ap.add_argument('--viz', action='store_true', help='Render a simple 2D scatter visualization to out-dir')
    ap.add_argument('--viz-k', type=int, default=3, help='Number of deep centers to visualize')
    ap.add_argument('--export-csv', action='store_true', help='After saving stages.npz, also export per-key CSV files')
    args = ap.parse_args()

    summary = run_probe(args.cfg, args.device, args.num_points, args.feature_dim, args.project_to, args.out_dir, viz=args.viz, viz_k=args.viz_k)
    print("=== Layer fusion probe summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if args.out_dir and args.export_csv:
        npz_path = Path(args.out_dir) / 'stages.npz'
        export_npz_to_csv(npz_path, Path(args.out_dir))
        print(f"Exported CSV files to: {args.out_dir}")


if __name__ == '__main__':
    main()
