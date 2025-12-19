#!/usr/bin/env python3
"""
Generate a point_config.py-style OrderedDict by instantiating the model from YAML and
observing real shapes via a dummy forward pass.

- Supports PointNeXt and PointNet++ classification cfgs in the PointNeXt repo.
- Detects grouping stages by QueryAndGroup/GroupAll modules.
- Records per-stage: points_out (S), group width (K), and Conv2d channel stacks.
- Builds a PNN-style OrderedDict with grouper, Conv2D/BN/ReLU, MaxPool, and Linear head.

This avoids heuristic drift from cfg-only parsing and stays faithful to actual module behavior.
"""
import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

# Ensure PointNeXt repo is importable
ROOT = Path(__file__).resolve().parents[3] / 'PointNeXt'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg


class StageRecord:
    def __init__(self) -> None:
        self.points_out: Optional[int] = None  # S
        self.group_width: Optional[int] = None  # K
        self.conv_channels: List[int] = []  # sequence of Cout
        self.radius: Optional[float] = None  # grouping radius if available


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
    return ("queryandgroup" in name) or ("groupall" in name) or ("ballquery" in name and hasattr(m, 'group_all') and m.group_all)


def _collect_stages(model: nn.Module, sample_input: Dict[str, torch.Tensor]) -> Tuple[List[StageRecord], List[Tuple[str, Tuple[int, int]]]]:
    stages: List[StageRecord] = []
    conv2d_locs: List[Tuple[str, Tuple[int, int]]] = []  # (name, (S, K)) at time of first conv in stage

    hooks = []
    active_stage_idx = -1
    last_S, last_K = None, None

    def register_hooks():
        nonlocal active_stage_idx, last_S, last_K

        def grouper_hook(_m: nn.Module, inputs: tuple, output: Any):
            nonlocal active_stage_idx, last_S, last_K
            # Expect output as grouped features: (B, Cg, S, K)
            # Fallback to infer from any tensor in output
            def find_tensor(o):
                if isinstance(o, torch.Tensor):
                    return o
                if isinstance(o, (list, tuple)):
                    for it in o:
                        t = find_tensor(it)
                        if t is not None:
                            return t
                if isinstance(o, dict):
                    for it in o.values():
                        t = find_tensor(it)
                        if t is not None:
                            return t
                return None

            t = find_tensor(output)
            S = None
            K = None
            if isinstance(t, torch.Tensor) and t.dim() >= 4:
                # Guess layout (B, Cg, S, K) or (B, Cg, K, S)
                dims = list(t.shape)
                # choose the two spatial dims as the last two dims
                spatial = dims[-2:]
                # Heuristic: larger is K if not GroupAll, but both acceptable; we will overwrite later if needed
                S, K = spatial[0], spatial[1]
            # Start a new stage
            stages.append(StageRecord())
            active_stage_idx = len(stages) - 1
            last_S, last_K = S, K
            # Try to fetch metadata from module
            try:
                if hasattr(_m, 'nsample') and stages[active_stage_idx].group_width is None:
                    stages[active_stage_idx].group_width = int(getattr(_m, 'nsample'))
            except Exception:
                pass
            try:
                if hasattr(_m, 'radius'):
                    r = getattr(_m, 'radius')
                    if isinstance(r, (int, float)):
                        stages[active_stage_idx].radius = float(r)
            except Exception:
                pass

        def conv2d_hook(name: str):
            def _hook(_m: nn.Module, inputs: tuple, output: Any):
                nonlocal active_stage_idx, last_S, last_K
                if active_stage_idx < 0:
                    return
                # Determine Cout from module
                if hasattr(_m, 'out_channels'):
                    Cout = int(_m.out_channels)
                else:
                    # Try weight
                    w = getattr(_m, 'weight', None)
                    Cout = int(w.shape[0]) if isinstance(w, torch.Tensor) else 0
                # Infer (S, K) from input tensor to conv2d: shape (B, C, H, W)
                xin = inputs[0] if inputs else None
                if isinstance(xin, torch.Tensor) and xin.dim() == 4:
                    H, W = int(xin.shape[-2]), int(xin.shape[-1])
                    last_S, last_K = H, W
                # Record channel and first conv spatial dims for this stage
                stages[active_stage_idx].conv_channels.append(Cout)
                if len([c for c in stages[active_stage_idx].conv_channels]) == 1:
                    conv2d_locs.append((name, (last_S or 0, last_K or 0)))
            return _hook

        for name, m in model.named_modules():
            if _is_grouper_module(m):
                hooks.append(m.register_forward_hook(grouper_hook))
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv2d_hook(name)))

    register_hooks()
    with torch.no_grad():
        model(sample_input)
    for h in hooks:
        h.remove()

    # Post-process S and K per stage using conv locs
    for idx, (name, (S, K)) in enumerate(conv2d_locs):
        if idx < len(stages):
            stages[idx].points_out = S
            stages[idx].group_width = K

    # Fallback: fill missing S/K from next-known or assume 1
    for st in stages:
        if st.points_out is None:
            st.points_out = 1
        if st.group_width is None:
            st.group_width = 1

    return stages, conv2d_locs


def _derive_points_list(N0: int, stages: List[StageRecord]) -> List[int]:
    pts = [N0]
    for st in stages:
        pts.append(int(st.points_out or 1))
    return pts


def _find_num_points_in_cfg(obj: Any) -> Optional[int]:
    # Recursively search for keys commonly used for point count
    KEYS = {"num_points", "npoints", "points"}
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in KEYS and isinstance(v, int) and v > 0:
                    return int(v)
                res = _find_num_points_in_cfg(v)
                if isinstance(res, int) and res > 0:
                    return res
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                res = _find_num_points_in_cfg(it)
                if isinstance(res, int) and res > 0:
                    return res
    except Exception:
        pass
    return None


def _infer_num_points_from_data(cfg: EasyConfig, split: str = 'val') -> Optional[int]:
    try:
        dataset_cfg = cfg.get('dataset', None)
        dataloader_cfg = cfg.get('dataloader', {})
        datatransforms_cfg = cfg.get('datatransforms', None)
        if dataset_cfg is None:
            return None
        loader = build_dataloader_from_cfg(
            batch_size=1,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            datatransforms_cfg=datatransforms_cfg,
            split=split,
            distributed=False,
        )
        # Prefer dataset attribute
        if hasattr(loader.dataset, 'num_points'):
            n = getattr(loader.dataset, 'num_points')
            if isinstance(n, int) and n > 0:
                return int(n)
        # Fallback: peek one batch
        it = iter(loader)
        batch = next(it)
        pos = batch.get('pos', None)
        if isinstance(pos, torch.Tensor):
            if pos.dim() == 3:  # (B, N, 3)
                return int(pos.shape[1])
            if pos.dim() == 2:  # (N, 3), B likely 1
                return int(pos.shape[0])
        return None
    except Exception:
        return None


def generate_from_model(cfg_path: str, num_points: int, th: int = 64, device: str = 'cpu', auto_num_points: bool = False, infer_from_data: bool = False) -> str:
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    model_cfg = cfg.get('model')
    if model_cfg is None:
        raise KeyError("'model' section not found in cfg")

    dev = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(model_cfg).to(dev)
    model.eval()

    # Optionally infer num_points from cfg (dataset/transforms) or dataloader if available
    inferred_n = None
    if auto_num_points:
        inferred_n = _find_num_points_in_cfg(cfg)
    if inferred_n is None and infer_from_data:
        inferred_n = _infer_num_points_from_data(cfg, split='val') or _infer_num_points_from_data(cfg, split='train')
    N0 = int(inferred_n or num_points)

    sample_input = _build_sample_inputs(cfg, batch_size=1, num_points=N0, feature_dim=3, device=dev)

    stages, _ = _collect_stages(model, sample_input)

    # Pull radius and sampler from cfg where available (metadata)
    enc = model_cfg.get('encoder_args', {})
    radius_list = enc.get('radius', [])
    num_samples = enc.get('num_samples', [])
    strides = enc.get('strides', [])
    sampler = str(enc.get('sampler', 'fps')).lower()
    tree_flag = (sampler == 'kdtree')

    points = _derive_points_list(N0, stages)

    lines: List[str] = []
    # Header with environment activation hint
    lines.append('#!/usr/bin/env python3')
    lines.append('# NOTE: Activate conda env before use:')
    lines.append('#   conda activate pointnext')
    lines.append('# Auto-generated by gen_point_config_from_model.py from: ' + os.path.basename(cfg_path))
    lines.append('from collections import OrderedDict')
    lines.append('import math')
    lines.append('')
    func_name = 'point_config_from_model'
    lines.append(f'def {func_name}(batch_size=1):')
    lines.append(f'    points = {points}')
    default_g = stages[0].group_width if stages else 32
    lines.append(f'    grouped_points_default = {default_g}')
    lines.append(f'    th = {th}')
    lines.append('    cfg = OrderedDict([')

    Cin_prev = enc.get('in_channels', model_cfg.get('in_channels', 3))
    feature_type = enc.get('aggr_args', {}).get('feature_type', 'dp_fj')

    for i, st in enumerate(stages):
        S = int(st.points_out)
        K = int(st.group_width)
        N_in = f'points[{i}]'
        N_out = f'points[{i+1}]'
        # Choose radius: prefer observed from module
        radius = stages[i].radius if hasattr(stages[i], 'radius') else None
        # Fallback to cfg-provided list if it is indexable
        if radius is None:
            try:
                radius = radius_list[i]
            except Exception:
                radius = None
        is_group_all = (radius is None and (i < len(num_samples) and num_samples[i] is None)) or S == 1
        if is_group_all:
            # Group all input points into 1; group width = N_in
            lines.append(
                f"        ('grouper{i+1}', [{N_in}, {N_out}, batch_size, {N_in}, None, {tree_flag}, th]),  # GroupAll"
            )
            group_w = N_in
        else:
            group_w = K
            lines.append(
                f"        ('grouper{i+1}', [{N_in}, {N_out}, batch_size, {group_w}, {radius if radius is not None else 'None'}, {tree_flag}, th]),"
            )
        # Conv2d stack
        Cin_group = (Cin_prev + 3) if 'dp' in str(feature_type) else Cin_prev
        prev_C = Cin_group
        for li, Cout in enumerate(st.conv_channels, start=1):
            lines.append(
                f"        ('conv2d{i+1}-{li}', [{N_out}, {group_w}, {prev_C}, {Cout}, 1, 0, 1, batch_size]),"
            )
            lines.append(f"        ('bn{i+1}-{li}', []),")
            lines.append(f"        ('relu{i+1}-{li}', []),")
            prev_C = Cout
        Cin_prev = prev_C
        lines.append(
            f"        ('maxpool{i+1}', [{N_out}, {group_w}, {Cin_prev}, 1, 0, 1, batch_size]),"
        )

    # Head from cfg (more reliable than trying to infer via mixed modules)
    cls_args = model_cfg.get('cls_args', cfg.get('cls_args', {}))
    head_mlps = cls_args.get('mlps', [512, 256])
    num_classes = int(cls_args.get('num_classes', 40))
    in_dim = Cin_prev
    for idx, out_dim in enumerate(head_mlps, start=1):
        lines.append(f"        ('linear{idx}-1', [{in_dim}, {out_dim}, 1, batch_size]),")
        lines.append(f"        ('bn{idx+5}-1', []),")
        lines.append(f"        ('relu{idx+5}-1', []),")
        lines.append(f"        ('dropout{idx}', []),")
        in_dim = out_dim
    lines.append(f"        ('linear{len(head_mlps)+1}-1', [{in_dim}, {num_classes}, 1, batch_size]),")

    lines.append('    ])')
    lines.append('    return cfg')

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='YAML cfg path')
    ap.add_argument('--num-points', type=int, default=1024)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--out', required=True)
    ap.add_argument('--th', type=int, default=64)
    ap.add_argument('--auto-num-points', action='store_true', help='Try to infer input points from cfg (dataset/transforms)')
    ap.add_argument('--infer-from-data', action='store_true', help='Instantiate dataloader and peek a batch to infer input points')
    args = ap.parse_args()

    content = generate_from_model(args.cfg, args.num_points, th=args.th, device=args.device, auto_num_points=args.auto_num_points, infer_from_data=args.infer_from_data)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(content)
    print(f'Wrote: {args.out}')


if __name__ == '__main__':
    main()
