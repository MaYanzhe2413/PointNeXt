#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from easydict import EasyDict as edict
from openpoints.dataset import build_dataloader_from_cfg


def load_shape_names(modelnet_dir: Path) -> List[str]:
    names_file = modelnet_dir / 'shape_names.txt'
    if names_file.exists():
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    # Fallback: try upper folder
    up = modelnet_dir.parent / 'modelnet40_ply_hdf5_2048' / 'shape_names.txt'
    if up.exists():
        with open(up, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    return []


def main():
    ap = argparse.ArgumentParser(description='Dump a few samples from ModelNet40 dataloader for simulator validation.')
    ap.add_argument('--data-dir', type=str, required=True, help='Path to modelnet40_ply_hdf5_2048 dir (contains ply_data_*.h5)')
    ap.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--num-points', type=int, default=1024)
    ap.add_argument('--class', dest='class_name', type=str, default='airplane')
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--out', type=str, default=str(ROOT / 'data' / 'simu' / 'modelnet40_airplane_samples.npz'))
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists(), f'data-dir not found: {data_dir}'

    # Build a minimal dataloader cfg compatible with openpoints
    dataset_cfg = edict({
        'common': edict({'NAME': 'ModelNet40Ply2048', 'data_dir': str(data_dir)}),
        args.split: edict({'split': args.split, 'num_points': int(args.num_points)})
    })
    dataloader_cfg = edict({'num_workers': 0, 'batch_size': 1, 'shuffle': args.split == 'train'})
    datatransforms_cfg = edict({args.split: ['PointsToTensor']})

    loader = build_dataloader_from_cfg(
        batch_size=1,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        datatransforms_cfg=datatransforms_cfg,
        split=args.split,
        distributed=False,
    )

    # Map class name to index via shape_names.txt if available
    shape_names = load_shape_names(data_dir)
    target_idx = None
    if shape_names:
        name2idx = {n: i for i, n in enumerate(shape_names)}
        target_idx = name2idx.get(args.class_name.lower())
    
    xyz_list = []
    label_list = []
    xfeat_list = []

    for batch in loader:
        pos = batch.get('pos')  # (B, N, 3)
        if pos is None:
            continue
        y = batch.get('y') or batch.get('label')
        if y is None:
            continue
        # Get scalar label
        if hasattr(y, 'item'):
            y_val = int(y.item())
        elif isinstance(y, (list, tuple, np.ndarray)):
            y_val = int(np.array(y).reshape(-1)[0])
        else:
            try:
                y_val = int(y)
            except Exception:
                continue
        # If we know the target index, filter by it; otherwise accept all
        if target_idx is not None and y_val != target_idx:
            continue
        xyz = pos.detach().cpu().numpy()[0]  # (N,3)
        xfeat = batch.get('x')
        if xfeat is not None:
            xfeat = xfeat.detach().cpu().numpy()[0]  # could be (C,N) or (N,C) depending on feature_last
        xyz_list.append(xyz.astype(np.float32))
        label_list.append(y_val)
        xfeat_list.append(xfeat)
        if len(xyz_list) >= args.count:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as npz; store x as an object array if shapes vary or are None
    np.savez(out_path,
             xyz=np.array(xyz_list, dtype=object),
             label=np.array(label_list, dtype=np.int64),
             x=np.array(xfeat_list, dtype=object),
             class_name=args.class_name,
             split=args.split,
             num_points=args.num_points)
    print(f'Wrote {len(xyz_list)} samples to {out_path}')


if __name__ == '__main__':
    main()
