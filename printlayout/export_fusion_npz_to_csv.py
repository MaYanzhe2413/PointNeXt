#!/usr/bin/env python3
"""
Export fusion probe NPZ tensors to CSV files.

Reads stages.npz (saved by layer_fusion_probe.py) and writes:
 - idx_group_i.csv: (S, K) int indices into previous stage
 - idx_sample_i.csv: (S,) int sampling indices (FPS/KDTree)
 - query_xyz_i.csv: (S, 3) float centers
 - support_xyz_i.csv: (N, 3) float support points of previous stage

Skips entries that are None (object arrays). Handles (B, ...) by assuming B==1 and squeezing.
"""
import argparse
import os
from pathlib import Path
import numpy as np


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """If array has batch-first shape (1, ...), squeeze the first dim."""
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr.reshape(arr.shape[1:])
    return arr


def _make_header_for_key(key: str, data: np.ndarray) -> str:
    # Heuristic headers by key pattern
    if key.startswith('query_xyz_') or key.startswith('support_xyz_'):
        return 'x,y,z'
    if key.startswith('idx_sample_'):
        return 'sample_idx'
    if key.startswith('idx_group_'):
        # 2D (S,K)
        if data.ndim >= 2:
            return ','.join([f'k{j}' for j in range(data.shape[1])])
        return 'k0'
    # Generic
    if data.ndim == 1:
        return 'value'
    elif data.ndim >= 2:
        cols = data.shape[1]
        return ','.join([f'c{j}' for j in range(cols)])
    return ''


def export_npz(npz_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = np.load(npz_path, allow_pickle=True)
    index_lines = []

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
        data = squeeze_batch(np.array(arr))
        # Determine file name and save
        fname = f"{key}.csv"
        fpath = out / fname
        # 1D/2D save as-is; >2D flatten last dims to rows
        header = _make_header_for_key(key, data)
        if data.ndim == 1:
            np.savetxt(fpath, data, delimiter=",", header=header, comments='')
        elif data.ndim == 2:
            np.savetxt(fpath, data, delimiter=",", header=header, comments='')
        else:
            # reshape to (rows, cols)
            rows = data.shape[0]
            cols = int(np.prod(data.shape[1:]))
            flat = data.reshape(rows, cols)
            gen_header = ','.join([f'c{j}' for j in range(cols)])
            np.savetxt(fpath, flat, delimiter=",", header=gen_header, comments='')
        index_lines.append(f"{fname}\tshape={tuple(data.shape)}\tdtype={data.dtype}")

    with open(out / "csv_index.txt", "w") as f:
        f.write("\n".join(index_lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to stages.npz")
    ap.add_argument("--out", required=True, help="Output directory for CSV files")
    args = ap.parse_args()
    export_npz(args.npz, args.out)


if __name__ == "__main__":
    main()
