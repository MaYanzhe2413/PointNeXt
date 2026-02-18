"""
Collect neighbor reuse statistics during a single validation pass.
Logs how often points are reused across neighborhoods (cross-center overlap)
and whether a single neighborhood contains duplicates.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
# Ensure repository root is on sys.path when running as a script
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from openpoints.utils import EasyConfig, set_random_seed, load_checkpoint
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers.group import QueryAndGroup, KNNGroup


class NeighborStats:
    def __init__(self):
        self.records = defaultdict(lambda: dict(
            dup_entries=0,
            total_entries=0,
            inner_dup=0,
            total_neighbors=0,
            batches=0,
            last_k=None,
        ))

    def hook(self, name):
        def _hook(idx: torch.Tensor):
            self.record(name, idx)
        return _hook

    def record(self, name: str, idx: torch.Tensor) -> None:
        idx_cpu = idx.detach().to(device='cpu', non_blocking=True)
        b, m, k = idx_cpu.shape

        # Global counts of each index across all neighborhoods (no dedup)
        flat = idx_cpu.reshape(-1)
        uniq, counts = torch.unique(flat, return_counts=True)

        # For each neighborhood, count how many neighbor entries appear elsewhere (count > 1)
        overlap_count_sum = 0
        for bi in range(b):
            for mi in range(m):
                neigh = idx_cpu[bi, mi].flatten()
                idx_pos = torch.searchsorted(uniq, neigh)
                neigh_counts = counts[idx_pos]
                overlap_count = (neigh_counts > 1).sum().item()
                overlap_count_sum += overlap_count

        # Inner-dup: duplicates inside a neighborhood
        inner_dup = 0
        for bi in range(b):
            for mi in range(m):
                inner_dup += k - torch.unique(idx_cpu[bi, mi]).numel()

        rec = self.records[name]
        rec['dup_entries'] += int(overlap_count_sum)
        rec['total_entries'] += int(b * m)
        rec['inner_dup'] += int(inner_dup)
        rec['total_neighbors'] += int(b * m * k)
        rec['batches'] += 1
        rec['last_k'] = k
    def summary(self):
        report = {}
        for name, rec in self.records.items():
            dup_ratio = rec['dup_entries'] / rec['total_entries'] if rec['total_entries'] else 0.0
            inner_ratio = rec['inner_dup'] / rec['total_neighbors'] if rec['total_neighbors'] else 0.0
            report[name] = dict(
                dup_ratio=dup_ratio,
                inner_dup_ratio=inner_ratio,
                dup_entries=rec['dup_entries'],
                total_entries=rec['total_entries'],
                inner_dup=rec['inner_dup'],
                total_neighbors=rec['total_neighbors'],
                batches=rec['batches'],
                k=rec['last_k'],
            )
        return report


def attach_neighbor_hooks(model, collector: NeighborStats):
    for name, module in model.named_modules():
        if isinstance(module, (QueryAndGroup, KNNGroup)):
            module._neighbor_logger = collector.hook(name)


def build_cfg(cfg_path: str, opts) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = 42
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    return cfg


def build_model(cfg: EasyConfig):
    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda().eval()
    return model


def build_loader(cfg: EasyConfig, split: str):
    loader = build_dataloader_from_cfg(
        cfg.get('val_batch_size', cfg.batch_size),
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split=split,
        distributed=False,
    )
    return loader


def run_val(model, loader, cfg: EasyConfig, collector: NeighborStats, max_batches: int = None):
    model.eval()
    npoints = cfg.num_points
    pbar = tqdm(enumerate(loader), total=len(loader), desc='neighbor-val')
    with torch.no_grad():
        for bid, data in pbar:
            if max_batches is not None and bid >= max_batches:
                break
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            points = data['x'][:, :npoints]
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
            _ = model(data)

    return collector.summary()


def main():
    parser = argparse.ArgumentParser(description='Neighbor reuse statistics on validation set')
    parser.add_argument('--cfg', required=True, help='Path to config yaml')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path (override cfg.pretrained_path)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--max-batches', type=int, default=None, help='Limit batches for quick stats')
    parser.add_argument('--out', type=str, default=None, help='Save JSON summary to this path')
    args, opts = parser.parse_known_args()

    cfg = build_cfg(args.cfg, opts)
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    model = build_model(cfg)
    ckpt_path = args.ckpt or cfg.get('pretrained_path', None)
    ckpt_path = Path(ckpt_path) if ckpt_path else None
    if ckpt_path and ckpt_path.exists():
        load_checkpoint(model, pretrained_path=str(ckpt_path))
    elif ckpt_path:
        print(f"[WARN] checkpoint not found: {ckpt_path}; proceeding without loading")

    loader = build_loader(cfg, args.split)

    collector = NeighborStats()
    attach_neighbor_hooks(model, collector)
    summary = run_val(model, loader, cfg, collector, max_batches=args.max_batches)
    print("\nNeighbor reuse summary:")
    for name, rec in summary.items():
        print(f"- {name}: dup_ratio={rec['dup_ratio']:.6f}, inner_dup_ratio={rec['inner_dup_ratio']:.6f}, k={rec['k']}, batches={rec['batches']}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {out_path}")



if __name__ == '__main__':
    main()
