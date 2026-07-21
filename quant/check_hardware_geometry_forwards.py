"""Run one real validation batch through each hardware-geometry graph."""

import argparse
import gc
import json
import os
import sys
import time

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from check_hardware_geometry_graphs import load_record_model
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.models.layers.group import ball_query
from openpoints.models.layers.kdpoint_geometry import (
    strict_ball_query,
    wfu_weights_exact_tensor,
)
from openpoints.models.layers.upsampling import three_nn


def _to_device(data, device):
    return {
        key: value.to(device, non_blocking=False)
        if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def check_cuda_primitives(device):
    support = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [3.0, 0.0, 0.0],
    ]], device=device)
    query = torch.zeros((1, 1, 3), device=device)
    bq_indices = strict_ball_query(
        ball_query, 2.0, 4, support, query,
        torch.ones((1, 1, 1), dtype=torch.float64, device=device),
    )
    expected_bq = torch.tensor([[[0, 1, 3, 0]]], dtype=torch.int32, device=device)
    if not torch.equal(bq_indices, expected_bq):
        raise RuntimeError(
            f"strict BQ boundary/order mismatch: {bq_indices.cpu().tolist()}"
        )

    tied_known = torch.tensor([[
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]], device=device)
    _, knn_indices = three_nn(query.contiguous(), tied_known.contiguous())
    expected_knn = torch.tensor([[[0, 1, 2]]], dtype=torch.int32, device=device)
    if not torch.equal(knn_indices, expected_knn):
        raise RuntimeError(
            f"three-NN tie-break mismatch: {knn_indices.cpu().tolist()}"
        )

    distance_sq = torch.tensor([[0, 1, 1], [1, 1, 1]], device=device)
    weights = wfu_weights_exact_tensor(distance_sq)
    if weights[0].tolist() != [128, 0, 0]:
        raise RuntimeError(f"WFU zero-distance policy mismatch: {weights[0].tolist()}")
    if not torch.equal(weights.sum(dim=-1), torch.full((2,), 128, device=device)):
        raise RuntimeError(f"WFU weights do not sum to 128: {weights.cpu().tolist()}")
    return {
        "strict_bq_indices": bq_indices.cpu().tolist(),
        "three_nn_tie_indices": knn_indices.cpu().tolist(),
        "wfu_weights": weights.cpu().tolist(),
    }


def check_forward(root, record_path, device):
    record, cfg, model, checkpoint = load_record_model(root, record_path)
    cfg.dataloader.num_workers = 0
    loader = build_dataloader_from_cfg(
        1, cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    data = _to_device(next(iter(loader)), device)
    data["x"] = get_features_by_keys(data, cfg.get("feature_keys", "pos"))

    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(data)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not torch.is_tensor(output):
        raise RuntimeError(f"model returned {type(output).__name__}, expected Tensor")
    if not bool(torch.isfinite(output).all().detach().cpu().item()):
        raise RuntimeError("model output contains non-finite values")

    result = {
        "record": os.path.basename(record_path),
        "task": record["task"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_pos_shape": list(data["pos"].shape),
        "input_feature_shape": list(data["x"].shape),
        "output_shape": list(output.shape),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }
    del output, data, loader, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="One-batch CUDA smoke test for four KDPoint geometry graphs"
    )
    parser.add_argument("records", nargs="+", help="authoritative PTQ result JSON files")
    parser.add_argument("--root", default=ROOT, help="PointNeXt repository root")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ball-query and interpolation smoke tests")
    device = torch.device(args.device)
    primitive_checks = check_cuda_primitives(device)
    results = [
        check_forward(os.path.abspath(args.root), os.path.abspath(record), device)
        for record in args.records
    ]
    tasks = sorted(result["task"] for result in results)
    if len(results) != 4 or tasks != ["cls", "cls", "seg", "seg"]:
        raise RuntimeError(f"expected two cls and two seg records, got {tasks}")
    print(json.dumps({
        "status": "PASS",
        "primitive_checks": primitive_checks,
        "networks": results,
    }, indent=2))


if __name__ == "__main__":
    main()
