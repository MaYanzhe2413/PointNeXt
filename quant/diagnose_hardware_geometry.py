"""Isolate q9/BQ changes from runtime DpQuant on one calibrated state."""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kdpoint_export import prepare_calibrated_model
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.utils import EasyConfig
from ptq_general import (
    EVALUATORS,
    bind_hardware_dp_qparams,
    set_hardware_geometry_mode,
)
import openpoints.models.backbone.pointnetv2 as pointnetv2
import openpoints.models.backbone.pointnext as pointnext
from openpoints.models.layers.kdpoint_geometry import q9_main_codes
from openpoints.models.layers.upsampling import three_interpolate, three_nn


def _ideal_three_interpolation(
        unknown_xyz, known_xyz, known_feat, coordinate_bits, **_):
    if coordinate_bits == 8:
        unknown = q9_main_codes(unknown_xyz)
        known = q9_main_codes(known_xyz)
    elif coordinate_bits == 9:
        unknown = unknown_xyz.to(torch.int64)
        known = known_xyz.to(torch.int64)
    else:
        raise ValueError(f"unsupported coordinate width: {coordinate_bits}")
    unknown = unknown.to(dtype=unknown_xyz.dtype).contiguous()
    known = known.to(dtype=known_xyz.dtype).contiguous()
    distances, indices = three_nn(unknown, known)
    zero = distances == 0
    any_zero = zero.any(dim=-1, keepdim=True)
    first_zero = zero & (zero.to(torch.int64).cumsum(dim=-1) == 1)
    reciprocal = 1.0 / distances.clamp_min(1e-8)
    ideal = reciprocal / reciprocal.sum(dim=-1, keepdim=True)
    weights = torch.where(any_zero, first_zero.to(ideal.dtype), ideal)
    return three_interpolate(
        known_feat, indices.contiguous(), weights.contiguous()
    )


def _set_decoder_interpolation(coordinate_bits):
    implementation = functools.partial(
        _ideal_three_interpolation, coordinate_bits=coordinate_bits
    )
    pointnetv2.three_interpolation = implementation
    pointnext.three_interpolation = implementation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()

    model, metadata, _ = prepare_calibrated_model(
        args.project_root, args.state
    )
    cfg = EasyConfig()
    cfg.load(str(args.project_root / metadata["cfg"]), recursive=True)
    cfg.update(metadata.get("opts", []))
    loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    evaluate = EVALUATORS[metadata["task"]]

    set_hardware_geometry_mode(model, False)
    disabled = evaluate(model, loader, cfg, tag="INT8 geometry-disabled")

    set_hardware_geometry_mode(model, True)
    unbound = evaluate(model, loader, cfg, tag="INT8 q9/BQ, dp-unbound")

    set_hardware_geometry_mode(model, True)
    bindings = bind_hardware_dp_qparams(model)
    bound = evaluate(model, loader, cfg, tag="INT8 q9/BQ, dp-bound")

    decoder_ablation = None
    if getattr(model, "decoder", None) is not None:
        original_pn2 = pointnetv2.three_interpolation
        original_pnx = pointnext.three_interpolation
        try:
            _set_decoder_interpolation(8)
            q8_ideal = evaluate(
                model, loader, cfg, tag="INT8 q8 KNN, ideal weights"
            )
            _set_decoder_interpolation(9)
            q9_ideal = evaluate(
                model, loader, cfg, tag="INT8 q9 KNN, ideal weights"
            )
        finally:
            pointnetv2.three_interpolation = original_pn2
            pointnext.three_interpolation = original_pnx
        decoder_ablation = {
            "q8_knn_ideal_weights": q8_ideal,
            "q9_knn_ideal_weights": q9_ideal,
        }

    print(json.dumps({
        "status": "PASS",
        "geometry_disabled": disabled,
        "q9_bq_dp_unbound": unbound,
        "q9_bq_dp_bound": bound,
        "decoder_ablation": decoder_ablation,
        "dp_bindings": bindings,
    }, indent=2))


if __name__ == "__main__":
    main()
