"""CPU-only structural gate for KDPoint's four hardware-geometry graphs."""

import argparse
import gc
import json
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openpoints.models import build_model_from_cfg
from openpoints.models.layers.group import GroupAll, QueryAndGroup
from openpoints.utils import EasyConfig


def _resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_record_model(root, record_path):
    with open(record_path) as stream:
        record = json.load(stream)

    cfg_path = _resolve(root, record["cfg"])
    ckpt_path = _resolve(root, record["ckpt"])
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg.update(record.get("opts", []))
    cfg.model.encoder_args.coord_hardware_exact = True

    model = build_model_from_cfg(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state, strict=True)
    _require(not incompatible.missing_keys, f"missing keys: {incompatible.missing_keys}")
    _require(
        not incompatible.unexpected_keys,
        f"unexpected keys: {incompatible.unexpected_keys}",
    )
    model.eval()
    return record, cfg, model, checkpoint


def check_record(root, record_path):
    record, _, model, checkpoint = load_record_model(root, record_path)

    encoder = model.encoder
    _require(
        getattr(encoder, "coord_hardware_exact", False),
        "encoder did not retain coord_hardware_exact=True",
    )
    encoder_groupers = [
        module for module in encoder.modules() if isinstance(module, QueryAndGroup)
    ]
    _require(encoder_groupers, "encoder graph has no radius-BQ QueryAndGroup module")
    _require(
        all(module.coord_hardware_exact for module in encoder_groupers),
        "at least one encoder QueryAndGroup did not enable hardware geometry",
    )
    encoder_group_all = [
        module for module in encoder.modules() if isinstance(module, GroupAll)
    ]
    _require(
        all(module.coord_hardware_exact for module in encoder_group_all),
        "at least one encoder GroupAll did not enable hardware geometry",
    )

    decoder = getattr(model, "decoder", None)
    decoder_fp = []
    if record["task"] == "seg":
        _require(decoder is not None, "segmentation graph has no decoder")
        _require(
            getattr(decoder, "coord_hardware_exact", False),
            "decoder did not retain coord_hardware_exact=True",
        )
        decoder_fp = [
            module
            for module in decoder.modules()
            if module.__class__.__name__ in {"FeaturePropogation", "PointNetFPModule"}
        ]
        _require(decoder_fp, "segmentation decoder graph has no FP module")
        _require(
            all(module.coord_hardware_exact for module in decoder_fp),
            "at least one decoder FP module did not enable exact interpolation",
        )
    else:
        _require(decoder is None, "classification graph unexpectedly has a decoder")

    result = {
        "record": os.path.basename(record_path),
        "task": record["task"],
        "cfg": record["cfg"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "strict_load": True,
        "encoder_query_groupers": len(encoder_groupers),
        "encoder_group_all": len(encoder_group_all),
        "decoder_fp_modules": len(decoder_fp),
    }
    del checkpoint, model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build and strictly load hardware-geometry graphs without a dataset/GPU"
    )
    parser.add_argument("records", nargs="+", help="authoritative PTQ result JSON files")
    parser.add_argument("--root", default=ROOT, help="PointNeXt repository root")
    args = parser.parse_args()

    results = [
        check_record(os.path.abspath(args.root), os.path.abspath(record))
        for record in args.records
    ]
    tasks = sorted(result["task"] for result in results)
    _require(len(results) == 4, f"expected four network records, got {len(results)}")
    _require(tasks == ["cls", "cls", "seg", "seg"], f"unexpected task set: {tasks}")
    print(json.dumps({"status": "PASS", "networks": results}, indent=2))


if __name__ == "__main__":
    main()
