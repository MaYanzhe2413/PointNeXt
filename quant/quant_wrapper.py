"""
PTQ (Post-Training Quantization) for PointNeXt-S on S3DIS.

Uses prepare_qat() to insert FakeQuantize modules, but does NOT train.
Phase 1: disable fake_quant -> calibrate (pure observer, like standard PTQ)
Phase 2: enable  fake_quant -> evaluate  (true INT8-sim accuracy)

Usage:
    conda activate pointnext
    python quant/quant_wrapper.py
"""

import os, sys, time, logging
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.quantization as quant
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.utils import EasyConfig, ConfusionMatrix, get_mious
from openpoints.models.layers.quant_utils import (
    swap_custom_convs_to_standard,
    fuse_convbn_modules,
    disable_quantization_for_geometry,
    print_fakequant_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ======================= config =======================
CFG_PATH = "cfgs/s3dis/pointnext-s_kdtree_simple_128.yaml"
CKPT_PATH = (
    "log/s3dis/"
    "s3dis-train-pointnext-s_kdtree_simple_128"
    "-ngpus4-20260110-192922-Vfdejy4CdHAKfmY35stCtJ"
    "/checkpoint/"
    "s3dis-train-pointnext-s_kdtree_simple_128"
    "-ngpus4-20260110-192922-Vfdejy4CdHAKfmY35stCtJ"
    "_ckpt_best.pth"
)
BACKEND = "fbgemm"
NUM_CALIB_BATCHES = 30
OUTPUT_DIR = "quant/output"


def evaluate(model, val_loader, cfg, device="cuda", tag="FP32"):
    """Evaluate segmentation mIoU / OA / mAcc on full val set."""
    model.eval()
    model.to(device)
    num_classes = cfg.num_classes
    ignore_index = cfg.get("ignore_index", None)
    feature_keys = cfg.get("feature_keys", "pos")
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            for k in data:
                data[k] = data[k].to(device, non_blocking=True)
            data["x"] = get_features_by_keys(data, feature_keys)
            logits = model(data)
            target = data["y"].squeeze(-1)
            cm.update(logits.argmax(dim=1), target)
    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)
    elapsed = time.time() - t0
    logger.info("[%s] OA=%.2f  mAcc=%.2f  mIoU=%.2f  (%.1fs)", tag, oa, macc, miou, elapsed)
    with np.printoptions(precision=2, suppress=True):
        logger.info("[%s] IoU per class: %s", tag, ious)
    return miou, macc, oa, ious, accs


def main():
    # ===== 1. build model + load weights =====
    logger.info("Config: %s", CFG_PATH)
    cfg = EasyConfig()
    cfg.load(CFG_PATH, recursive=True)
    model = build_model_from_cfg(cfg.model)

    logger.info("Checkpoint: %s", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    logger.info("Loaded (epoch=%s, best_val=%s)", ckpt.get("epoch", "?"), ckpt.get("best_val", "?"))
    model.eval()

    # ===== 2. build val dataloader =====
    logger.info("Building S3DIS val dataloader ...")
    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    logger.info("Val set: %d samples", len(val_loader.dataset))

    # ===== 3. FP32 baseline evaluation =====
    logger.info("=" * 60)
    logger.info("Evaluating FP32 baseline on GPU ...")
    model.cuda()
    fp32_miou, fp32_macc, fp32_oa, fp32_ious, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="FP32"
    )

    # ===== 4. quantization preparation =====
    model.cpu().eval()
    logger.info("=" * 60)
    logger.info("[Step 0] swap custom Conv to standard")
    swap_custom_convs_to_standard(model)

    logger.info("[Step 1] fuse Conv-BN-ReLU")
    fuse_convbn_modules(model)

    # ---- 4a. evaluate fused model (shows BN-folding error only) ----
    logger.info("=" * 60)
    logger.info("Evaluating FP32-fused model (after BN-folding, before quant) ...")
    model.cuda()
    fused_miou, fused_macc, fused_oa, _, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="FP32-fused"
    )
    logger.info("BN-folding delta:  mIoU %+.2f", fused_miou - fp32_miou)
    model.cpu().eval()

    logger.info("[Step 2] disable quant for geometry ops")
    disable_quantization_for_geometry(model)

    logger.info("[Step 3] set QAT qconfig (backend=%s)", BACKEND)
    torch.backends.quantized.engine = BACKEND
    model.qconfig = quant.get_default_qat_qconfig(BACKEND)

    logger.info("[Step 4] prepare_qat (insert FakeQuantize modules)")
    model.train()
    quant.prepare_qat(model, inplace=True)
    print_fakequant_status(model, tag="after prepare_qat")

    # ===== 5. calibration (Phase 1: disable fake_quant, observers only) =====
    # Disable fake_quant so observers see raw FP32 activations
    # (standard PTQ calibration behavior).
    logger.info("[Phase 1] Disable FakeQuantize, enable Observer -> calibrate")
    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)
    print_fakequant_status(model, tag="Phase1: calibration")
    model.eval()
    model.cuda()

    feature_keys = cfg.get("feature_keys", "pos")
    logger.info("Calibrating %d batches on GPU ...", NUM_CALIB_BATCHES)
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= NUM_CALIB_BATCHES:
                break
            for k in data:
                data[k] = data[k].cuda(non_blocking=True)
            data["x"] = get_features_by_keys(data, feature_keys)
            model(data)
            if (i + 1) % 10 == 0:
                logger.info("  batch %d/%d", i + 1, NUM_CALIB_BATCHES)
    elapsed = time.time() - t0
    logger.info("Calibration done in %.1fs (%d batches)", elapsed,
                min(NUM_CALIB_BATCHES, len(val_loader)))

    # ===== 6. evaluate with FakeQuantize enabled (true INT8-sim) =====
    # Phase 2: Enable fake_quant (simulates INT8 clamp+round-trip),
    #          disable observer updates (freeze scale/zero_point).
    # Model stays on GPU because geometry ops are CUDA-only.
    logger.info("=" * 60)
    logger.info("[Phase 2] Enable FakeQuantize, disable Observer -> evaluate")
    model.apply(quant.enable_fake_quant)
    model.apply(quant.disable_observer)
    print_fakequant_status(model, tag="Phase2: INT8-sim eval")
    int8_miou, int8_macc, int8_oa, int8_ious, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="INT8-sim"
    )

    # ===== 7. convert to actual INT8 (for saving) =====
    logger.info("Converting to real INT8 (for export only) ...")
    model.cpu().eval()
    quant.convert(model, inplace=True)

    q_count = sum(1 for _, m in model.named_modules()
                  if "quantized" in type(m).__module__)
    logger.info("Quantized modules: %d", q_count)

    # ===== 8. summary =====
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    logger.info("mIoU:  FP32=%.2f  FP32-fused=%.2f  INT8-sim=%.2f",
                fp32_miou, fused_miou, int8_miou)
    logger.info("  BN-folding delta:   %+.2f", fused_miou - fp32_miou)
    logger.info("  Quantization delta: %+.2f  (INT8-sim vs FP32-fused)",
                int8_miou - fused_miou)
    logger.info("  Total delta:        %+.2f  (INT8-sim vs FP32)",
                int8_miou - fp32_miou)
    logger.info("-" * 60)
    logger.info("OA:    FP32=%.2f  ->  INT8-sim=%.2f  (delta=%+.2f)",
                fp32_oa, int8_oa, int8_oa - fp32_oa)
    logger.info("mAcc:  FP32=%.2f  ->  INT8-sim=%.2f  (delta=%+.2f)",
                fp32_macc, int8_macc, int8_macc - fp32_macc)

    # ===== 9. save =====
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "pointnext_s_simple128_ptq_int8.pth")
    torch.save({
        "model": model.state_dict(),
        "config": CFG_PATH,
        "backend": BACKEND,
        "calib_batches": NUM_CALIB_BATCHES,
        "quantized_modules": q_count,
        "fp32_miou": fp32_miou, "fp32_oa": fp32_oa, "fp32_macc": fp32_macc,
        "fused_miou": fused_miou, "fused_oa": fused_oa, "fused_macc": fused_macc,
        "int8_miou": int8_miou, "int8_oa": int8_oa, "int8_macc": int8_macc,
    }, save_path)
    logger.info("Saved: %s", save_path)
    logger.info("All done!")


if __name__ == "__main__":
    main()
