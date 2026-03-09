"""
Batch PTQ for all PointNeXt-S kdtree_simple models.
Usage: conda activate pointnext && python quant/run_all_ptq.py
"""

import os, sys, time, json, logging, gc
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

BACKEND = "fbgemm"
NUM_CALIB_BATCHES = 30
OUTPUT_DIR = "quant/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "batch_ptq_log.txt"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    (16, "cfgs/s3dis/pointnext-s_kdtree_simple_16.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_16-ngpus4-20260108-235751-YG7hDWqigCKNtXkbySXUyF"),
    (64, "cfgs/s3dis/pointnext-s_kdtree_simple_64.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_64-ngpus4-20260108-120528-hgFLdEiETa7zEpXGeGM7gc"),
    (96, "cfgs/s3dis/pointnext-s_kdtree_simple_96.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_96-ngpus4-20260109-171704-JwJHikVaiK8q4z2DavdcJU"),
    (128, "cfgs/s3dis/pointnext-s_kdtree_simple_128.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_128-ngpus4-20260110-192922-Vfdejy4CdHAKfmY35stCtJ"),
    (160, "cfgs/s3dis/pointnext-s_kdtree_simple_160.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_160-ngpus4-20260112-145953-ezd4Lqc9wjdyJxEZLBw424"),
    (192, "cfgs/s3dis/pointnext-s_kdtree_simple_192.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_192-ngpus4-20260113-100402-Yw4aXu5x22pGVub6FzPTAa"),
    (256, "cfgs/s3dis/pointnext-s_kdtree_simple_256.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_256-ngpus4-20260111-134626-4XdFxK2jvuTogm4Zi5Tu8k"),
    (512, "cfgs/s3dis/pointnext-s_kdtree_simple_512.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_512-ngpus4-20260115-161254-E2R4YpmvfQVeaoBNKGdEC3"),
    (1024, "cfgs/s3dis/pointnext-s_kdtree_simple_1024.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_1024-ngpus4-20260114-100848-RxWoUB773rjuVVFrL56jGL"),
    (2048, "cfgs/s3dis/pointnext-s_kdtree_simple_2048.yaml",
     "s3dis-train-pointnext-s_kdtree_simple_2048-ngpus4-20260116-112647-Lnhs998ZXBtgUYKE4wKe4V"),
]


def evaluate(model, val_loader, cfg, device="cuda", tag="FP32"):
    model.eval()
    model.to(device)
    num_classes = cfg.num_classes
    ignore_index = cfg.get("ignore_index", None)
    feature_keys = cfg.get("feature_keys", "pos")
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    t0 = time.time()
    with torch.no_grad():
        for data in val_loader:
            for k in data:
                data[k] = data[k].to(device, non_blocking=True)
            data["x"] = get_features_by_keys(data, feature_keys)
            logits = model(data)
            target = data["y"].squeeze(-1)
            cm.update(logits.argmax(dim=1), target)
    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)
    elapsed = time.time() - t0
    logger.info("[%s] OA=%.2f  mAcc=%.2f  mIoU=%.2f  (%.1fs)", tag, oa, macc, miou, elapsed)
    return miou, macc, oa, ious, accs

def run_ptq(leaf_size, cfg_path, ckpt_dir):
    ckpt_path = "log/s3dis/{}/checkpoint/{}_ckpt_best.pth".format(ckpt_dir, ckpt_dir)
    logger.info("=" * 70)
    logger.info("PTQ for leaf_size=%d", leaf_size)
    logger.info("  Config: %s", cfg_path)
    logger.info("  Checkpoint: %s", ckpt_path)
    logger.info("=" * 70)

    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    model = build_model_from_cfg(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    best_val = ckpt.get("best_val", "?")
    logger.info("Loaded (epoch=%s, best_val=%s)", ckpt.get("epoch", "?"), best_val)
    model.eval()

    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    logger.info("Val set: %d samples", len(val_loader.dataset))

    # FP32 baseline
    model.cuda()
    fp32_miou, fp32_macc, fp32_oa, fp32_ious, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="FP32[leaf=%d]" % leaf_size)

    # Quant prep
    model.cpu().eval()
    swap_custom_convs_to_standard(model)
    fuse_convbn_modules(model)

    # FP32-fused eval
    model.cuda()
    fused_miou, fused_macc, fused_oa, _, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="FP32-fused[leaf=%d]" % leaf_size)
    model.cpu().eval()

    disable_quantization_for_geometry(model)
    torch.backends.quantized.engine = BACKEND
    model.qconfig = quant.get_default_qat_qconfig(BACKEND)
    model.train()
    quant.prepare_qat(model, inplace=True)
    print_fakequant_status(model, tag="after prepare_qat")

    # Phase 1: calibration
    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)
    model.eval()
    model.cuda()
    feature_keys = cfg.get("feature_keys", "pos")
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= NUM_CALIB_BATCHES:
                break
            for k in data:
                data[k] = data[k].cuda(non_blocking=True)
            data["x"] = get_features_by_keys(data, feature_keys)
            model(data)
    calib_time = time.time() - t0
    logger.info("Calibration done in %.1fs (%d batches)", calib_time, min(NUM_CALIB_BATCHES, len(val_loader)))

    # Phase 2: INT8-sim eval
    model.apply(quant.enable_fake_quant)
    model.apply(quant.disable_observer)
    print_fakequant_status(model, tag="INT8-sim eval")
    int8_miou, int8_macc, int8_oa, int8_ious, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="INT8-sim[leaf=%d]" % leaf_size)

    # Convert
    model.cpu().eval()
    quant.convert(model, inplace=True)
    q_count = sum(1 for _, m in model.named_modules() if "quantized" in type(m).__module__)

    # Save
    save_name = "pointnext_s_simple%d_ptq_int8.pth" % leaf_size
    save_path = os.path.join(OUTPUT_DIR, save_name)
    torch.save({
        "model": model.state_dict(),
        "config": cfg_path,
        "leaf_size": leaf_size,
        "backend": BACKEND,
        "calib_batches": NUM_CALIB_BATCHES,
        "quantized_modules": q_count,
        "fp32_miou": fp32_miou, "fp32_oa": fp32_oa, "fp32_macc": fp32_macc,
        "fused_miou": fused_miou, "fused_oa": fused_oa, "fused_macc": fused_macc,
        "int8_miou": int8_miou, "int8_oa": int8_oa, "int8_macc": int8_macc,
    }, save_path)
    logger.info("Saved: %s", save_path)

    logger.info("-" * 60)
    logger.info("[leaf=%d] mIoU: FP32=%.2f  FP32-fused=%.2f  INT8-sim=%.2f",
                leaf_size, fp32_miou, fused_miou, int8_miou)
    logger.info("[leaf=%d] BN-folding delta: %+.2f", leaf_size, fused_miou - fp32_miou)
    logger.info("[leaf=%d] Quant delta:      %+.2f (INT8 vs fused)", leaf_size, int8_miou - fused_miou)
    logger.info("[leaf=%d] Total delta:      %+.2f (INT8 vs FP32)", leaf_size, int8_miou - fp32_miou)
    logger.info("-" * 60)

    del model, ckpt, state
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "leaf_size": leaf_size,
        "best_val": float(best_val) if isinstance(best_val, (int, float)) else str(best_val),
        "fp32_miou": round(fp32_miou, 4), "fp32_macc": round(fp32_macc, 4), "fp32_oa": round(fp32_oa, 4),
        "fused_miou": round(fused_miou, 4), "fused_macc": round(fused_macc, 4), "fused_oa": round(fused_oa, 4),
        "int8_miou": round(int8_miou, 4), "int8_macc": round(int8_macc, 4), "int8_oa": round(int8_oa, 4),
        "bn_fold_delta": round(fused_miou - fp32_miou, 4),
        "quant_delta": round(int8_miou - fused_miou, 4),
        "total_delta": round(int8_miou - fp32_miou, 4),
        "q_modules": q_count,
    }

def main():
    results = []
    total = len(EXPERIMENTS)

    for idx, (leaf_size, cfg_path, ckpt_dir) in enumerate(EXPERIMENTS, 1):
        logger.info("\n\n>>> [%d/%d] Starting PTQ for leaf_size=%d <<<\n", idx, total, leaf_size)
        try:
            r = run_ptq(leaf_size, cfg_path, ckpt_dir)
            results.append(r)
        except Exception as e:
            logger.error("FAILED leaf_size=%d: %s", leaf_size, e, exc_info=True)
            results.append({"leaf_size": leaf_size, "error": str(e)})

    logger.info("\n" + "=" * 90)
    logger.info("BATCH PTQ SUMMARY  (%d models)", len(results))
    logger.info("=" * 90)
    logger.info("%6s | %10s | %11s | %10s | %8s | %8s | %8s",
                "Leaf", "FP32 mIoU", "Fused mIoU", "INT8 mIoU", "BN-fold", "Quant D", "Total D")
    logger.info("-" * 90)
    for r in sorted(results, key=lambda x: x["leaf_size"]):
        if "error" in r:
            logger.info("%6d | ERROR: %s", r["leaf_size"], r["error"])
        else:
            logger.info("%6d | %10.2f | %11.2f | %10.2f | %+8.2f | %+8.2f | %+8.2f",
                        r["leaf_size"], r["fp32_miou"], r["fused_miou"], r["int8_miou"],
                        r["bn_fold_delta"], r["quant_delta"], r["total_delta"])
    logger.info("=" * 90)

    json_path = os.path.join(OUTPUT_DIR, "batch_ptq_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", json_path)


if __name__ == "__main__":
    main()

