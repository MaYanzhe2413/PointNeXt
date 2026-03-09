"""
QAT (Quantization-Aware Training) for PointNeXt-S on S3DIS.

Usage:
    conda activate pointnext
    python quant/quant_wrapper_QAT.py
"""

import os, sys, time, logging
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.quantization as quant
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.utils import EasyConfig, ConfusionMatrix, get_mious, AverageMeter
from openpoints.loss import build_criterion_from_cfg
from openpoints.models.layers.quant_utils import (
    swap_custom_convs_to_standard,
    fuse_convbn_modules,
    disable_quantization_for_geometry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ==================== config ====================
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
NUM_FINETUNE_EPOCHS = 10
QAT_LR = 1e-4
QAT_WEIGHT_DECAY = 1e-4
FREEZE_OBSERVER_EPOCH = 8   # freeze observer in last 2 epochs
FREEZE_BN_EPOCH = 8         # freeze BN stats in last 2 epochs
OUTPUT_DIR = "quant/output"

def evaluate(model, val_loader, cfg, device="cuda", tag="FP32"):
    """Evaluate segmentation mIoU / OA / mAcc."""
    model.eval()
    model.to(device)
    num_classes = cfg.num_classes
    ignore_index = cfg.get("ignore_index", None)
    feature_keys = cfg.get("feature_keys", "pos")
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    with torch.no_grad():
        for data in val_loader:
            for k in data:
                data[k] = data[k].to(device, non_blocking=True)
            data["x"] = get_features_by_keys(data, feature_keys)
            logits = model(data)
            target = data["y"].squeeze(-1)
            cm.update(logits.argmax(dim=1), target)
    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)
    logger.info("[%s] OA=%.2f  mAcc=%.2f  mIoU=%.2f", tag, oa, macc, miou)
    return miou, macc, oa, ious, accs


def train_one_epoch(model, train_loader, criterion, optimizer, cfg, epoch):
    """One QAT finetune epoch."""
    model.train()
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.get("ignore_index", None))
    feature_keys = cfg.get("feature_keys", "pos")

    for i, data in enumerate(train_loader):
        for k in data:
            data[k] = data[k].cuda(non_blocking=True)
        data["x"] = get_features_by_keys(data, feature_keys)
        target = data["y"].squeeze(-1)

        logits = model(data)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_meter.update(loss.item())
        cm.update(logits.argmax(dim=1), target)

        if (i + 1) % 20 == 0:
            logger.info("  Epoch %d  batch %d/%d  loss=%.4f", epoch, i + 1, len(train_loader), loss_meter.avg)

    miou, macc, oa, _, _ = get_mious(cm.tp, cm.union, cm.count)
    return loss_meter.avg, miou, macc, oa

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

    # ===== 2. build dataloaders =====
    logger.info("Building S3DIS train + val dataloaders ...")
    train_loader = build_dataloader_from_cfg(
        cfg.batch_size, cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="train", distributed=False,
    )
    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_loader.dataset), len(val_loader.dataset))

    # ===== 3. FP32 baseline evaluation =====
    logger.info("=" * 60)
    logger.info("Evaluating FP32 baseline on GPU ...")
    model.cuda()
    fp32_miou, fp32_macc, fp32_oa, _, _ = evaluate(
        model, val_loader, cfg, device="cuda", tag="FP32"
    )
    model.cpu().eval()
    # ===== 4. quantization preparation (QAT) =====
    logger.info("=" * 60)
    logger.info("[Step 0] swap custom Conv to standard")
    swap_custom_convs_to_standard(model)

    logger.info("[Step 1] fuse Conv-BN-ReLU")
    fuse_convbn_modules(model)

    logger.info("[Step 2] disable quant for geometry ops")
    disable_quantization_for_geometry(model)

    logger.info("[Step 3] set QAT qconfig (backend=%s)", BACKEND)
    torch.backends.quantized.engine = BACKEND
    model.qconfig = quant.get_default_qat_qconfig(BACKEND)

    logger.info("[Step 4] quant.prepare_qat (insert FakeQuantize modules)")
    quant.prepare_qat(model, inplace=True)

    fq_count = sum(1 for _, m in model.named_modules()
                   if type(m).__name__ == "FakeQuantize")
    logger.info("FakeQuantize modules inserted: %d", fq_count)

    # ===== 5. setup loss, optimizer, scheduler =====
    ignore_idx = cfg.get("ignore_index", None)
    ce_kwargs = dict(label_smoothing=cfg.get("criterion_args", {}).get("label_smoothing", 0.0))
    if ignore_idx is not None:
        ce_kwargs["ignore_index"] = ignore_idx
    criterion = nn.CrossEntropyLoss(**ce_kwargs).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=QAT_LR,
        weight_decay=QAT_WEIGHT_DECAY,
    )

    # cosine annealing over finetune epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_FINETUNE_EPOCHS, eta_min=QAT_LR * 0.01
    )

    logger.info("QAT finetune: %d epochs, lr=%.1e, weight_decay=%.1e",
                NUM_FINETUNE_EPOCHS, QAT_LR, QAT_WEIGHT_DECAY)
    # ===== 6. QAT finetune loop =====
    model.cuda()
    best_miou = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_FINETUNE_EPOCHS + 1):
        logger.info("=" * 60)
        logger.info("QAT Epoch %d/%d  lr=%.2e", epoch, NUM_FINETUNE_EPOCHS, optimizer.param_groups[0]["lr"])

        # freeze observer & BN in late epochs
        if epoch >= FREEZE_OBSERVER_EPOCH:
            logger.info("  >> Freezing observers")
            model.apply(quant.disable_observer)
        if epoch >= FREEZE_BN_EPOCH:
            logger.info("  >> Freezing BN stats")
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        t0 = time.time()
        train_loss, train_miou, train_macc, train_oa = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg, epoch
        )
        scheduler.step()
        train_time = time.time() - t0

        logger.info("  Train: loss=%.4f  mIoU=%.2f  mAcc=%.2f  OA=%.2f  (%.1fs)",
                    train_loss, train_miou, train_macc, train_oa, train_time)

        # validate
        t0 = time.time()
        val_miou, val_macc, val_oa, _, _ = evaluate(
            model, val_loader, cfg, device="cuda", tag=f"QAT-E{epoch}"
        )
        val_time = time.time() - t0
        logger.info("  Val:   mIoU=%.2f  mAcc=%.2f  OA=%.2f  (%.1fs)",
                    val_miou, val_macc, val_oa, val_time)

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            # save best checkpoint (before convert)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            best_path = os.path.join(OUTPUT_DIR, "pointnext_s_simple128_qat_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "optimizer": optimizer.state_dict(),
            }, best_path)
            logger.info("  >> New best! mIoU=%.2f at epoch %d, saved %s", best_miou, epoch, best_path)

    logger.info("=" * 60)
    logger.info("QAT finetune done. Best mIoU=%.2f at epoch %d", best_miou, best_epoch)
    # ===== 7. convert to INT8 =====
    logger.info("Converting best QAT model to INT8 ...")
    # reload best checkpoint
    best_ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    model.cpu().eval()
    quant.convert(model, inplace=True)

    q_count = sum(1 for _, m in model.named_modules() if "quantized" in type(m).__module__)
    logger.info("Quantized modules: %d", q_count)

    # ===== 8. final eval (fake-quant on GPU was already best_miou) =====
    logger.info("=" * 60)
    qat_miou = best_miou
    logger.info("mIoU:  FP32=%.2f  ->  QAT-INT8=%.2f  (delta=%+.2f)",
                fp32_miou, qat_miou, qat_miou - fp32_miou)
    logger.info("OA:    FP32=%.2f", fp32_oa)
    logger.info("mAcc:  FP32=%.2f", fp32_macc)

    # ===== 9. save final INT8 model =====
    save_path = os.path.join(OUTPUT_DIR, "pointnext_s_simple128_qat_int8.pth")
    torch.save({
        "model": model.state_dict(),
        "config": CFG_PATH,
        "backend": BACKEND,
        "qat_epochs": NUM_FINETUNE_EPOCHS,
        "best_epoch": best_epoch,
        "quantized_modules": q_count,
        "fp32_miou": fp32_miou, "fp32_oa": fp32_oa, "fp32_macc": fp32_macc,
        "qat_miou": qat_miou,
    }, save_path)
    logger.info("Saved INT8 model: %s", save_path)
    logger.info("All done!")


if __name__ == "__main__":
    main()
