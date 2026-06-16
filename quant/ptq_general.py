"""
Generalized PTQ (Post-Training Quantization) for PointNeXt / PointNet++ across
segmentation (S3DIS), classification (ModelNet40), and part-seg (ShapeNet).

Two-phase eager-mode PTQ (same flow as quant/quant_wrapper.py):
  Phase 1: prepare_qat -> disable fake_quant, enable observer -> calibrate
  Phase 2: enable fake_quant, disable observer -> eval (true INT8-sim accuracy)

Supports stacking on block-FPS: pass sampler overrides via plain cfg opts, e.g.
  model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=1500 ...

Usage:
  python quant/ptq_general.py --task seg \
      --cfg cfgs/s3dis/pointnet++.yaml \
      --ckpt log/s3dis/.../ckpt_best.pth \
      [model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=1500 \
       model.encoder_args.sampler_args.strategy=fps \
       dataset.common.data_root=/path/to/data]

  --task: seg | cls | partseg
  extra positional opts after the flags are passed to cfg.update() (sampler / data_root / etc.)
"""

import os, sys, time, json, argparse, logging
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


# ----------------------------------------------------------------------------
# Task-specific evaluators (all call model(data); only the metric differs)
# ----------------------------------------------------------------------------
def _to_cuda(data):
    for k in data:
        data[k] = data[k].cuda(non_blocking=True)
    return data


def evaluate_seg(model, loader, cfg, tag="FP32"):
    model.eval().cuda()
    fk = cfg.get("feature_keys", "pos")
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.get("ignore_index", None))
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            data = _to_cuda(data)
            data["x"] = get_features_by_keys(data, fk)
            logits = model(data)
            cm.update(logits.argmax(dim=1), data["y"].squeeze(-1))
    miou, macc, oa, ious, _ = get_mious(cm.tp, cm.union, cm.count)
    logger.info("[%s] mIoU=%.2f  OA=%.2f  mAcc=%.2f  (%.1fs)", tag, miou, oa, macc, time.time() - t0)
    return {"primary": miou, "miou": miou, "oa": oa, "macc": macc,
            "ious": [round(float(x), 2) for x in ious]}


def evaluate_cls(model, loader, cfg, tag="FP32"):
    model.eval().cuda()
    fk = cfg.get("feature_keys", "pos")
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            data = _to_cuda(data)
            data["x"] = get_features_by_keys(data, fk)
            logits = model(data)
            target = data["y"].flatten()
            cm.update(logits.argmax(dim=1).flatten(), target)
    macc, oa, _ = ConfusionMatrix.cal_acc(cm.tp, cm.count)
    logger.info("[%s] OA=%.2f  mAcc=%.2f  (%.1fs)", tag, oa, macc, time.time() - t0)
    return {"primary": oa, "oa": oa, "macc": macc}


def _get_ins_mious(pred, target, cls, cls2parts):
    """Instance-wise mIoU for part-seg (mirrors examples/shapenetpart/main.py)."""
    ins_mious = []
    for shape_idx in range(pred.shape[0]):
        cur_cls = cls[shape_idx][0]
        parts = cls2parts[cur_cls]
        part_ious = []
        for part in parts:
            pred_p = (pred[shape_idx] == part)
            target_p = (target[shape_idx] == part)
            I = torch.logical_and(pred_p, target_p).sum()
            U = torch.logical_or(pred_p, target_p).sum()
            iou = (I / U) if U > 0 else torch.tensor(1.0, device=pred.device)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


def evaluate_partseg(model, loader, cfg, tag="FP32"):
    model.eval().cuda()
    fk = cfg.get("feature_keys", "pos")
    cls2parts = cfg.get("cls2parts", None)
    if cls2parts is None and hasattr(loader.dataset, "cls2parts"):
        cls2parts = loader.dataset.cls2parts
    if cls2parts is None:
        raise RuntimeError("partseg eval needs cls2parts (from cfg or dataset)")
    ins_miou_list = []
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            data = _to_cuda(data)
            data["x"] = get_features_by_keys(data, fk)
            logits = model(data)                 # [B, num_parts, N]
            preds = logits.argmax(dim=1)         # [B, N]
            ins_miou_list += _get_ins_mious(preds, data["y"], data["cls"], cls2parts)
    ins_miou = (torch.sum(torch.stack(ins_miou_list)) / len(ins_miou_list)).item() * 100
    logger.info("[%s] Instance mIoU=%.2f  (%.1fs)", tag, ins_miou, time.time() - t0)
    return {"primary": ins_miou, "ins_miou": ins_miou}


EVALUATORS = {"seg": evaluate_seg, "cls": evaluate_cls, "partseg": evaluate_partseg}


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Generalized PTQ")
    parser.add_argument("--task", required=True, choices=["seg", "cls", "partseg"])
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backend", default="fbgemm")
    parser.add_argument("--calib-batches", type=int, default=30)
    parser.add_argument("--out", default="quant/output")
    parser.add_argument("--tag", default=None, help="label for output file")
    parser.add_argument("--quant-mode", default="w8a8", choices=["w8a8", "w8a32", "w32a8"],
                        help="w8a8=both, w8a32=weights only, w32a8=activations only (diagnostic)")
    args, opts = parser.parse_known_args()   # extra opts -> cfg.update (sampler / data_root)

    os.makedirs(args.out, exist_ok=True)
    eval_fn = EVALUATORS[args.task]

    # ----- build config (+ apply opts: sampler override, data_root, ...) -----
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    if opts:
        cfg.update(opts)
        logger.info("Applied cfg opts: %s", opts)

    logger.info("Task=%s  Cfg=%s", args.task, args.cfg)
    model = build_model_from_cfg(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    logger.info("Loaded ckpt (epoch=%s)", ckpt.get("epoch", "?"))
    model.eval()

    # ----- val dataloader -----
    split = "val"
    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split=split, distributed=False,
    )
    logger.info("Val set: %d samples", len(val_loader.dataset))

    # ----- FP32 baseline -----
    logger.info("=" * 60)
    res_fp32 = eval_fn(model, val_loader, cfg, tag="FP32")

    # ----- quant prep -----
    model.cpu().eval()
    logger.info("[Step 0] swap custom Conv -> standard")
    swap_custom_convs_to_standard(model)
    logger.info("[Step 1] fuse Conv-BN-ReLU")
    fuse_convbn_modules(model)

    logger.info("[Step 1b] eval FP32-fused (BN-folding only)")
    res_fused = eval_fn(model, val_loader, cfg, tag="FP32-fused")
    model.cpu().eval()

    logger.info("[Step 2] disable quant for geometry ops")
    disable_quantization_for_geometry(model)

    logger.info("[Step 3] set QAT qconfig (backend=%s)", args.backend)
    torch.backends.quantized.engine = args.backend
    model.qconfig = quant.get_default_qat_qconfig(args.backend)

    logger.info("[Step 4] prepare_qat")
    model.train()
    quant.prepare_qat(model, inplace=True)
    print_fakequant_status(model, tag="after prepare_qat")

    # ----- Phase 1: calibrate (observers only) -----
    logger.info("[Phase 1] calibrate %d batches", args.calib_batches)
    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)
    model.eval().cuda()
    fk = cfg.get("feature_keys", "pos")
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= args.calib_batches:
                break
            data = _to_cuda(data)
            data["x"] = get_features_by_keys(data, fk)
            model(data)

    # ----- Phase 2: INT8-sim eval -----
    logger.info("[Phase 2] INT8-sim eval (mode=%s)", args.quant_mode)
    model.apply(quant.enable_fake_quant)
    model.apply(quant.disable_observer)
    # W/A decomposition: selectively disable one fake-quant group (diagnostic)
    if args.quant_mode != "w8a8":
        from torch.quantization import FakeQuantize
        n_w, n_a = 0, 0
        for name, m in model.named_modules():
            if isinstance(m, FakeQuantize):
                is_weight = "weight_fake_quant" in name
                if args.quant_mode == "w8a32" and not is_weight:
                    m.disable_fake_quant(); n_a += 1     # keep weights, drop activations
                elif args.quant_mode == "w32a8" and is_weight:
                    m.disable_fake_quant(); n_w += 1     # keep activations, drop weights
        logger.info("  [%s] disabled fake_quant on %d weight + %d act modules",
                    args.quant_mode, n_w, n_a)
    res_int8 = eval_fn(model, val_loader, cfg, tag="INT8-sim")

    # ----- summary -----
    p0, pf, pi = res_fp32["primary"], res_fused["primary"], res_int8["primary"]
    logger.info("=" * 60)
    logger.info("SUMMARY (task=%s)", args.task)
    logger.info("  FP32        = %.2f", p0)
    logger.info("  FP32-fused  = %.2f   (BN-fold delta %+.2f)", pf, pf - p0)
    logger.info("  INT8-sim    = %.2f   (quant delta %+.2f, total %+.2f)  mode=%s",
                pi, pi - pf, pi - p0, args.quant_mode)
    # per-class IoU delta for seg (which classes drive the drop)
    if args.task == "seg" and "ious" in res_fp32 and "ious" in res_int8:
        deltas = [(i, a, b, round(b - a, 2))
                  for i, (a, b) in enumerate(zip(res_fp32["ious"], res_int8["ious"]))]
        worst = sorted(deltas, key=lambda x: x[3])[:5]
        logger.info("  per-class IoU (FP32->INT8), 5 worst drops:")
        for i, a, b, d in worst:
            logger.info("    class %2d: %.2f -> %.2f  (%+.2f)", i, a, b, d)

    tag = args.tag or (os.path.splitext(os.path.basename(args.cfg))[0] + "_" + args.task)
    if args.quant_mode != "w8a8":
        tag = f"{tag}_{args.quant_mode}"
    out_path = os.path.join(args.out, f"ptq_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": args.task, "cfg": args.cfg, "ckpt": args.ckpt,
            "backend": args.backend, "calib_batches": args.calib_batches,
            "opts": opts,
            "fp32": res_fp32, "fused": res_fused, "int8": res_int8,
            "primary_delta": pi - p0,
        }, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
