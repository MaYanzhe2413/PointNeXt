"""
Compare FP32 vs INT8 weights and activations for a specific layer.
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.quantization as quant
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.utils import EasyConfig
from openpoints.models.layers.quant_utils import (
    swap_custom_convs_to_standard,
    fuse_convbn_modules,
    disable_quantization_for_geometry,
)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

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

TARGET_LAYERS = [
    "encoder.encoder.0.0.convs.0.0",   # Conv1d [32, 4, 1]
    "encoder.encoder.2.0.convs.1.0",   # Conv2d [128, 64, 1, 1]
]

def get_module(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def pstats(tag, t):
    a = t.detach().float().cpu().numpy()
    print(f"  [{tag}] shape={list(t.shape)} dtype={t.dtype}")
    print(f"    min={a.min():.6f}  max={a.max():.6f}  mean={a.mean():.6f}  std={a.std():.6f}")


def psnippet(tag, w, r=8, c=8):
    flat = w.detach().float().cpu().reshape(w.shape[0], -1).numpy()
    r, c = min(r, flat.shape[0]), min(c, flat.shape[1])
    print(f"  [{tag}] weight[:{r}, :{c}]:")
    print(flat[:r, :c])


def act_snippet(tag, act, r=8, c=8):
    a = act.detach().float().cpu()
    a = a.reshape(a.shape[0], a.shape[1], -1)[0]  # first sample
    r, c = min(r, a.shape[0]), min(c, a.shape[1])
    print(f"  [{tag}] activation[0, :{r}, :{c}]:")
    print(a[:r, :c].numpy())

LOG_DIR = "quant/output"


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w")
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"compare_weights_acts_{timestamp}.log")
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"Log file: {log_path}")

    cfg = EasyConfig()
    cfg.load(CFG_PATH, recursive=True)
    model = build_model_from_cfg(cfg.model)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.get("batch_size", 1)),
        cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val", distributed=False,
    )
    data = next(iter(val_loader))

    # ========== FP32 ==========
    print("\n" + "=" * 80)
    print("  FP32 MODEL -- Weights & Activations")
    print("=" * 80)
    model.cuda()
    for k in data:
        data[k] = data[k].cuda()
    data["x"] = get_features_by_keys(data, cfg.get("feature_keys", "pos"))

    fp32_w = {}
    fp32_a = {}
    hooks = []
    for ln in TARGET_LAYERS:
        m = get_module(model, ln)
        fp32_w[ln] = m.weight.detach().clone()
        def _h(n):
            def fn(mod, inp, out):
                fp32_a[n] = out.detach().clone()
            return fn
        hooks.append(m.register_forward_hook(_h(ln)))

    with torch.no_grad():
        model(data)
    for h in hooks:
        h.remove()

    for ln in TARGET_LAYERS:
        print(f"\n--- {ln} ---")
        pstats("FP32 weight", fp32_w[ln])
        psnippet("FP32", fp32_w[ln])
        if ln in fp32_a:
            pstats("FP32 activation", fp32_a[ln])
            act_snippet("FP32", fp32_a[ln])
    # ========== Quantize ==========
    model.cpu().eval()
    swap_custom_convs_to_standard(model)
    fuse_convbn_modules(model)
    disable_quantization_for_geometry(model)
    torch.backends.quantized.engine = "fbgemm"
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)

    # calibrate
    model.cuda()
    with torch.no_grad():
        for i, d in enumerate(val_loader):
            if i >= 10:
                break
            for k in d:
                d[k] = d[k].cuda()
            d["x"] = get_features_by_keys(d, cfg.get("feature_keys", "pos"))
            model(d)
    print("\nCalibration done (10 batches)")

    # ---- fake-quant activation ----
    print("\n" + "=" * 80)
    print("  FAKE-QUANT MODEL -- Activations (simulated INT8)")
    print("=" * 80)
    fq_a = {}
    hooks = []
    for ln in TARGET_LAYERS:
        m = get_module(model, ln)
        def _h(n):
            def fn(mod, inp, out):
                fq_a[n] = out.detach().clone()
            return fn
        hooks.append(m.register_forward_hook(_h(ln)))

    data2 = next(iter(val_loader))
    for k in data2:
        data2[k] = data2[k].cuda()
    data2["x"] = get_features_by_keys(data2, cfg.get("feature_keys", "pos"))
    with torch.no_grad():
        model(data2)
    for h in hooks:
        h.remove()

    for ln in TARGET_LAYERS:
        if ln in fq_a:
            print(f"\n--- {ln} ---")
            pstats("FakeQuant activation", fq_a[ln])
            act_snippet("FakeQuant", fq_a[ln])
    # ========== Convert to INT8 & inspect weights ==========
    print("\n" + "=" * 80)
    print("  INT8 CONVERTED -- Weights")
    print("=" * 80)
    model.cpu().eval()
    quant.convert(model, inplace=True)

    for ln in TARGET_LAYERS:
        print(f"\n--- {ln} ---")
        m = get_module(model, ln)
        print(f"  Module type: {type(m).__name__}")
        if hasattr(m, "weight"):
            w = m.weight()
            print(f"  Quantized weight dtype: {w.dtype}")
            if w.dtype == torch.qint8:
                ir = w.int_repr()
                if w.qscheme() == torch.per_channel_affine:
                    sc = w.q_per_channel_scales()
                    zp = w.q_per_channel_zero_points()
                else:
                    sc = torch.tensor([w.q_scale()])
                    zp = torch.tensor([w.q_zero_point()])
                dq = w.dequantize()

                print(f"  q_scheme: {w.qscheme()}")
                print(f"  scale: shape={list(sc.shape)}, min={sc.min():.6f}, max={sc.max():.6f}")
                print(f"  zero_point: unique={zp.unique().tolist()}")

                pstats("INT8 int_repr", ir.float())
                flat_ir = ir.reshape(ir.shape[0], -1).numpy()
                r, c = min(8, flat_ir.shape[0]), min(8, flat_ir.shape[1])
                print(f"  [INT8] int_repr[:{r}, :{c}]:")
                print(flat_ir[:r, :c])

                pstats("INT8 dequantized", dq)
                psnippet("INT8 dequant", dq)

                fp = fp32_w[ln].cpu()
                diff = (dq - fp).abs()
                rel = diff.mean() / fp.abs().mean() if fp.abs().mean() > 0 else 0
                print(f"\n  ** Weight diff (FP32 vs INT8-dequant) **")
                print(f"    abs: mean={diff.mean():.6f}  max={diff.max():.6f}  relative={rel:.4f}")
    # ========== Activation comparison ==========
    print("\n" + "=" * 80)
    print("  ACTIVATION COMPARISON: FP32 vs FakeQuant")
    print("=" * 80)
    for ln in TARGET_LAYERS:
        if ln in fp32_a and ln in fq_a:
            print(f"\n--- {ln} ---")
            fp = fp32_a[ln].cpu().float()
            fq = fq_a[ln].cpu().float()
            diff = (fp - fq).abs()
            print(f"  abs diff: mean={diff.mean():.6f}  max={diff.max():.6f}")
            if fp.abs().mean() > 0:
                print(f"  relative mean diff: {diff.mean() / fp.abs().mean():.6f}")

    print("\n" + "=" * 80)
    print("Done!")
    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
