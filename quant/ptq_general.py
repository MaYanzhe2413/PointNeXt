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

import argparse
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
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


class QATConv1d(nn.Conv1d):
    """PyTorch 1.10-compatible QAT Conv1d.

    PyTorch 1.10.1 has no Conv1d entry in its default QAT module mapping.  A
    normal Conv1d therefore receives an activation observer during
    ``prepare_qat`` but continues to use its FP32 weight.  This module mirrors
    ``torch.nn.qat.Conv2d`` and applies ``weight_fake_quant`` in ``forward``.
    """

    _FLOAT_MODULE = nn.Conv1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", qconfig=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, **factory_kwargs,
        )
        if qconfig is None:
            raise ValueError("QATConv1d requires a valid qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, inputs):
        quantized_weight = self.weight_fake_quant(self.weight)
        return self._conv_forward(inputs, quantized_weight, self.bias)

    @classmethod
    def from_float(cls, module):
        if type(module) is not cls._FLOAT_MODULE:
            raise TypeError("QATConv1d.from_float requires an nn.Conv1d")
        if not getattr(module, "qconfig", None):
            raise ValueError("Input Conv1d must have a valid qconfig")
        qat_conv = cls(
            module.in_channels, module.out_channels, module.kernel_size,
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups,
            bias=module.bias is not None, padding_mode=module.padding_mode,
            qconfig=module.qconfig, device=module.weight.device,
            dtype=module.weight.dtype,
        )
        qat_conv.weight = module.weight
        qat_conv.bias = module.bias
        qat_conv.train(module.training)
        return qat_conv


class QATConvReLU1d(QATConv1d, nni._FusedModule):
    """QAT equivalent of the eval-fused ``nni.ConvReLU1d`` module."""

    _FLOAT_MODULE = nni.ConvReLU1d

    def forward(self, inputs):
        quantized_weight = self.weight_fake_quant(self.weight)
        return F.relu(self._conv_forward(inputs, quantized_weight, self.bias))

    @classmethod
    def from_float(cls, module):
        if type(module) is not cls._FLOAT_MODULE:
            raise TypeError(
                "QATConvReLU1d.from_float requires an nni.ConvReLU1d"
            )
        if not getattr(module, "qconfig", None):
            raise ValueError("Input ConvReLU1d must have a valid qconfig")
        conv = module[0]
        qat_conv = cls(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=conv.bias is not None,
            padding_mode=conv.padding_mode, qconfig=module.qconfig,
            device=conv.weight.device, dtype=conv.weight.dtype,
        )
        qat_conv.weight = conv.weight
        qat_conv.bias = conv.bias
        qat_conv.train(module.training)
        return qat_conv


QUANTIZED_OPERATOR_TYPES = (nn.Conv1d, nn.Conv2d, nn.Linear)
PER_CHANNEL_QSCHEMES = (torch.per_channel_affine, torch.per_channel_symmetric)
PER_TENSOR_QSCHEMES = (torch.per_tensor_affine, torch.per_tensor_symmetric)


def get_qat_module_mapping():
    mapping = dict(quant.get_default_qat_module_mappings())
    mapping[nn.Conv1d] = QATConv1d
    mapping[nni.ConvReLU1d] = QATConvReLU1d
    return mapping


def get_hardware_qat_qconfig(backend):
    """Return the QAT config required by KDPoint's uint8 activation ABI."""
    default = quant.get_default_qat_qconfig(backend)
    activation = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    return quant.QConfig(activation=activation, weight=default.weight)


def _qualified_type(module):
    return f"{type(module).__module__}.{type(module).__name__}"


def _fake_quant_error(fake_quant, role):
    if not isinstance(fake_quant, quant.FakeQuantize):
        return f"{role} is not a FakeQuantize module"
    if role == "weight":
        if fake_quant.dtype != torch.qint8:
            return f"weight dtype is {fake_quant.dtype}, expected torch.qint8"
        if fake_quant.qscheme not in PER_CHANNEL_QSCHEMES:
            return f"weight qscheme is {fake_quant.qscheme}, expected per-channel"
        if getattr(fake_quant, "ch_axis", None) != 0:
            return f"weight ch_axis is {getattr(fake_quant, 'ch_axis', None)}, expected 0"
    else:
        if fake_quant.dtype != torch.quint8:
            return f"activation dtype is {fake_quant.dtype}, expected torch.quint8"
        if fake_quant.qscheme not in PER_TENSOR_QSCHEMES:
            return f"activation qscheme is {fake_quant.qscheme}, expected per-tensor"
        if fake_quant.quant_min != 0 or fake_quant.quant_max != 255:
            return (f"activation range is [{fake_quant.quant_min}, "
                    f"{fake_quant.quant_max}], expected [0, 255]")
    return None


def audit_quantized_operators(model, stage="after prepare_qat"):
    """Require real QAT weight and activation fake-quant on every compute op."""
    operators = []
    errors = []
    for name, module in model.named_modules():
        if not isinstance(module, QUANTIZED_OPERATOR_TYPES):
            continue
        weight_fake_quant = getattr(module, "weight_fake_quant", None)
        activation_fake_quant = getattr(module, "activation_post_process", None)
        entry = {
            "name": name,
            "type": _qualified_type(module),
            "weight_shape": list(module.weight.shape),
        }
        operators.append((entry, module))

        if isinstance(module, nn.Conv1d) and not isinstance(module, QATConv1d):
            errors.append(f"{name}: Conv1d was not converted to QATConv1d")
        weight_error = _fake_quant_error(weight_fake_quant, "weight")
        if weight_error:
            errors.append(f"{name}: {weight_error}")
        activation_error = _fake_quant_error(
            activation_fake_quant, "activation"
        )
        if activation_error:
            errors.append(f"{name}: {activation_error}")

    if not operators:
        errors.append("model has no Conv1d/Conv2d/Linear operators")
    if errors:
        details = "\n  ".join(errors)
        raise RuntimeError(f"Quantization audit failed at {stage}:\n  {details}")
    logger.info(
        "[QUANT AUDIT][PASS] %s: %d Conv1d/Conv2d/Linear operators have "
        "per-channel qint8 weight fake-quant and per-tensor uint8 activation "
        "fake-quant", stage, len(operators),
    )
    return operators


def _tensor_json(tensor):
    return tensor.detach().cpu().tolist()


def _fake_quant_qparams(fake_quant):
    scale, zero_point = fake_quant.calculate_qparams()
    return {
        "type": _qualified_type(fake_quant),
        "dtype": str(fake_quant.dtype),
        "qscheme": str(fake_quant.qscheme),
        "quant_min": fake_quant.quant_min,
        "quant_max": fake_quant.quant_max,
        "ch_axis": getattr(fake_quant, "ch_axis", None),
        "scale": _tensor_json(scale),
        "zero_point": _tensor_json(zero_point),
    }


def collect_operator_qparams(operators, forward_calls):
    manifest = []
    for entry, module in operators:
        item = dict(entry)
        item["weight_fake_quant"] = _fake_quant_qparams(
            module.weight_fake_quant
        )
        item["activation_fake_quant"] = _fake_quant_qparams(
            module.activation_post_process
        )
        item["weight_fake_quant_forward_calls"] = forward_calls[entry["name"]]
        manifest.append(item)
    return manifest


def register_weight_fake_quant_usage(operators):
    calls = {entry["name"]: 0 for entry, _ in operators}
    handles = []
    for entry, module in operators:
        name = entry["name"]

        def count_call(_module, _inputs, _output, op_name=name):
            calls[op_name] += 1

        handles.append(module.weight_fake_quant.register_forward_hook(count_call))
    return calls, handles


def assert_weight_fake_quant_used(calls):
    unused = [name for name, count in calls.items() if count == 0]
    if unused:
        raise RuntimeError(
            "Weight fake-quant was not executed for: " + ", ".join(unused)
        )
    logger.info(
        "[QUANT AUDIT][PASS] weight fake-quant executed for all %d operators",
        len(calls),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_tree(root):
    paths = [os.path.join(root, "quant", "ptq_general.py")]
    source_root = os.path.join(root, "openpoints")
    for directory, _, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.endswith(".py"):
                paths.append(os.path.join(directory, filename))
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(paths)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def resolved_config_sha256(cfg):
    payload = json.dumps(
        _jsonable(dict(cfg)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_metadata(root):
    def run_git(*args):
        return subprocess.check_output(
            ["git", "-C", root, *args], text=True
        ).strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "status": run_git("status", "--short"),
    }


def cpu_state_dict(model):
    state = model.state_dict()
    cpu_state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in state.items()
    )
    if hasattr(state, "_metadata"):
        cpu_state._metadata = state._metadata
    return cpu_state


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
    return {"primary": miou, "miou": miou, "oa": oa, "macc": macc}


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
    parser.add_argument(
        "--baseline-json", default=None,
        help="authoritative prior run; cfg/checkpoint/calibration/options must match",
    )
    args, opts = parser.parse_known_args()   # extra opts -> cfg.update (sampler / data_root)

    tag = args.tag or (
        os.path.splitext(os.path.basename(args.cfg))[0] + "_" + args.task
    )
    artifact_prefix = os.path.join(args.out, f"ptq_{tag}")
    out_path = artifact_prefix + ".json"
    state_path = artifact_prefix + "_calibrated.pth"
    qparams_path = artifact_prefix + "_qparams.json"
    for path in (out_path, state_path, qparams_path):
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    os.makedirs(args.out, exist_ok=True)
    eval_fn = EVALUATORS[args.task]

    # ----- build config (+ apply opts: sampler override, data_root, ...) -----
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    if opts:
        cfg.update(opts)
        logger.info("Applied cfg opts: %s", opts)

    baseline = None
    if args.baseline_json:
        with open(args.baseline_json) as stream:
            baseline = json.load(stream)
        expected = {
            "task": args.task,
            "cfg": args.cfg,
            "ckpt": args.ckpt,
            "backend": args.backend,
            "calib_batches": args.calib_batches,
            "opts": opts,
        }
        mismatches = [
            f"{key}: baseline={baseline.get(key)!r}, run={value!r}"
            for key, value in expected.items()
            if baseline.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Run does not match authoritative baseline:\n  "
                + "\n  ".join(mismatches)
            )

    source_tree_sha256, source_file_count = sha256_source_tree(ROOT)
    provenance = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": os.getcwd(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "git": git_metadata(ROOT),
        "sha256": {
            "source_tree": source_tree_sha256,
            "source_file_count": source_file_count,
            "ptq_general.py": sha256_file(__file__),
            "config_file": sha256_file(args.cfg),
            "resolved_config": resolved_config_sha256(cfg),
            "checkpoint": sha256_file(args.ckpt),
            "baseline_json": (
                sha256_file(args.baseline_json) if args.baseline_json else None
            ),
        },
    }

    logger.info("Task=%s  Cfg=%s", args.task, args.cfg)
    model = build_model_from_cfg(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict checkpoint load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    logger.info(
        "[CHECKPOINT][PASS] strict=True missing=[] unexpected=[] (epoch=%s)",
        ckpt.get("epoch", "?"),
    )
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
    model.qconfig = get_hardware_qat_qconfig(args.backend)

    logger.info("[Step 4] prepare_qat")
    model.train()
    quant.prepare_qat(model, mapping=get_qat_module_mapping(), inplace=True)
    print_fakequant_status(model, tag="after prepare_qat")
    operators = audit_quantized_operators(model)
    for entry, _ in operators:
        logger.info(
            "[QUANT OP] %s | %s | weight_shape=%s",
            entry["name"], entry["type"], entry["weight_shape"],
        )
    forward_calls, usage_handles = register_weight_fake_quant_usage(operators)

    # ----- Phase 1: calibrate (observers only) -----
    logger.info("[Phase 1] calibrate %d batches", args.calib_batches)
    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)
    model.eval().cuda()
    fk = cfg.get("feature_keys", "pos")
    calibrated_batches = 0
    try:
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if i >= args.calib_batches:
                    break
                data = _to_cuda(data)
                data["x"] = get_features_by_keys(data, fk)
                model(data)
                calibrated_batches += 1
    finally:
        for handle in usage_handles:
            handle.remove()
    if calibrated_batches != args.calib_batches:
        raise RuntimeError(
            f"Requested {args.calib_batches} calibration batches, "
            f"but loader produced {calibrated_batches}"
        )
    logger.info("[CALIBRATION][PASS] consumed exactly %d batches", calibrated_batches)
    assert_weight_fake_quant_used(forward_calls)

    # ----- Phase 2: INT8-sim eval -----
    logger.info("[Phase 2] INT8-sim eval")
    model.apply(quant.enable_fake_quant)
    model.apply(quant.disable_observer)
    operators = audit_quantized_operators(model, stage="after calibration")
    operator_qparams = collect_operator_qparams(operators, forward_calls)

    artifact_metadata = {
        "format_version": 1,
        "task": args.task,
        "cfg": args.cfg,
        "ckpt": args.ckpt,
        "backend": args.backend,
        "calib_batches": calibrated_batches,
        "opts": opts,
        "baseline_json": args.baseline_json,
        "provenance": provenance,
    }
    qparams_payload = {
        "metadata": artifact_metadata,
        "quantized_operators": operator_qparams,
    }
    with open(qparams_path, "x") as stream:
        json.dump(qparams_payload, stream, indent=2)
    with open(state_path, "xb") as stream:
        torch.save({
            "metadata": artifact_metadata,
            "model_state_dict": cpu_state_dict(model),
            "quantized_operators": operator_qparams,
        }, stream)
    artifact_hashes = {
        "calibrated_state": {
            "path": state_path,
            "sha256": sha256_file(state_path),
        },
        "qparams": {
            "path": qparams_path,
            "sha256": sha256_file(qparams_path),
        },
    }
    logger.info("Saved calibrated state: %s", state_path)
    logger.info("Saved calibrated qparams: %s", qparams_path)

    res_int8 = eval_fn(model, val_loader, cfg, tag="INT8-sim")

    # ----- summary -----
    p0, pf, pi = res_fp32["primary"], res_fused["primary"], res_int8["primary"]
    logger.info("=" * 60)
    logger.info("SUMMARY (task=%s)", args.task)
    logger.info("  FP32        = %.2f", p0)
    logger.info("  FP32-fused  = %.2f   (BN-fold delta %+.2f)", pf, pf - p0)
    logger.info("  INT8-sim    = %.2f   (quant delta %+.2f, total %+.2f)", pi, pi - pf, pi - p0)

    baseline_metrics = None
    comparison = None
    if baseline:
        baseline_metrics = {
            key: baseline[key]
            for key in ("fp32", "fused", "int8", "primary_delta")
        }
        comparison = {
            "fp32_primary": p0 - baseline["fp32"]["primary"],
            "fused_primary": pf - baseline["fused"]["primary"],
            "int8_primary": pi - baseline["int8"]["primary"],
            "primary_delta": (pi - p0) - baseline["primary_delta"],
        }
        logger.info(
            "  Previous INT8-sim = %.2f   (change %+.2f)",
            baseline["int8"]["primary"], comparison["int8_primary"],
        )

    with open(out_path, "x") as stream:
        json.dump({
            "task": args.task, "cfg": args.cfg, "ckpt": args.ckpt,
            "backend": args.backend, "calib_batches": args.calib_batches,
            "opts": opts,
            "fp32": res_fp32, "fused": res_fused, "int8": res_int8,
            "primary_delta": pi - p0,
            "baseline": {
                "record": args.baseline_json,
                "metrics": baseline_metrics,
            } if baseline else None,
            "comparison_to_baseline": comparison,
            "checkpoint_load": {"strict": True, "missing": [], "unexpected": []},
            "quantized_operators": operator_qparams,
            "artifacts": artifact_hashes,
            "provenance": provenance,
            "resolved_config": _jsonable(dict(cfg)),
        }, stream, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
