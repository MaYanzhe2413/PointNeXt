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
from openpoints.models.layers.group import GroupAll, QueryAndGroup

try:
    from .kdpoint_requant import q31_multiplier, rtl_requant
except ImportError:
    from kdpoint_requant import q31_multiplier, rtl_requant

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
PER_TENSOR_QSCHEMES = (torch.per_tensor_affine, torch.per_tensor_symmetric)


def get_qat_module_mapping():
    mapping = dict(quant.get_default_qat_module_mappings())
    mapping[nn.Conv1d] = QATConv1d
    mapping[nni.ConvReLU1d] = QATConvReLU1d
    return mapping


def get_hardware_qat_qconfig(backend):
    """Return the QAT config required by KDPoint's uint8 activation ABI."""
    activation = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    # PyTorch 1.10's default fused weight fake-quant can leave its runtime
    # scale/zero_point buffers inconsistent with the symmetric observer.  The
    # RTL consumes those runtime buffers, so use the non-fused implementation.
    weight = quant.FakeQuantize.with_args(
        observer=quant.MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        reduce_range=False,
    )
    return quant.QConfig(activation=activation, weight=weight)


def _qualified_type(module):
    return f"{type(module).__module__}.{type(module).__name__}"


def _fake_quant_error(fake_quant, role):
    if not isinstance(fake_quant, quant.FakeQuantize):
        return f"{role} is not a FakeQuantize module"
    if role == "weight":
        if fake_quant.dtype != torch.qint8:
            return f"weight dtype is {fake_quant.dtype}, expected torch.qint8"
        if fake_quant.qscheme != torch.per_channel_symmetric:
            return (f"weight qscheme is {fake_quant.qscheme}, expected "
                    "torch.per_channel_symmetric")
        if fake_quant.quant_min != -128 or fake_quant.quant_max != 127:
            return (f"weight range is [{fake_quant.quant_min}, "
                    f"{fake_quant.quant_max}], expected [-128, 127]")
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


def _fake_quant_enabled_error(fake_quant, role):
    fake_quant_enabled = getattr(fake_quant, "fake_quant_enabled", None)
    observer_enabled = getattr(fake_quant, "observer_enabled", None)
    if fake_quant_enabled is None or fake_quant_enabled.numel() != 1:
        return f"{role} fake_quant_enabled is malformed"
    if observer_enabled is None or observer_enabled.numel() != 1:
        return f"{role} observer_enabled is malformed"
    if not bool(fake_quant_enabled.detach().cpu().item()):
        return f"{role} fake quant is disabled"
    if bool(observer_enabled.detach().cpu().item()):
        return f"{role} observer is still enabled"
    return None


def _calibrated_weight_buffer_error(fake_quant, output_channels):
    state_error = _fake_quant_enabled_error(fake_quant, "weight")
    if state_error:
        return state_error

    scale = fake_quant.scale.detach()
    zero_point = fake_quant.zero_point.detach()
    if scale.numel() != output_channels:
        return (f"weight scale buffer has {scale.numel()} values, expected "
                f"one per output channel ({output_channels})")
    if zero_point.numel() != output_channels:
        return (f"weight zero-point buffer has {zero_point.numel()} values, "
                f"expected one per output channel ({output_channels})")
    if not bool(torch.isfinite(scale).all().detach().cpu().item()):
        return "weight scale buffer contains non-finite values"
    if not bool((scale > 0).all().detach().cpu().item()):
        return "weight scale buffer contains non-positive values"
    zero_point_i64 = zero_point.to(dtype=torch.int64)
    if not bool((zero_point_i64 == 0).all().detach().cpu().item()):
        return "weight zero-point buffer is not all zero"

    calculated_scale, calculated_zero_point = fake_quant.calculate_qparams()
    calculated_scale = calculated_scale.to(
        device=scale.device, dtype=scale.dtype
    )
    calculated_zero_point = calculated_zero_point.to(
        device=zero_point.device, dtype=torch.int64
    )
    if scale.shape != calculated_scale.shape:
        return (
            "weight scale buffer/observer shape mismatch: "
            f"buffer={list(scale.shape)}, observer={list(calculated_scale.shape)}"
        )
    # PyTorch 1.10 can recompute the observer scale one FP32 ULP below the
    # serialized FakeQuantize buffer. The buffer drives forward/export, so
    # allow exactly that round-trip effect but reject any wider drift.
    lower = torch.nextafter(
        calculated_scale, torch.full_like(calculated_scale, -float("inf"))
    )
    upper = torch.nextafter(
        calculated_scale, torch.full_like(calculated_scale, float("inf"))
    )
    within_one_ulp = (scale >= lower) & (scale <= upper)
    if not bool(within_one_ulp.all().detach().cpu().item()):
        delta = (scale - calculated_scale).abs()
        index = int(delta.argmax().detach().cpu().item())
        denominator = calculated_scale[index].abs().clamp_min(
            torch.finfo(calculated_scale.dtype).tiny
        )
        relative = delta[index] / denominator
        return (
            "weight scale buffer disagrees with observer qparams: "
            f"max_abs={float(delta[index].detach().cpu().item()):.9g}, "
            f"max_rel={float(relative.detach().cpu().item()):.9g}, "
            f"index={index}, "
            f"buffer={float(scale[index].detach().cpu().item()):.9g}, "
            f"observer={float(calculated_scale[index].detach().cpu().item()):.9g}"
        )
    if zero_point_i64.shape != calculated_zero_point.shape or not torch.equal(
            zero_point_i64, calculated_zero_point):
        return "weight zero-point buffer disagrees with observer qparams"
    return None


def audit_quantized_operators(
        model, stage="after prepare_qat", require_calibrated_buffers=False):
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
        if require_calibrated_buffers and not weight_error:
            buffer_error = _calibrated_weight_buffer_error(
                weight_fake_quant, module.weight.shape[0]
            )
            if buffer_error:
                errors.append(f"{name}: {buffer_error}")
        if require_calibrated_buffers and not activation_error:
            state_error = _fake_quant_enabled_error(
                activation_fake_quant, "activation"
            )
            if state_error:
                errors.append(f"{name}: {state_error}")

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
    # These buffers, rather than a fresh observer calculation, are consumed by
    # FakeQuantize.forward and are therefore the hardware export authority.
    scale = fake_quant.scale.detach()
    zero_point = fake_quant.zero_point.detach()
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


def _activation_fake_quant(module, label):
    fake_quant = getattr(module, "activation_post_process", None)
    error = _fake_quant_error(fake_quant, "activation")
    if error:
        raise RuntimeError(f"{label}: {error}")
    return fake_quant


def _copy_activation_domain(target, source, target_name, source_name):
    """Make one activation boundary use another boundary's exact qparams."""
    target_error = _fake_quant_error(target, "activation")
    source_error = _fake_quant_error(source, "activation")
    if target_error:
        raise RuntimeError(f"{target_name}: {target_error}")
    if source_error:
        raise RuntimeError(f"{source_name}: {source_error}")
    target.load_state_dict(source.state_dict(), strict=True)
    target_scale, target_zero_point = target.calculate_qparams()
    source_scale, source_zero_point = source.calculate_qparams()
    if not torch.equal(target_scale, source_scale):
        raise RuntimeError(
            f"hardware domain lock scale mismatch: {target_name} != {source_name}"
        )
    if not torch.equal(target_zero_point, source_zero_point):
        raise RuntimeError(
            f"hardware domain lock zero-point mismatch: "
            f"{target_name} != {source_name}"
        )
    return {
        "target": target_name,
        "source": source_name,
        "scale": _tensor_json(target_scale),
        "zero_point": _tensor_json(target_zero_point),
    }


def _last_quantized_output(module, module_names, label):
    candidates = []
    for child in module.modules():
        if isinstance(child, QUANTIZED_OPERATOR_TYPES):
            candidates.append(child)
    if not candidates:
        raise RuntimeError(f"{label}: no Conv1d/Conv2d/Linear output found")
    output_module = candidates[-1]
    output_name = module_names[id(output_module)] + ".activation_post_process"
    return _activation_fake_quant(output_module, output_name), output_name


def _pointnet2_aggregation_operator(aggregation, label):
    """Unwrap PointNet++ LocalAggregation's configured implementation."""
    operator = getattr(aggregation, "SA_CONFIG_operator", None)
    if operator is None:
        raise RuntimeError(
            f"{label}: LocalAggregation is missing SA_CONFIG_operator"
        )
    missing = [
        name for name in ("quant_feat", "convs")
        if not hasattr(operator, name)
    ]
    if missing:
        raise RuntimeError(
            f"{label}: SA_CONFIG_operator is missing {missing}"
        )
    return operator


def _lock_pointnext_encoder_domains(model, module_names):
    encoder = model.encoder
    stages = list(encoder.encoder.children())
    if not stages:
        raise RuntimeError("PointNeXt encoder has no stages")
    root_fq = _activation_fake_quant(
        encoder.quant_input, "encoder.quant_input.activation_post_process"
    )
    root_name = module_names[id(encoder.quant_input)] + ".activation_post_process"
    ties = []

    stem_blocks = list(stages[0].children())
    if len(stem_blocks) != 1:
        raise RuntimeError(
            "hardware domain lock expects one PointNeXt stem block"
        )
    stem_fq, stem_name = _last_quantized_output(
        stem_blocks[0], module_names, "PointNeXt stem"
    )
    tie = _copy_activation_domain(stem_fq, root_fq, stem_name, root_name)
    tie["kind"] = "pointnext_fused_lift_shared_domain"
    ties.append(tie)
    incoming_fq, incoming_name = root_fq, root_name

    for stage_index, stage in enumerate(stages[1:], start=1):
        blocks = list(stage.children())
        if len(blocks) != 1:
            raise RuntimeError(
                "hardware domain lock only admits one block per PointNeXt stage; "
                f"stage {stage_index} has {len(blocks)}"
            )
        block = blocks[0]
        if not hasattr(block, "quant_feat"):
            raise RuntimeError(
                f"PointNeXt stage {stage_index} is missing quant_feat"
            )
        quant_feat_name = (
            module_names[id(block.quant_feat)] + ".activation_post_process"
        )
        tie = _copy_activation_domain(
            _activation_fake_quant(block.quant_feat, quant_feat_name),
            incoming_fq, quant_feat_name, incoming_name,
        )
        tie["kind"] = "sa_feature_passthrough"
        ties.append(tie)
        if hasattr(block, "quant_skip"):
            quant_skip_name = (
                module_names[id(block.quant_skip)] + ".activation_post_process"
            )
            tie = _copy_activation_domain(
                _activation_fake_quant(block.quant_skip, quant_skip_name),
                incoming_fq, quant_skip_name, incoming_name,
            )
            tie["kind"] = "sa_skip_feature_passthrough"
            ties.append(tie)

        if getattr(block, "use_res", False):
            output_module = block.qadd.ff
            output_name = (
                module_names[id(output_module)] + ".activation_post_process"
            )
            output_fq = _activation_fake_quant(output_module, output_name)
        else:
            output_fq, output_name = _last_quantized_output(
                block, module_names, f"PointNeXt stage {stage_index}"
            )
        incoming_fq, incoming_name = output_fq, output_name
    return ties


def _pointnet2_feature_boundaries(model, module_names):
    """Yield each PointNet++ grouped-feature source and target domain."""
    encoder = model.encoder
    incoming_fq = _activation_fake_quant(
        encoder.quant_input, "encoder.quant_input.activation_post_process"
    )
    incoming_name = (
        module_names[id(encoder.quant_input)] + ".activation_post_process"
    )
    for stage_index, stage in enumerate(encoder.SA_modules):
        aggregations = list(stage.local_aggregations.children())
        if len(aggregations) != 1:
            raise RuntimeError(
                "hardware domain lock only admits one PointNet++ aggregation "
                f"per stage; stage {stage_index} has {len(aggregations)}"
            )
        aggregation = _pointnet2_aggregation_operator(
            aggregations[0], f"PointNet++ stage {stage_index}"
        )
        quant_feat_name = (
            module_names[id(aggregation.quant_feat)] + ".activation_post_process"
        )
        target_fq = _activation_fake_quant(
            aggregation.quant_feat, quant_feat_name
        )
        yield (
            stage_index + 1, incoming_fq, incoming_name,
            target_fq, quant_feat_name,
        )
        incoming_fq, incoming_name = _last_quantized_output(
            aggregation, module_names, f"PointNet++ stage {stage_index}"
        )


def _lock_pointnet2_encoder_domains(model, module_names):
    """Validate PointNet++ boundaries without destroying grouped dp range."""
    # PointNet++ groups signed dp with non-negative incoming features. SA2+
    # therefore needs an explicit feature requant before concatenation; copying
    # the incoming zero-point (normally zero after ReLU) onto quant_feat clips
    # negative dp and caused the observed S3DIS accuracy collapse.
    list(_pointnet2_feature_boundaries(model, module_names))
    return []


def _scalar_activation_domain(fake_quant, label):
    scale = fake_quant.scale.detach()
    zero_point = fake_quant.zero_point.detach()
    if scale.numel() != 1 or zero_point.numel() != 1:
        raise RuntimeError(f"{label}: activation domain must be per-tensor")
    scale_value = float(scale.cpu().item())
    zero_value = int(zero_point.cpu().item())
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise RuntimeError(f"{label}: invalid activation scale")
    if not 0 <= zero_value <= 255:
        raise RuntimeError(f"{label}: invalid activation zero point")
    return scale_value, zero_value


def _verify_feature_requant_codebook(
        source_scale, source_zero, target_scale, target_zero,
        multiplier, shift, label):
    """Prove all uint8 source codes match the framework fake-quant mapping."""
    source_codes = torch.arange(256, dtype=torch.float32)
    source_values = (
        source_codes - float(source_zero)
    ) * torch.tensor(source_scale, dtype=torch.float32)
    reference = torch.quantize_per_tensor(
        source_values, target_scale, target_zero, torch.quint8
    ).int_repr().to(dtype=torch.int64)
    rtl_codes = torch.tensor([
        min(255, max(0, rtl_requant(
            code - source_zero, multiplier, shift
        ) + target_zero))
        for code in range(256)
    ], dtype=torch.int64)
    mismatch = torch.nonzero(reference != rtl_codes).flatten()
    if mismatch.numel():
        code = int(mismatch[0].item())
        raise RuntimeError(
            f"{label}: Q31 feature requant is not bit-exact at source code "
            f"{code}: framework={int(reference[code].item())}, "
            f"rtl={int(rtl_codes[code].item())}"
        )


def collect_hardware_feature_requants(model, scope="all"):
    """Build PointNet++ grouped-feature requant records from frozen qparams."""
    if scope not in {"all", "encoder", "decoder"}:
        raise ValueError(f"invalid hardware domain lock scope: {scope}")
    if scope == "decoder":
        return []
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("hardware feature requant requires model.encoder")
    if not hasattr(encoder, "SA_modules"):
        return []

    module_names = {id(module): name for name, module in model.named_modules()}
    records = []
    for stage_index, source_fq, source_name, target_fq, target_name in (
            _pointnet2_feature_boundaries(model, module_names)):
        source_scale, source_zero = _scalar_activation_domain(
            source_fq, source_name
        )
        target_scale, target_zero = _scalar_activation_domain(
            target_fq, target_name
        )
        identity = source_scale == target_scale and source_zero == target_zero
        if identity:
            multiplier, shift = 1, 0
        else:
            multiplier, shift = q31_multiplier(source_scale / target_scale)
        _verify_feature_requant_codebook(
            source_scale, source_zero, target_scale, target_zero,
            multiplier, shift, f"PointNet++ SA{stage_index}",
        )
        records.append({
            "schema": "kdpoint.feature-requant-stage.v1",
            "stage": f"SA{stage_index}",
            "kind": "pointnet2_grouped_feature_requant",
            "source": source_name,
            "target": target_name,
            "source_scale": source_scale,
            "source_zero_point": source_zero,
            "target_scale": target_scale,
            "target_zero_point": target_zero,
            "m0": multiplier,
            "shift": shift,
            "identity": identity,
            "uint8_codebook_bit_exact": True,
            "formula": (
                "clamp(rtl_requant(q_in - source_zero_point, m0, shift) "
                "+ target_zero_point, 0, 255)"
            ),
        })
    if not records:
        raise RuntimeError("PointNet++ encoder has no grouped-feature boundaries")
    return records


def apply_hardware_domain_lock(model, scope="all"):
    """Tie eager-mode fake-quant domains to the domains implemented in RTL."""
    if scope not in {"all", "encoder", "decoder"}:
        raise ValueError(f"invalid hardware domain lock scope: {scope}")
    lock_encoder = scope in {"all", "encoder"}
    lock_decoder = scope in {"all", "decoder"}
    module_names = {id(module): name for name, module in model.named_modules()}
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("hardware domain lock requires model.encoder")
    ties = []
    if lock_encoder:
        if hasattr(encoder, "encoder") and hasattr(encoder, "quant_input"):
            ties = _lock_pointnext_encoder_domains(model, module_names)
        elif hasattr(encoder, "SA_modules") and hasattr(encoder, "quant_input"):
            ties = _lock_pointnet2_encoder_domains(model, module_names)
        else:
            raise RuntimeError(
                "hardware domain lock supports PointNeXt and PointNet++ encoders only"
            )

    interpolation_count = 0
    if lock_decoder:
        for module in model.modules():
            if not hasattr(module, "quant_interp") or not hasattr(module, "qcat"):
                continue
            qcat_ff = getattr(module.qcat, "ff", None)
            if qcat_ff is None:
                raise RuntimeError("decoder qcat is missing FloatFunctional state")
            interp_name = (
                module_names[id(module.quant_interp)] + ".activation_post_process"
            )
            qcat_name = module_names[id(qcat_ff)] + ".activation_post_process"
            tie = _copy_activation_domain(
                _activation_fake_quant(module.quant_interp, interp_name),
                _activation_fake_quant(qcat_ff, qcat_name),
                interp_name, qcat_name,
            )
            tie["kind"] = "decoder_interp_to_qcat"
            ties.append(tie)
            interpolation_count += 1

    decoder = getattr(model, "decoder", None)
    expected_interp = 0
    if decoder is not None:
        expected_interp = sum(
            1 for module in decoder.modules() if hasattr(module, "quant_interp")
        )
    if lock_decoder and interpolation_count != expected_interp:
        raise RuntimeError(
            "hardware domain lock missed decoder interpolation boundaries: "
            f"locked={interpolation_count}, expected={expected_interp}"
        )
    return ties


def set_hardware_geometry_mode(model, enabled):
    """Toggle the opt-in q9 geometry path without changing the model graph."""
    toggled = 0
    for module in model.modules():
        if hasattr(module, "coord_hardware_exact"):
            module.coord_hardware_exact = bool(enabled)
            toggled += 1
        if isinstance(module, QueryAndGroup):
            module.coord_step = None
            module.coord_dp_qparams = None
        elif isinstance(module, GroupAll):
            module.coord_step = None
            module.coord_origin = None
    if toggled == 0:
        raise RuntimeError("hardware geometry requested but no compatible module exists")
    return toggled


def bind_hardware_dp_qparams(model):
    """Bind each encoder grouper to its calibrated DpQuant output domain."""
    module_names = {id(module): name for name, module in model.named_modules()}
    bindings = []
    for module in model.modules():
        grouper = getattr(module, "grouper", None)
        quant_feat = getattr(module, "quant_feat", None)
        if not isinstance(grouper, QueryAndGroup) or quant_feat is None:
            continue
        if not grouper.coord_hardware_exact:
            continue
        quant_name = module_names[id(quant_feat)] + ".activation_post_process"
        fake_quant = _activation_fake_quant(quant_feat, quant_name)
        scale = fake_quant.scale.detach()
        zero_point = fake_quant.zero_point.detach()
        if scale.numel() != 1 or zero_point.numel() != 1:
            raise RuntimeError(f"{quant_name}: DpQuant domain must be per-tensor")
        scale_value = float(scale.cpu().item())
        zero_value = int(zero_point.cpu().item())
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise RuntimeError(f"{quant_name}: invalid DpQuant scale")
        if not 0 <= zero_value <= 255:
            raise RuntimeError(f"{quant_name}: invalid DpQuant zero point")
        grouper.coord_dp_qparams = (scale_value, zero_value)
        bindings.append({
            "grouper": module_names[id(grouper)],
            "target": quant_name,
            "scale": scale_value,
            "zero_point": zero_value,
            "radius": grouper.radius,
            "normalize_dp": grouper.normalize_dp,
        })
    if not bindings:
        raise RuntimeError("hardware geometry found no encoder DpQuant bindings")
    return bindings


def _record_operator_chain(module, first_domain, module_names, result, label):
    domain = first_domain
    found = 0
    for operator in module.modules():
        if not isinstance(operator, QUANTIZED_OPERATOR_TYPES):
            continue
        operator_name = module_names[id(operator)]
        if operator_name in result:
            raise RuntimeError(
                f"{label}: duplicate hardware input-domain record for "
                f"{operator_name}"
            )
        result[operator_name] = domain
        domain = operator_name + ".activation_post_process"
        found += 1
    if found == 0:
        raise RuntimeError(f"{label}: no Conv1d/Conv2d/Linear operators")
    return domain


def collect_hardware_operator_input_domains(model):
    """Record the exact fake-quant boundary feeding every hardware operator."""
    module_names = {id(module): name for name, module in model.named_modules()}
    result = OrderedDict()
    encoder = model.encoder
    root_domain = (
        module_names[id(encoder.quant_input)] + ".activation_post_process"
    )

    if hasattr(encoder, "encoder"):
        stages = list(encoder.encoder.children())
        stem_blocks = list(stages[0].children())
        incoming_domain = _record_operator_chain(
            stem_blocks[0].convs, root_domain, module_names, result,
            "PointNeXt stem",
        )
        for stage_index, stage in enumerate(stages[1:], start=1):
            blocks = list(stage.children())
            if len(blocks) != 1:
                raise RuntimeError(
                    "hardware input-domain map expects one PointNeXt block "
                    f"per stage; stage {stage_index} has {len(blocks)}"
                )
            block = blocks[0]
            if hasattr(block, "skipconv") and not isinstance(
                    block.skipconv, nn.Identity):
                _record_operator_chain(
                    block.skipconv, incoming_domain, module_names, result,
                    f"PointNeXt stage {stage_index} skipconv",
                )
            main_domain = (
                module_names[id(block.quant_feat)] + ".activation_post_process"
            )
            main_output = _record_operator_chain(
                block.convs, main_domain, module_names, result,
                f"PointNeXt stage {stage_index} main",
            )
            if getattr(block, "use_res", False):
                incoming_domain = (
                    module_names[id(block.qadd.ff)] + ".activation_post_process"
                )
            else:
                incoming_domain = main_output
    elif hasattr(encoder, "SA_modules"):
        incoming_domain = root_domain
        for stage_index, stage in enumerate(encoder.SA_modules):
            aggregations = list(stage.local_aggregations.children())
            if len(aggregations) != 1:
                raise RuntimeError(
                    "hardware input-domain map expects one PointNet++ "
                    f"aggregation per stage; stage {stage_index} has "
                    f"{len(aggregations)}"
                )
            aggregation = _pointnet2_aggregation_operator(
                aggregations[0], f"PointNet++ stage {stage_index}"
            )
            main_domain = (
                module_names[id(aggregation.quant_feat)] +
                ".activation_post_process"
            )
            incoming_domain = _record_operator_chain(
                aggregation.convs, main_domain, module_names, result,
                f"PointNet++ stage {stage_index}",
            )
    else:
        raise RuntimeError("unsupported encoder for hardware input-domain map")

    decoder_output_domains = {}
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "decoder"):
        for stage in decoder.decoder:
            blocks = list(stage.children())
            if len(blocks) != 1:
                raise RuntimeError("PointNeXt decoder stage must contain one block")
            block = blocks[0]
            qcat_domain = (
                module_names[id(block.qcat.ff)] + ".activation_post_process"
            )
            decoder_output_domains[id(block)] = _record_operator_chain(
                block.convs, qcat_domain, module_names, result,
                module_names[id(block)],
            )
        final_block = list(decoder.decoder[0].children())[0]
        final_feature_domain = decoder_output_domains[id(final_block)]
    elif decoder is not None and hasattr(decoder, "FP_modules"):
        for block in decoder.FP_modules:
            qcat_domain = (
                module_names[id(block.qcat.ff)] + ".activation_post_process"
            )
            decoder_output_domains[id(block)] = _record_operator_chain(
                block.convs, qcat_domain, module_names, result,
                module_names[id(block)],
            )
        final_feature_domain = decoder_output_domains[id(decoder.FP_modules[0])]
    else:
        final_feature_domain = incoming_domain

    if hasattr(model, "prediction") and hasattr(model.prediction, "head"):
        _record_operator_chain(
            model.prediction.head, incoming_domain, module_names, result,
            "classification head",
        )
    if hasattr(model, "head") and hasattr(model.head, "head"):
        _record_operator_chain(
            model.head.head, final_feature_domain, module_names, result,
            "segmentation head",
        )

    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, QUANTIZED_OPERATOR_TYPES)
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(
            "hardware input-domain map does not cover every quantized operator: "
            f"missing={missing}, extra={extra}"
        )
    return result


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
        "--hardware-domain-lock", action="store_true",
        help=(
            "tie eager fake-quant boundaries to the feature passthrough, "
            "fused-lift and decoder qcat domains implemented by KDPoint RTL"
        ),
    )
    parser.add_argument(
        "--hardware-domain-lock-scope",
        choices=("all", "encoder", "decoder"), default="all",
        help="ablation scope; final export accepts only the default all",
    )
    parser.add_argument(
        "--hardware-geometry", action="store_true",
        help=(
            "evaluate KDPoint's isotropic q9 encoder sidecar, q8 main-path "
            "FPS/KD/KNN, dynamic DpQuant, and exact U1.7 interpolation weights"
        ),
    )
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
    if args.hardware_geometry:
        cfg.model.encoder_args.coord_hardware_exact = True

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
    if args.hardware_geometry:
        toggled = set_hardware_geometry_mode(model, False)
        logger.info(
            "[KDPOINT GEOMETRY] disabled on %d modules for FP32 baselines",
            toggled,
        )

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

    if args.hardware_geometry:
        toggled = set_hardware_geometry_mode(model, True)
        logger.info(
            "[KDPOINT GEOMETRY] enabled on %d modules for calibration/INT8",
            toggled,
        )

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
    hardware_domain_ties = []
    hardware_operator_input_domains = {}
    hardware_dp_bindings = []
    hardware_feature_requants = []
    if args.hardware_domain_lock:
        hardware_domain_ties = apply_hardware_domain_lock(
            model, scope=args.hardware_domain_lock_scope
        )
        hardware_feature_requants = collect_hardware_feature_requants(
            model, scope=args.hardware_domain_lock_scope
        )
        hardware_operator_input_domains = collect_hardware_operator_input_domains(
            model
        )
        logger.info(
            "[HARDWARE DOMAIN LOCK][PASS] tied %d activation boundaries; "
            "recorded %d feature requants; mapped %d operator inputs",
            len(hardware_domain_ties),
            len(hardware_feature_requants),
            len(hardware_operator_input_domains),
        )
    if args.hardware_geometry:
        hardware_dp_bindings = bind_hardware_dp_qparams(model)
        logger.info(
            "[KDPOINT GEOMETRY][PASS] bound %d dynamic DpQuant stages",
            len(hardware_dp_bindings),
        )
    operators = audit_quantized_operators(
        model, stage="after calibration", require_calibrated_buffers=True
    )
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
        "hardware_domain_lock": args.hardware_domain_lock,
        "hardware_domain_lock_scope": args.hardware_domain_lock_scope,
        "hardware_geometry": args.hardware_geometry,
        "hardware_dp_bindings": hardware_dp_bindings,
        "hardware_domain_ties": hardware_domain_ties,
        "hardware_feature_requants": hardware_feature_requants,
        "hardware_operator_input_domains": hardware_operator_input_domains,
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
            "hardware_domain_lock": args.hardware_domain_lock,
            "hardware_domain_lock_scope": args.hardware_domain_lock_scope,
            "hardware_geometry": args.hardware_geometry,
            "hardware_dp_bindings": hardware_dp_bindings,
            "hardware_domain_ties": hardware_domain_ties,
            "hardware_feature_requants": hardware_feature_requants,
            "hardware_operator_input_domains": hardware_operator_input_domains,
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
