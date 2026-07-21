"""Hardware-exact integer export helpers for KDPoint D5 artifacts."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.quantization as quant

try:
    from .kdpoint_requant import q31_multiplier, rtl_requant
except ImportError:
    from kdpoint_requant import q31_multiplier, rtl_requant


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1


def scalar_qparams(fake_quant: nn.Module, label: str) -> Tuple[float, int]:
    if not isinstance(fake_quant, quant.FakeQuantize):
        raise ValueError(f"{label}: expected FakeQuantize")
    if fake_quant.dtype != torch.quint8:
        raise ValueError(f"{label}: activation dtype must be quint8")
    if fake_quant.quant_min != 0 or fake_quant.quant_max != 255:
        raise ValueError(f"{label}: activation range must be [0,255]")
    if not bool(fake_quant.fake_quant_enabled[0].item()):
        raise ValueError(f"{label}: activation fake-quant is disabled")
    if bool(fake_quant.observer_enabled[0].item()):
        raise ValueError(f"{label}: activation observer must be frozen")
    # The buffers are what FakeQuantize.forward actually uses. Recomputing from
    # the serialized observer can differ by one FP32 ULP after a round trip.
    scale = fake_quant.scale.detach()
    zero_point = fake_quant.zero_point.detach()
    if scale.numel() != 1 or zero_point.numel() != 1:
        raise ValueError(f"{label}: activation qparams must be per-tensor")
    scale_value = float(scale.detach().cpu().item())
    zero_value = int(zero_point.detach().cpu().item())
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"{label}: invalid activation scale {scale_value}")
    if not 0 <= zero_value <= 255:
        raise ValueError(f"{label}: zero point outside uint8: {zero_value}")
    return scale_value, zero_value


def _weight_qparams(module: nn.Module, label: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_quant = getattr(module, "weight_fake_quant", None)
    if not isinstance(fake_quant, quant.FakeQuantize):
        raise ValueError(f"{label}: missing weight FakeQuantize")
    if fake_quant.dtype != torch.qint8:
        raise ValueError(f"{label}: weight dtype must be qint8")
    if fake_quant.quant_min != -128 or fake_quant.quant_max != 127:
        raise ValueError(f"{label}: weight range must be [-128,127]")
    if getattr(fake_quant, "ch_axis", None) != 0:
        raise ValueError(f"{label}: weight channel axis must be 0")
    if not bool(fake_quant.fake_quant_enabled[0].item()):
        raise ValueError(f"{label}: weight fake-quant is disabled")
    if bool(fake_quant.observer_enabled[0].item()):
        raise ValueError(f"{label}: weight observer must be frozen")
    scales = fake_quant.scale.detach().cpu().to(torch.float64)
    zero_points = fake_quant.zero_point.detach().cpu().to(torch.int64)
    if scales.numel() != module.weight.shape[0]:
        raise ValueError(f"{label}: one weight scale is required per output")
    if not torch.isfinite(scales).all() or not torch.all(scales > 0):
        raise ValueError(f"{label}: invalid per-output weight scales")
    if not torch.equal(zero_points, torch.zeros_like(zero_points)):
        raise ValueError(f"{label}: symmetric weight zero points must all be zero")
    return scales, zero_points


def quantized_weight_matrix(
        module: nn.Module, label: str) -> Tuple[List[List[int]], List[float]]:
    """Return signed int8 weights in logical Cin x Cout order."""
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        raise ValueError(f"{label}: unsupported operator {type(module).__name__}")
    if getattr(module, "groups", 1) != 1:
        raise ValueError(f"{label}: grouped convolution is not a pointwise PU op")
    weight = module.weight.detach().cpu().to(torch.float32)
    if weight.ndim > 2 and any(size != 1 for size in weight.shape[2:]):
        raise ValueError(f"{label}: only 1x1 pointwise kernels are supported")
    cout, cin = weight.shape[:2]
    scales, zero_points = _weight_qparams(module, label)
    view_shape = (cout,) + (1,) * (weight.ndim - 1)
    scales_f32 = scales.to(torch.float32).reshape(view_shape)
    reference = torch.fake_quantize_per_channel_affine(
        weight, scales.to(torch.float32),
        zero_points.to(torch.int32),
        0, -128, 127,
    )
    # Recover the exact integer codes emitted by PyTorch's fake-quant kernel.
    # Re-quantizing the FP32 source independently differs at negative ties in
    # torch 1.10, including real stem weights at the -127/-128 boundary.
    quantized = torch.round(reference / scales_f32).clamp(-128, 127).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scales_f32
    if not torch.allclose(dequantized, reference, rtol=0.0, atol=1e-7):
        raise ValueError(f"{label}: integer weights disagree with fake-quant output")

    output_major = quantized.reshape(cout, cin)
    input_major = output_major.transpose(0, 1).contiguous()
    matrix = [[int(value) for value in row] for row in input_major.tolist()]
    return matrix, [float(value) for value in scales.tolist()]


def _bias_int32(
        module: nn.Module, input_scale: float,
        weight_scales: Sequence[float], label: str) -> List[int]:
    if module.bias is None:
        return [0] * len(weight_scales)
    bias = module.bias.detach().cpu().to(torch.float64)
    denominators = torch.tensor(weight_scales, dtype=torch.float64) * input_scale
    values = torch.round(bias / denominators)
    if not torch.isfinite(values).all():
        raise ValueError(f"{label}: non-finite folded bias")
    result = [int(value) for value in values.tolist()]
    for index, value in enumerate(result):
        if not INT32_MIN <= value <= INT32_MAX:
            raise ValueError(f"{label}: bias_int32[{index}] overflows int32")
    return result


def _weight_bytes(matrix: Sequence[Sequence[int]]) -> bytes:
    return bytes(value & 0xff for row in matrix for value in row)


def validate_operator_payload(
        payload: Dict[str, object], matrix: Sequence[Sequence[int]]) -> None:
    name = str(payload["name"])
    cin = int(payload["cin"])
    cout = int(payload["cout"])
    if len(matrix) != cin or any(len(row) != cout for row in matrix):
        raise ValueError(f"{name}: weight matrix shape mismatch")
    fields = ("weight_scale_per_output", "bias_int32", "m0", "shift", "col_sum_w")
    for field in fields:
        if len(payload[field]) != cout:
            raise ValueError(f"{name}: {field} length mismatch")

    col_sums = [sum(matrix[row][col] for row in range(cin)) for col in range(cout)]
    if list(payload["col_sum_w"]) != col_sums:
        raise ValueError(f"{name}: col_sum_w mismatch")

    input_scale = float(payload["input_scale"])
    output_scale = float(payload["output_scale"])
    input_zero = int(payload["input_zero_point"])
    for col in range(cout):
        scale = float(payload["weight_scale_per_output"][col])
        ratio = input_scale * scale / output_scale
        multiplier = int(payload["m0"][col])
        shift = int(payload["shift"][col])
        actual = math.ldexp(float(multiplier), -shift)
        identity = multiplier == 1 and shift == 0 and math.isclose(
            ratio, 1.0, rel_tol=1e-12, abs_tol=1e-15)
        if not identity and not (1 << 30) <= multiplier < (1 << 31):
            raise ValueError(f"{name}: m0[{col}] is not normalized Q31")
        tolerance = max(math.ldexp(0.5, -shift), abs(ratio) * 1e-12, 1e-15)
        if abs(actual - ratio) > tolerance:
            raise ValueError(f"{name}: fixed multiplier mismatch at output {col}")

        weights = [int(matrix[row][col]) for row in range(cin)]
        centered_min = sum(min(
            -input_zero * weight, (255 - input_zero) * weight
        ) for weight in weights)
        centered_max = sum(max(
            -input_zero * weight, (255 - input_zero) * weight
        ) for weight in weights)
        biased_min = centered_min + int(payload["bias_int32"][col])
        biased_max = centered_max + int(payload["bias_int32"][col])
        if biased_min < INT32_MIN or biased_max > INT32_MAX:
            raise ValueError(f"{name}: accumulator bound exceeds int32 at output {col}")
        if max(abs(biased_min), abs(biased_max)) * multiplier > INT64_MAX:
            raise ValueError(f"{name}: requant product exceeds int64 at output {col}")


def direct_operator_payload(
        name: str, module: nn.Module, input_fake_quant: nn.Module
        ) -> Tuple[Dict[str, object], bytes]:
    input_scale, input_zero = scalar_qparams(input_fake_quant, f"{name}/input")
    output_fake_quant = getattr(module, "activation_post_process", None)
    output_scale, output_zero = scalar_qparams(
        output_fake_quant, f"{name}/output"
    )
    matrix, weight_scales = quantized_weight_matrix(module, name)
    cin = len(matrix)
    cout = len(matrix[0])
    bias = _bias_int32(module, input_scale, weight_scales, name)
    multipliers, shifts = zip(*(
        q31_multiplier(input_scale * scale / output_scale)
        for scale in weight_scales
    ))
    payload = {
        "schema": "kdpoint.quant-operator.v1",
        "name": name,
        "cin": cin,
        "cout": cout,
        "input_scale": input_scale,
        "input_zero_point": input_zero,
        "weight_scale_per_output": weight_scales,
        "bias_int32": bias,
        "output_scale": output_scale,
        "output_zero_point": output_zero,
        "m0": list(multipliers),
        "shift": list(shifts),
        "col_sum_w": [sum(matrix[row][col] for row in range(cin))
                      for col in range(cout)],
    }
    validate_operator_payload(payload, matrix)
    return payload, _weight_bytes(matrix)


def fused_lift_payload(
        name: str, stem_module: nn.Module, input_fake_quant: nn.Module,
        coord_dim: int = 3) -> Tuple[Dict[str, object], bytes]:
    """Build [dp;raw] -> [dp identity;stem] for PointNeXt stem absorption."""
    source, source_bytes = direct_operator_payload(
        name + ".source_stem", stem_module, input_fake_quant
    )
    if not math.isclose(
            source["input_scale"], source["output_scale"],
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{name}: fused-lift stem output scale is not input-locked")
    if source["input_zero_point"] != source["output_zero_point"]:
        raise ValueError(f"{name}: fused-lift stem output zero point is not input-locked")

    raw_cin = int(source["cin"])
    stem_cout = int(source["cout"])
    source_signed = [value if value < 128 else value - 256 for value in source_bytes]
    source_matrix = [
        source_signed[row * stem_cout:(row + 1) * stem_cout]
        for row in range(raw_cin)
    ]
    cin = coord_dim + raw_cin
    cout = coord_dim + stem_cout
    matrix = [[0] * cout for _ in range(cin)]
    for axis in range(coord_dim):
        matrix[axis][axis] = 1
    for row in range(raw_cin):
        for col in range(stem_cout):
            matrix[coord_dim + row][coord_dim + col] = source_matrix[row][col]

    payload = {
        "schema": "kdpoint.quant-operator.v1",
        "name": name,
        "cin": cin,
        "cout": cout,
        "input_scale": source["input_scale"],
        "input_zero_point": source["input_zero_point"],
        "weight_scale_per_output": [1.0] * coord_dim +
                                   list(source["weight_scale_per_output"]),
        "bias_int32": [0] * coord_dim + list(source["bias_int32"]),
        "output_scale": source["output_scale"],
        "output_zero_point": source["output_zero_point"],
        "m0": [1] * coord_dim + list(source["m0"]),
        "shift": [0] * coord_dim + list(source["shift"]),
        "col_sum_w": [sum(matrix[row][col] for row in range(cin))
                      for col in range(cout)],
    }
    validate_operator_payload(payload, matrix)
    return payload, _weight_bytes(matrix)


def residual_record(
        stage: str, main_qparams: Tuple[float, int],
        skip_qparams: Tuple[float, int], output_qparams: Tuple[float, int]
        ) -> Dict[str, object]:
    main_scale, main_zero = main_qparams
    skip_scale, skip_zero = skip_qparams
    output_scale, output_zero = output_qparams
    m0_a, shift_a = q31_multiplier(main_scale / output_scale)
    m0_b, shift_b = q31_multiplier(skip_scale / output_scale)
    bias = (output_zero - rtl_requant(main_zero, m0_a, shift_a)
            - rtl_requant(skip_zero, m0_b, shift_b))
    if not INT32_MIN <= bias <= INT32_MAX:
        raise ValueError(f"{stage}: residual bias overflows int32")
    return {
        "stage": stage,
        "main": {"scale": main_scale, "zero_point": main_zero,
                 "m0": m0_a, "shift": shift_a},
        "skip": {"scale": skip_scale, "zero_point": skip_zero,
                 "m0": m0_b, "shift": shift_b},
        "output": {"scale": output_scale, "zero_point": output_zero},
        "bias": bias,
        "clamp_lo": output_zero,
        "rounding": "add_positive_half_then_arithmetic_shift",
    }


def interp_qcat_record(
        stage: str, coarse_qparams: Tuple[float, int],
        skip_qparams: Tuple[float, int], qcat_qparams: Tuple[float, int],
        weight_fraction_bits: int = 7) -> Dict[str, object]:
    coarse_scale, coarse_zero = coarse_qparams
    skip_scale, skip_zero = skip_qparams
    qcat_scale, qcat_zero = qcat_qparams
    interp_m0, interp_shift = q31_multiplier(
        coarse_scale / ((1 << weight_fraction_bits) * qcat_scale)
    )
    skip_m0, skip_shift = q31_multiplier(skip_scale / qcat_scale)
    return {
        "stage": stage,
        "weight_format": f"U1.{weight_fraction_bits}",
        "interpolation": {
            "input_scale": coarse_scale,
            "z_f": coarse_zero,
            "bias": 0,
            "m0": interp_m0,
            "shift": interp_shift,
            "output_scale": qcat_scale,
            "z_out": qcat_zero,
        },
        "qcat_skip": {
            "input_scale": skip_scale,
            "z_in": skip_zero,
            "m0": skip_m0,
            "shift": skip_shift,
            "output_scale": qcat_scale,
            "z_out": qcat_zero,
        },
        "rounding": "add_positive_half_then_arithmetic_shift",
    }


def flatten_signed_bytes(data: bytes) -> List[int]:
    return [value if value < 128 else value - 256 for value in data]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _ptq_general_module():
    try:
        from . import ptq_general
    except ImportError:
        import ptq_general
    return ptq_general


def _last_operator(module: nn.Module, label: str) -> nn.Module:
    operator_types = _ptq_general_module().QUANTIZED_OPERATOR_TYPES

    operators = [child for child in module.modules()
                 if isinstance(child, operator_types)]
    if not operators:
        raise ValueError(f"{label}: no quantized operator")
    return operators[-1]


def _operator_layouts(network_layout: Mapping[str, object]) -> Dict[str, dict]:
    result = {}
    for pu in network_layout["physical_pus"]:
        for operator in pu["operators"]:
            name = operator["name"]
            if name in result:
                raise ValueError(f"duplicate preload layout operator: {name}")
            result[name] = operator
    return result


def _load_json(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def verify_hardware_domain_ties(model: nn.Module, ties: Sequence[dict]) -> None:
    modules = _module_map(model)
    for index, tie in enumerate(ties):
        target_name = tie.get("target")
        source_name = tie.get("source")
        if target_name not in modules or source_name not in modules:
            raise ValueError(f"hardware domain tie {index} references a missing module")
        target = scalar_qparams(modules[target_name], target_name)
        source = scalar_qparams(modules[source_name], source_name)
        if target != source:
            raise ValueError(
                f"hardware domain tie does not reproduce: {target_name} != {source_name}"
            )
        recorded_scale = tie.get("scale")
        recorded_zero = tie.get("zero_point")
        if recorded_scale != [target[0]] or recorded_zero != [target[1]]:
            raise ValueError(f"hardware domain tie metadata drift: {target_name}")


def verify_hardware_feature_requants(
        model: nn.Module, records: Optional[Sequence[dict]]) -> List[dict]:
    """Require saved PointNet++ feature-requant config to reproduce exactly."""
    derived = _ptq_general_module().collect_hardware_feature_requants(
        model, scope="all"
    )
    # PointNeXt artifacts created immediately before this metadata field was
    # introduced legitimately omit it: their derived table is empty.  Keep
    # PointNet++ strict, because a missing non-empty table would silently
    # restore the incorrect raw-code concatenation.
    if records is None:
        if derived:
            raise ValueError(
                "calibrated state lacks hardware feature-requant records"
            )
        return []

    normalized = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("saved hardware feature-requant record is not a map")
        normalized_record = dict(record)
        proof = normalized_record.get("uint8_codebook_bit_exact")
        if proof not in (None, True):
            raise ValueError("saved hardware feature-requant proof is not true")
        # Artifacts emitted immediately before the exhaustive codebook gate
        # contain all qparams but not this redundant attestation. Re-deriving
        # the record above runs the gate, so upgrading only this field is safe.
        normalized_record["uint8_codebook_bit_exact"] = True
        normalized.append(normalized_record)
    if normalized != derived:
        raise ValueError("saved hardware feature-requant records do not reproduce")
    return list(records)


def prepare_calibrated_model(
        project_root: Path, state_path: Path, require_saved_map: bool = True
        ) -> Tuple[nn.Module, dict, dict]:
    """Rebuild the exact QAT graph and strictly load one calibrated state."""
    project_root = project_root.resolve()
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig
    from openpoints.models.layers.quant_utils import (
        disable_quantization_for_geometry,
        fuse_convbn_modules,
        swap_custom_convs_to_standard,
    )
    ptq_general = _ptq_general_module()

    state_path = state_path.resolve()
    payload = torch.load(str(state_path), map_location="cpu")
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError("calibrated state is missing metadata")
    metadata = payload["metadata"]
    if not metadata.get("hardware_domain_lock"):
        raise ValueError("calibrated state was not produced with hardware-domain-lock")
    if metadata.get("hardware_domain_lock_scope", "all") != "all":
        raise ValueError("calibrated state used a partial hardware-domain-lock scope")
    if not metadata.get("hardware_geometry"):
        raise ValueError("calibrated state was not produced with hardware-geometry")
    cfg_path = (project_root / metadata["cfg"]).resolve()
    ckpt_path = (project_root / metadata["ckpt"]).resolve()
    if not cfg_path.is_file() or not ckpt_path.is_file():
        raise ValueError("calibrated metadata cfg/checkpoint path is unavailable")

    cfg = EasyConfig()
    cfg.load(str(cfg_path), recursive=True)
    cfg.update(metadata.get("opts", []))
    cfg.model.encoder_args.coord_hardware_exact = True
    model = build_model_from_cfg(cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    source_state = checkpoint["model"] if "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(source_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("source checkpoint strict load returned incompatible keys")
    model.eval()
    swap_custom_convs_to_standard(model)
    fuse_convbn_modules(model)
    disable_quantization_for_geometry(model)
    backend = metadata["backend"]
    torch.backends.quantized.engine = backend
    model.qconfig = ptq_general.get_hardware_qat_qconfig(backend)
    model.train()
    quant.prepare_qat(
        model, mapping=ptq_general.get_qat_module_mapping(), inplace=True
    )
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("calibrated state strict load returned incompatible keys")
    model.eval()
    ptq_general.audit_quantized_operators(
        model, stage="calibrated state reload",
        require_calibrated_buffers=True,
    )
    ptq_general.set_hardware_geometry_mode(model, True)
    derived_dp_bindings = ptq_general.bind_hardware_dp_qparams(model)
    saved_dp_bindings = metadata.get("hardware_dp_bindings", [])
    if not saved_dp_bindings:
        raise ValueError("calibrated state lacks hardware DpQuant bindings")
    if saved_dp_bindings != derived_dp_bindings:
        raise ValueError("saved hardware DpQuant bindings do not reproduce")

    ties = metadata.get("hardware_domain_ties", [])
    feature_requants = verify_hardware_feature_requants(
        model, metadata.get("hardware_feature_requants")
    )
    if not ties and not feature_requants:
        raise ValueError("calibrated state has no hardware activation-domain records")
    verify_hardware_domain_ties(model, ties)
    derived_map = dict(
        ptq_general.collect_hardware_operator_input_domains(model)
    )
    saved_map = metadata.get("hardware_operator_input_domains", {})
    if require_saved_map and not saved_map:
        raise ValueError("calibrated state lacks saved hardware operator input domains")
    if saved_map and saved_map != derived_map:
        raise ValueError("saved hardware operator input-domain map does not reproduce")
    return model, metadata, derived_map


def validate_source_provenance(
        project_root: Path, state_path: Path, metrics_path: Path,
        metadata: Mapping[str, object], source_record: Mapping[str, object]
        ) -> None:
    expected = source_record["provenance"]
    hashes = metadata.get("provenance", {}).get("sha256", {})
    checks = {
        "config_file": expected["cfg_sha256"],
        "checkpoint": expected["checkpoint_sha256"],
    }
    for key, value in checks.items():
        if hashes.get(key) != value:
            raise ValueError(f"calibrated provenance mismatch: {key}")
    if metadata.get("backend") != expected["backend"]:
        raise ValueError("calibrated backend does not match source contract")
    if metadata.get("calib_batches") != expected["calib_batches"]:
        raise ValueError("calibration batch count does not match source contract")
    if metadata.get("opts") != expected["opts"]:
        raise ValueError("calibration options do not match source contract")

    metrics = _load_json(metrics_path, "metrics")
    for key in ("cfg", "ckpt", "backend", "calib_batches", "opts"):
        if metrics.get(key) != metadata.get(key):
            raise ValueError(f"metrics/calibrated metadata mismatch: {key}")
    if not metrics.get("hardware_domain_lock"):
        raise ValueError("metrics do not attest hardware-domain-lock")
    if metrics.get("hardware_domain_lock_scope", "all") != metadata.get(
            "hardware_domain_lock_scope", "all"):
        raise ValueError("metrics/calibrated hardware-domain-lock scope mismatch")
    if not metrics.get("hardware_geometry"):
        raise ValueError("metrics do not attest hardware-geometry")
    if metrics.get("hardware_dp_bindings") != metadata.get(
            "hardware_dp_bindings"):
        raise ValueError("metrics/calibrated hardware DpQuant bindings mismatch")
    if metrics.get("hardware_feature_requants") != metadata.get(
            "hardware_feature_requants"):
        raise ValueError("metrics/calibrated feature-requant records mismatch")
    if metrics.get("hardware_operator_input_domains") != metadata.get(
            "hardware_operator_input_domains"):
        raise ValueError("metrics/calibrated operator input-domain map mismatch")
    state_record = metrics.get("artifacts", {}).get("calibrated_state", {})
    if state_record.get("sha256") != sha256_file(state_path):
        raise ValueError("metrics calibrated-state hash mismatch")


def build_operator_exports(
        model: nn.Module, input_domains: Mapping[str, str],
        source_record: Mapping[str, object], network_layout: Mapping[str, object]
        ) -> Dict[str, Tuple[dict, bytes]]:
    modules = _module_map(model)
    layouts = _operator_layouts(network_layout)
    bindings = source_record["bindings"]
    if len(bindings) != source_record["logical_operator_count"]:
        raise ValueError("source binding count drift")
    result = {}
    consumed_sources = set()
    for binding in bindings:
        rtl_name = binding["rtl_operator"]
        source_name = binding["source_module"]
        if rtl_name not in layouts:
            raise ValueError(f"{rtl_name}: missing preload layout")
        if source_name not in modules:
            raise ValueError(f"{rtl_name}: source module {source_name} is absent")
        if source_name not in input_domains:
            raise ValueError(f"{rtl_name}: source input quantization domain is absent")
        domain_name = input_domains[source_name]
        if domain_name not in modules:
            raise ValueError(f"{rtl_name}: input domain {domain_name} is absent")
        source_module = modules[source_name]
        source_shape = list(source_module.weight.shape)
        if source_shape != binding["source_weight_shape"]:
            raise ValueError(
                f"{rtl_name}: source weight shape {source_shape} != "
                f"{binding['source_weight_shape']}"
            )
        transform = binding["transform"]
        if transform == "direct_pointwise":
            payload, weight_bytes = direct_operator_payload(
                rtl_name, source_module, modules[domain_name]
            )
        elif transform == "pointnext_stem_absorption_dp_identity":
            payload, weight_bytes = fused_lift_payload(
                rtl_name, source_module, modules[domain_name]
            )
        else:
            raise ValueError(f"{rtl_name}: unsupported transform {transform}")
        layout = layouts[rtl_name]
        if (payload["cin"], payload["cout"]) != (
                layout["cin"], layout["cout"]):
            raise ValueError(
                f"{rtl_name}: export shape {payload['cin']}x{payload['cout']} "
                f"!= layout {layout['cin']}x{layout['cout']}"
            )
        if rtl_name in result or source_name in consumed_sources:
            raise ValueError(f"{rtl_name}: duplicate logical/source operator")
        result[rtl_name] = (payload, weight_bytes)
        consumed_sources.add(source_name)
    if set(result) != set(layouts):
        raise ValueError("logical export does not cover preload layout exactly")
    if set(consumed_sources) != set(input_domains):
        missing = sorted(set(input_domains) - consumed_sources)
        raise ValueError(f"source contract does not consume operators: {missing}")
    return result


def build_residual_export(model: nn.Module) -> dict:
    encoder = getattr(model, "encoder", None)
    records = []
    if encoder is not None and hasattr(encoder, "encoder"):
        stages = list(encoder.encoder.children())
        for stage_index, stage in enumerate(stages[1:], start=1):
            blocks = list(stage.children())
            if len(blocks) != 1:
                raise ValueError(f"PointNeXt SA{stage_index}: expected one block")
            block = blocks[0]
            if not getattr(block, "use_res", False):
                continue
            main = _last_operator(block.convs, f"SA{stage_index}/main")
            skip_module = getattr(block, "skipconv", None)
            if isinstance(skip_module, nn.Identity):
                skip_fq = block.quant_skip.activation_post_process
            else:
                skip = _last_operator(skip_module, f"SA{stage_index}/skip")
                skip_fq = skip.activation_post_process
            records.append(residual_record(
                f"SA{stage_index}",
                scalar_qparams(main.activation_post_process,
                               f"SA{stage_index}/main"),
                scalar_qparams(skip_fq, f"SA{stage_index}/skip"),
                scalar_qparams(block.qadd.ff.activation_post_process,
                               f"SA{stage_index}/qadd"),
            ))
    return {"schema": "kdpoint.residual-config.v1", "stages": records}


def _encoder_feature_domains(model: nn.Module) -> List[Tuple[float, int]]:
    encoder = model.encoder
    if hasattr(encoder, "encoder"):
        stages = list(encoder.encoder.children())
        domains = [scalar_qparams(
            _last_operator(stages[0], "PointNeXt stem").activation_post_process,
            "PointNeXt stem",
        )]
        for index, stage in enumerate(stages[1:], start=1):
            block = list(stage.children())[0]
            output = (block.qadd.ff if getattr(block, "use_res", False)
                      else _last_operator(block.convs, f"PointNeXt SA{index}"))
            domains.append(scalar_qparams(
                output.activation_post_process, f"PointNeXt SA{index} output"
            ))
        return domains
    if hasattr(encoder, "SA_modules"):
        domains = [scalar_qparams(
            encoder.quant_input.activation_post_process, "PointNet++ input"
        )]
        for index, stage in enumerate(encoder.SA_modules, start=1):
            output = _last_operator(stage, f"PointNet++ SA{index}")
            domains.append(scalar_qparams(
                output.activation_post_process, f"PointNet++ SA{index} output"
            ))
        return domains
    raise ValueError("unsupported encoder for decoder domain tracing")


def _decoder_blocks(model: nn.Module) -> List[nn.Module]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return []
    if hasattr(decoder, "decoder"):
        blocks = []
        for stage in decoder.decoder:
            children = list(stage.children())
            if len(children) != 1:
                raise ValueError("PointNeXt decoder stage must contain one block")
            blocks.append(children[0])
        return blocks
    if hasattr(decoder, "FP_modules"):
        return list(decoder.FP_modules)
    raise ValueError("unsupported decoder for qcat domain tracing")


def build_interp_qcat_export(
        model: nn.Module, source_record: Mapping[str, object]) -> dict:
    blocks = _decoder_blocks(model)
    if not blocks:
        return {"schema": "kdpoint.interp-qcat-config.v1", "stages": []}
    feature_domains = _encoder_feature_domains(model)
    if len(feature_domains) != len(blocks) + 1:
        raise ValueError(
            "decoder depth does not match encoder skip/coarse domain count"
        )
    names = {id(module): name for name, module in model.named_modules()}
    stage_by_block = {}
    for binding in source_record["bindings"]:
        rtl_name = binding["rtl_operator"]
        if not rtl_name.startswith("FP") or ".mlp1" not in rtl_name:
            continue
        source_name = binding["source_module"]
        block_name = source_name.split(".convs.", 1)[0]
        stage_by_block[block_name] = rtl_name.split(".", 1)[0]

    records = []
    coarse_domain = feature_domains[-1]
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        block_name = names[id(block)]
        if block_name not in stage_by_block:
            raise ValueError(f"{block_name}: no RTL FP stage binding")
        qcat = scalar_qparams(
            block.qcat.ff.activation_post_process, f"{block_name}/qcat"
        )
        interp = scalar_qparams(
            block.quant_interp.activation_post_process, f"{block_name}/interp"
        )
        if interp != qcat:
            raise ValueError(f"{block_name}: interpolation is not qcat-domain locked")
        records.append(interp_qcat_record(
            stage_by_block[block_name], coarse_domain,
            feature_domains[index], qcat,
        ))
        coarse_domain = scalar_qparams(
            _last_operator(block.convs, block_name).activation_post_process,
            f"{block_name}/output",
        )
    records.sort(key=lambda record: int(record["stage"][2:]))
    return {"schema": "kdpoint.interp-qcat-config.v1", "stages": records}


def build_geometry_contract(model: nn.Module) -> dict:
    """Build the checked runtime q9/q8 coordinate and DpQuant contract."""
    from openpoints.models.layers.group import QueryAndGroup

    records = []
    for owner_name, owner in model.encoder.named_modules():
        grouper = getattr(owner, "grouper", None)
        quant_feat = getattr(owner, "quant_feat", None)
        if not isinstance(grouper, QueryAndGroup) or quant_feat is None:
            continue
        stage = f"SA{len(records) + 1}"
        if not grouper.coord_hardware_exact:
            raise ValueError(f"{stage}: encoder BQ/dp hardware geometry is disabled")
        target_scale, target_zero = scalar_qparams(
            quant_feat.activation_post_process, f"{stage}/quant_feat"
        )
        bound_qparams = getattr(grouper, "coord_dp_qparams", None)
        if bound_qparams != (target_scale, target_zero):
            raise ValueError(f"{stage}: runtime DpQuant qparams are not bound")
        radius = float(grouper.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{stage}: invalid BQ radius {radius!r}")
        normalize_dp = bool(grouper.normalize_dp)
        records.append({
            "stage": stage,
            "owner": owner_name,
            "radius": radius,
            "normalize_dp": normalize_dp,
            "strict_bq_threshold": "ceil((radius / scene_step)^2) - 1",
            "strict_bq_accept": "sum(delta_q9^2) <= threshold",
            "dp_source": "delta_q9",
            "dp_ratio": (
                "scene_step / (radius * target_scale)"
                if normalize_dp else "scene_step / target_scale"
            ),
            "target_scale": target_scale,
            "target_zero_point": target_zero,
            "runtime_multiplier": {
                "encoding": "positive Q31 multiplier plus arithmetic right shift",
                "rounding": "add positive half before arithmetic shift",
                "canonicalization": "remove common powers of two from M0 and shift",
            },
        })
    if not records:
        raise ValueError("encoder has no hardware-exact radius-BQ/DpQuant stages")

    decoder = getattr(model, "decoder", None)
    decoder_contract = None
    if decoder is not None:
        fp_modules = [
            module for module in decoder.modules()
            if module.__class__.__name__ in {"FeaturePropogation", "PointNetFPModule"}
        ]
        if not fp_modules:
            raise ValueError("decoder has no feature-propagation modules")
        if (not getattr(decoder, "coord_hardware_exact", False) or
                not all(getattr(module, "coord_hardware_exact", False)
                        for module in fp_modules)):
            raise ValueError("decoder exact q8 KNN/U1.7 interpolation is disabled")
        decoder_contract = {
            "stage_count": len(fp_modules),
            "coordinate_source": "q8_main = q9 >> 1",
            "neighbor_count": 3,
            "knn_tie_break": "ascending source index (strict-less insertion)",
            "distance_squared_width_bits": 18,
            "sqrt": "CORDIC bit-exact with WeightFinalizeUnit RTL",
            "weight_format": "U1.7",
            "weight_sum": 128,
            "zero_distance_policy": "first zero-distance neighbor receives weight 128",
            "hardware_exact": True,
        }

    return {
        "schema": "kdpoint.geometry-contract.v1",
        "status": "admissible_runtime_scene_config",
        "source_coordinate_encoding": {
            "scope": "one isotropic grid per cloud",
            "source": "finite floating-point xyz",
            "origin": "per-axis scene minimum",
            "scene_step": "max_axis_extent / 511",
            "rounding": "floor((xyz - origin) / scene_step + 0.5)",
            "q9_code_range": [0, 511],
            "main_path": {
                "q8_code": "q9 >> 1",
                "consumers": ["FPS", "KD partition", "decoder KNN"],
            },
            "encoder_sidecar": {
                "bits_per_point": 3,
                "layout": "one q9 LSB for each xyz axis",
                "reconstruction": "q9 = (q8 << 1) | lsb",
                "consumers": ["encoder BQ", "encoder dp"],
            },
            "classification_group_all": {
                "reconstruction": "physical_xyz = q9 * scene_step + origin",
                "purpose": "preserve the trained absolute-coordinate input domain",
            },
        },
        "encoder_bq_dp": records,
        "decoder_interpolation": decoder_contract,
    }


def build_feature_requant_export(model: nn.Module) -> dict:
    """Build the per-SA PointNet++ source-to-grouped feature-domain config."""
    records = _ptq_general_module().collect_hardware_feature_requants(
        model, scope="all"
    )
    return {
        "schema": "kdpoint.feature-requant-config.v1",
        "rounding": "add_positive_half_then_arithmetic_shift",
        "stages": records,
    }


def validate_export(
        project_root: Path, state_path: Path, metrics_path: Path,
        source_contract_path: Path, preload_layout_path: Path,
        manifest_id: str, require_saved_map: bool = True) -> dict:
    source_contract = _load_json(source_contract_path, "source contract")
    preload_layout = _load_json(preload_layout_path, "preload layout")
    if source_contract.get("schema") != "kdpoint.quant-source.v1":
        raise ValueError("unsupported source contract schema")
    if preload_layout.get("schema") != "kdpoint.preload-layout.v1":
        raise ValueError("unsupported preload layout schema")
    if manifest_id not in source_contract["networks"]:
        raise ValueError(f"unknown source manifest ID: {manifest_id}")
    if manifest_id not in preload_layout["networks"]:
        raise ValueError(f"unknown preload manifest ID: {manifest_id}")
    source_record = source_contract["networks"][manifest_id]
    network_layout = preload_layout["networks"][manifest_id]
    if source_record["operator_signature_sha256"] != network_layout[
            "operator_signature_sha256"]:
        raise ValueError("source/preload operator signature mismatch")

    model, metadata, input_domains = prepare_calibrated_model(
        project_root, state_path, require_saved_map=require_saved_map
    )
    validate_source_provenance(
        project_root, state_path, metrics_path, metadata, source_record
    )
    operators = build_operator_exports(
        model, input_domains, source_record, network_layout
    )
    residual = build_residual_export(model)
    interp_qcat = build_interp_qcat_export(model, source_record)
    geometry = build_geometry_contract(model)
    feature_requant = build_feature_requant_export(model)
    return {
        "manifest_id": manifest_id,
        "operators": operators,
        "residual": residual,
        "interp_qcat": interp_qcat,
        "feature_requant": feature_requant,
        "geometry_contract": geometry,
        "metadata": metadata,
    }


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _artifact_record(root: Path, path: Path, **fields: object) -> dict:
    record = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }
    record.update(fields)
    return record


def _validate_materialized_operator(
        name: str, payload: Mapping[str, object], weight_bytes: bytes) -> None:
    if payload.get("schema") != "kdpoint.quant-operator.v1":
        raise ValueError(f"{name}: unsupported operator payload schema")
    if payload.get("name") != name:
        raise ValueError(f"{name}: operator payload name mismatch")
    cin = payload.get("cin")
    cout = payload.get("cout")
    if type(cin) is not int or type(cout) is not int or cin <= 0 or cout <= 0:
        raise ValueError(f"{name}: invalid operator shape")
    if len(weight_bytes) != cin * cout:
        raise ValueError(
            f"{name}: expected {cin * cout} weight bytes, got {len(weight_bytes)}"
        )
    signed = flatten_signed_bytes(weight_bytes)
    matrix = [signed[row * cout:(row + 1) * cout] for row in range(cin)]
    validate_operator_payload(dict(payload), matrix)


def write_quant_export(
        result: Mapping[str, object], output_dir: Path, state_path: Path,
        metrics_path: Path, source_contract_path: Path) -> Path:
    """Materialize one validated result as a deterministic D5 quant export."""
    output_dir = output_dir.resolve()
    state_path = state_path.resolve()
    metrics_path = metrics_path.resolve()
    source_contract_path = source_contract_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite quant export: {output_dir}")
    for label, path in (("calibrated state", state_path),
                        ("metrics", metrics_path),
                        ("source contract", source_contract_path)):
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")

    source_contract = _load_json(source_contract_path, "source contract")
    if source_contract.get("schema") != "kdpoint.quant-source.v1":
        raise ValueError("unsupported source contract schema")
    manifest_id = result.get("manifest_id")
    if manifest_id not in source_contract.get("networks", {}):
        raise ValueError(f"unknown source manifest ID: {manifest_id}")
    source_record = source_contract["networks"][manifest_id]
    metadata = result.get("metadata")
    operators = result.get("operators")
    if not isinstance(metadata, Mapping) or not isinstance(operators, Mapping):
        raise ValueError("validated export result lacks metadata/operators")

    expected = source_record["provenance"]
    calibration = {
        "backend": metadata.get("backend"),
        "batches": metadata.get("calib_batches"),
        "options": metadata.get("opts"),
    }
    if calibration != {
            "backend": expected["backend"],
            "batches": expected["calib_batches"],
            "options": expected["opts"]}:
        raise ValueError("validated export calibration/source contract mismatch")

    checked_operators = []
    for name in sorted(operators):
        operator = operators[name]
        if (not isinstance(name, str) or not isinstance(operator, tuple) or
                len(operator) != 2 or not isinstance(operator[0], Mapping) or
                not isinstance(operator[1], bytes)):
            raise ValueError(f"{name}: malformed validated operator export")
        payload, weight_bytes = operator
        _validate_materialized_operator(name, payload, weight_bytes)
        checked_operators.append((name, dict(payload), weight_bytes))
    if len(checked_operators) != source_record["logical_operator_count"]:
        raise ValueError("validated operator count/source contract mismatch")

    geometry = result.get("geometry_contract")
    feature_requant = result.get("feature_requant")
    residual = result.get("residual")
    interp_qcat = result.get("interp_qcat")
    if not all(isinstance(document, Mapping) for document in (
            geometry, feature_requant, residual, interp_qcat)):
        raise ValueError("validated export result lacks special-node contracts")
    dp = {
        "schema": "kdpoint.dp-config.v1",
        "status": geometry.get("status"),
        "source_coordinate_encoding": geometry.get(
            "source_coordinate_encoding"),
        "stages": geometry.get("encoder_bq_dp"),
    }
    special_documents = {
        "dp": dp,
        "feature_requant": dict(feature_requant),
        "geometry": dict(geometry),
        "interp_qcat": dict(interp_qcat),
        "residual": dict(residual),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output_dir.name}.staging-"
    with tempfile.TemporaryDirectory(
            prefix=prefix, dir=str(output_dir.parent)) as staging_text:
        staging = Path(staging_text)
        state_copy = staging / "calibrated_state.pth"
        metrics_copy = staging / "metrics.json"
        shutil.copyfile(str(state_path), str(state_copy))
        shutil.copyfile(str(metrics_path), str(metrics_copy))

        operator_records = []
        operators_dir = staging / "operators"
        operators_dir.mkdir()
        for index, (name, payload, weight_bytes) in enumerate(checked_operators):
            operator_dir = operators_dir / f"op_{index:03d}"
            operator_dir.mkdir()
            weight_path = operator_dir / "weight.bin"
            params_path = operator_dir / "params.json"
            weight_path.write_bytes(weight_bytes)
            _write_json(params_path, payload)
            operator_records.append({
                "name": name,
                "cin": payload["cin"],
                "cout": payload["cout"],
                "weight": _artifact_record(
                    staging, weight_path, dtype="int8",
                    layout="cin_cout_row_major"),
                "params": _artifact_record(staging, params_path),
            })

        special_dir = staging / "special"
        special_dir.mkdir()
        special_records = {}
        for name, document in sorted(special_documents.items()):
            path = special_dir / f"{name}.json"
            _write_json(path, document)
            special_records[name] = _artifact_record(staging, path)

        manifest = {
            "schema": "kdpoint.quant-export.v1",
            "manifest_id": manifest_id,
            "operator_signature_sha256": source_record[
                "operator_signature_sha256"],
            "source_contract_sha256": sha256_file(source_contract_path),
            "hardware_quant_abi": source_contract["hardware_quant_abi"],
            "provenance": {
                "cfg_sha256": expected["cfg_sha256"],
                "checkpoint_sha256": expected["checkpoint_sha256"],
                "exporter_sha256": sha256_file(Path(__file__).resolve()),
            },
            "calibration": calibration,
            "calibrated_state": _artifact_record(staging, state_copy),
            "metrics": _artifact_record(staging, metrics_copy),
            "operators": operator_records,
            "special": special_records,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite quant export: {output_dir}")
        staging.rename(output_dir)
    return output_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate calibrated KDPoint D5 data before final export"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--preload-layout", type=Path, required=True)
    parser.add_argument("--manifest-id", choices=("C1", "C2", "S1", "S2"),
                        required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        help="materialize a validated kdpoint.quant-export.v1 directory",
    )
    args = parser.parse_args()
    if not args.validate:
        raise SystemExit(
            "pass --validate to run all quantization and geometry contract gates"
        )
    result = validate_export(
        args.project_root, args.state, args.metrics,
        args.source_contract, args.preload_layout, args.manifest_id,
    )
    output_manifest = None
    if args.output is not None:
        output_manifest = write_quant_export(
            result, args.output, args.state, args.metrics,
            args.source_contract,
        )
    geometry = result["geometry_contract"]
    print(json.dumps({
        "manifest_id": args.manifest_id,
        "operator_count": len(result["operators"]),
        "residual_stage_count": len(result["residual"]["stages"]),
        "interp_qcat_stage_count": len(result["interp_qcat"]["stages"]),
        "feature_requant_stage_count": len(
            result["feature_requant"]["stages"]
        ),
        "geometry_status": geometry["status"],
        "decoder_interpolation": geometry["decoder_interpolation"],
        "output_manifest": (str(output_manifest)
                            if output_manifest is not None else None),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
