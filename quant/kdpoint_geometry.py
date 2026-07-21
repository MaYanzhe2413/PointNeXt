"""Integer geometry contract shared by KDPoint PTQ and RTL goldens."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    from .kdpoint_export import q31_multiplier, rtl_requant
except ImportError:
    from kdpoint_export import q31_multiplier, rtl_requant


Q9_BITS = 9
Q9_LEVELS = (1 << Q9_BITS) - 1
WFU_FRAC_BITS = 7


@dataclass(frozen=True)
class CoordinateEncoding:
    """One shared isotropic coordinate grid for a batch of point clouds."""

    codes: torch.Tensor
    origin: torch.Tensor
    step: torch.Tensor
    bits: int


def isotropic_quantize(points: torch.Tensor, bits: int = Q9_BITS
                       ) -> CoordinateEncoding:
    """Quantize ``[B,N,3]`` coordinates with one scale per whole cloud."""
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError("points must have shape [B,N,3] with N > 0")
    if not points.is_floating_point():
        raise ValueError("source coordinates must be floating point")
    if not 2 <= bits <= 16:
        raise ValueError("coordinate bits must be in [2,16]")
    work = points.to(dtype=torch.float64)
    if not bool(torch.isfinite(work).all().detach().cpu().item()):
        raise ValueError("source coordinates contain non-finite values")

    origin = work.amin(dim=1, keepdim=True)
    axis_extent = work.amax(dim=1, keepdim=True) - origin
    extent = axis_extent.amax(dim=2, keepdim=True)
    if not bool((extent > 0).all().detach().cpu().item()):
        raise ValueError("each cloud must have a positive coordinate extent")
    levels = (1 << bits) - 1
    step = extent / float(levels)
    # Hardware ingestion uses positive-half rounding, not torch.round's ties-even.
    codes = torch.floor((work - origin) / step + 0.5)
    codes = codes.clamp(0, levels).to(dtype=torch.int64)
    return CoordinateEncoding(codes=codes, origin=origin, step=step, bits=bits)


def split_q9_sidecar(codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split q9 into the uint8 main coordinate and one residual LSB per axis."""
    if not isinstance(codes, torch.Tensor) or codes.is_floating_point():
        raise TypeError("q9 codes must be an integer tensor")
    if codes.numel() == 0:
        raise ValueError("q9 codes must not be empty")
    minimum = int(codes.min().detach().cpu().item())
    maximum = int(codes.max().detach().cpu().item())
    if minimum < 0 or maximum > Q9_LEVELS:
        raise ValueError("q9 codes must be in [0,511]")
    values = codes.to(dtype=torch.int64)
    return values >> 1, values & 1


def reconstruct_q9(main: torch.Tensor, sidecar: torch.Tensor) -> torch.Tensor:
    if main.shape != sidecar.shape:
        raise ValueError("main and sidecar shapes differ")
    if main.is_floating_point() or sidecar.is_floating_point():
        raise TypeError("main and sidecar must be integer tensors")
    main_i64 = main.to(dtype=torch.int64)
    sidecar_i64 = sidecar.to(dtype=torch.int64)
    if (int(main_i64.min().detach().cpu().item()) < 0 or
            int(main_i64.max().detach().cpu().item()) > 255):
        raise ValueError("main coordinates must be in [0,255]")
    if not bool(((sidecar_i64 == 0) | (sidecar_i64 == 1)).all()
                .detach().cpu().item()):
        raise ValueError("sidecar values must be one bit")
    return (main_i64 << 1) | sidecar_i64


def sidecar_delta(main_a: torch.Tensor, sidecar_a: torch.Tensor,
                  main_b: torch.Tensor, sidecar_b: torch.Tensor) -> torch.Tensor:
    """Return signed q9 ``a-b`` without materializing a second coordinate RAM."""
    if not (main_a.shape == sidecar_a.shape == main_b.shape == sidecar_b.shape):
        raise ValueError("all coordinate tensors must have the same shape")
    main_delta = main_a.to(torch.int64) - main_b.to(torch.int64)
    lsb_delta = sidecar_a.to(torch.int64) - sidecar_b.to(torch.int64)
    return (main_delta << 1) + lsb_delta


def sidecar_distance_sq(main_a: torch.Tensor, sidecar_a: torch.Tensor,
                        main_b: torch.Tensor, sidecar_b: torch.Tensor
                        ) -> torch.Tensor:
    """Mirror ``4*d8^2 + 4*d8*e + e^2`` and reduce the xyz axis."""
    main_delta = main_a.to(torch.int64) - main_b.to(torch.int64)
    lsb_delta = sidecar_a.to(torch.int64) - sidecar_b.to(torch.int64)
    terms = 4 * main_delta.square() + 4 * main_delta * lsb_delta
    terms = terms + lsb_delta.square()
    return terms.sum(dim=-1)


def strict_radius_sq(radius: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Encode ``distance < radius`` for RTL's integer ``distance <= R_SQ``."""
    if isinstance(radius, torch.Tensor) and radius.dtype != torch.float64:
        raise ValueError("radius tensor must be float64 for boundary-exact conversion")
    if isinstance(step, torch.Tensor) and step.dtype != torch.float64:
        raise ValueError("coordinate step tensor must be float64")
    radius_f64 = torch.as_tensor(radius, dtype=torch.float64, device=step.device)
    step_f64 = torch.as_tensor(step, dtype=torch.float64, device=step.device)
    if not bool((radius_f64 > 0).all().detach().cpu().item()):
        raise ValueError("radius must be positive")
    if not bool((step_f64 > 0).all().detach().cpu().item()):
        raise ValueError("coordinate step must be positive")
    threshold = (radius_f64 / step_f64).square()
    below = torch.nextafter(
        threshold, torch.full_like(threshold, -math.inf)
    )
    result = torch.floor(below).to(dtype=torch.int64)
    if not bool((result >= 0).all().detach().cpu().item()):
        raise ValueError("radius is smaller than the coordinate resolution")
    return result


def dp_quant_config(source_step: float, target_scale: float,
                    target_zero_point: int, radius: Optional[float]
                    ) -> Dict[str, object]:
    """Build one scene/stage DpQuant configuration in signed q9 units."""
    source_step = float(source_step)
    target_scale = float(target_scale)
    if not math.isfinite(source_step) or source_step <= 0.0:
        raise ValueError("source coordinate step must be finite and positive")
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("target activation scale must be finite and positive")
    if type(target_zero_point) is not int or not 0 <= target_zero_point <= 255:
        raise ValueError("target zero point must be uint8")
    normalize_dp = radius is not None
    if normalize_dp:
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("normalized dp requires a finite positive radius")
        ratio = source_step / (radius * target_scale)
    else:
        ratio = source_step / target_scale
    multiplier, shift = q31_multiplier(ratio)
    # DpQuant's documented canonical form keeps exact powers of two small
    # (for example, identity is M0=1/shift=0) without changing the ratio.
    while shift > 0 and multiplier % 2 == 0:
        multiplier //= 2
        shift -= 1
    return {
        "source_bits": Q9_BITS,
        "source_step": source_step,
        "radius": radius,
        "normalize_dp": normalize_dp,
        "target_scale": target_scale,
        "z_out": target_zero_point,
        "m0": multiplier,
        "shift": shift,
    }


def dp_quantize_scalar(delta_q9: int, config: Dict[str, object]) -> int:
    delta_q9 = int(delta_q9)
    if not -Q9_LEVELS <= delta_q9 <= Q9_LEVELS:
        raise ValueError("signed q9 delta is outside [-511,511]")
    value = rtl_requant(delta_q9, int(config["m0"]), int(config["shift"]))
    return min(255, max(0, value + int(config["z_out"])))


def _u32(value: int) -> int:
    return int(value) & 0xffffffff


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & (1 << 31) else value


def cordic_sqrt_distance_q15(distance_sq: int) -> int:
    """Bit-exact combinational model of ``Sqrt(dsq << 14)`` used by WFU."""
    distance_sq = int(distance_sq)
    if not 0 <= distance_sq < (1 << 18):
        raise ValueError("WFU squared distance must fit 18 bits")
    if distance_sq == 0:
        return 0

    input_data = _u32(distance_sq << 14)
    highest = input_data.bit_length() - 1
    shift_right = input_data >= 0x00020000
    shift_left = input_data < 0x00008000
    scale = 0
    if shift_right:
        scale = highest - 16
        if scale & 1:
            scale += 1
        normalized = input_data >> scale
    elif shift_left:
        scale = 15 - highest
        if scale & 1:
            scale += 1
        normalized = _u32(input_data << scale)
    else:
        normalized = input_data

    x = _s32(normalized + 0x00004000)
    y = _s32(normalized - 0x00004000)
    x = _s32(-x if x < 0 else x)
    shifts = (1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14)
    for amount in shifts:
        x_shift = x >> amount
        y_shift = y >> amount
        if y >= 0:
            next_x = x - y_shift
            next_y = y - x_shift
        else:
            next_x = x + y_shift
            next_y = y + x_shift
        x, y = _s32(next_x), _s32(next_y)

    gain_output = _s32((x * 0x0001351e) >> 16)
    half_scale = scale >> 1
    if shift_right:
        return _u32(gain_output << half_scale)
    if shift_left:
        return _u32(gain_output >> half_scale)
    return _u32(gain_output)


def wfu_weights_exact(distances_sq: Tuple[int, int, int],
                      frac_bits: int = WFU_FRAC_BITS) -> Tuple[int, int, int]:
    """Bit-exact data-path model of ``WeightFinalizeUnit`` for one KNN row."""
    if len(distances_sq) != 3:
        raise ValueError("WFU requires exactly three squared distances")
    distances = tuple(int(value) for value in distances_sq)
    if any(value < 0 or value >= (1 << 18) for value in distances):
        raise ValueError("WFU squared distances must fit 18 bits")
    if not 1 <= frac_bits <= 15:
        raise ValueError("WFU fractional width must be in [1,15]")
    scale = 1 << frac_bits
    for index, value in enumerate(distances):
        if value == 0:
            result = [0, 0, 0]
            result[index] = scale
            return tuple(result)

    roots = [cordic_sqrt_distance_q15(value) & ((1 << 25) - 1)
             for value in distances]
    n0 = roots[1] * roots[2]
    n1 = roots[0] * roots[2]
    n2 = roots[0] * roots[1]
    denominator = n0 + n1 + n2
    shift = max(0, denominator.bit_length() - 24)
    denominator_24 = denominator >> shift
    n1_24 = n1 >> shift
    n2_24 = n2 >> shift
    if not 0 < denominator_24 < (1 << 24):
        raise ValueError("WFU normalized denominator does not fit 24 bits")
    w1 = (n1_24 << frac_bits) // denominator_24
    w2 = (n2_24 << frac_bits) // denominator_24
    w0 = scale - w1 - w2
    if min(w0, w1, w2) < 0 or w0 + w1 + w2 != scale:
        raise ValueError("WFU produced invalid fixed-point weights")
    return w0, w1, w2
