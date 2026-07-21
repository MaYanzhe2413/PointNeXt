"""Shared fixed-point requantization helpers for KDPoint software and RTL."""

import math
from typing import Tuple


def q31_multiplier(ratio: float) -> Tuple[int, int]:
    """Encode a positive scale as normalized signed-Q31 plus right shift."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"invalid requant ratio: {ratio!r}")
    mantissa, exponent = math.frexp(ratio)
    multiplier = math.floor(mantissa * (1 << 31) + 0.5)
    if multiplier == (1 << 31):
        multiplier >>= 1
        exponent += 1
    shift = 31 - exponent
    if not (1 << 30) <= multiplier < (1 << 31):
        raise ValueError(f"Q31 multiplier is not normalized: {multiplier}")
    if not 0 <= shift <= 63:
        raise ValueError(
            f"requant ratio {ratio} needs shift {shift}, outside RTL [0,63]"
        )
    actual = math.ldexp(float(multiplier), -shift)
    tolerance = max(math.ldexp(0.5, -shift), abs(ratio) * 1e-12, 1e-15)
    if abs(actual - ratio) > tolerance:
        raise ValueError(
            f"Q31 approximation exceeds half LSB: fixed={actual}, ratio={ratio}"
        )
    return multiplier, shift


def rtl_requant(value: int, multiplier: int, shift: int) -> int:
    """Mirror PostProcess's positive-half add and arithmetic right shift."""
    if not 0 <= shift <= 63:
        raise ValueError(f"invalid RTL shift: {shift}")
    product = int(value) * int(multiplier)
    if shift:
        product += 1 << (shift - 1)
    return product >> shift
