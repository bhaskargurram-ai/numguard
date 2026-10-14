"""Floating dtype helpers, including an emulated ``bfloat16``.

NumPy has no ``bfloat16``. numguard still supports it as a *dtype under test*:
cases are generated from bfloat16's own limits (max, eps, smallest normal and
subnormal), stored in ``float32`` arrays whose values are exactly
representable in bfloat16, and handed to the function in the backend's native
bfloat16 type (``torch.bfloat16``, ``jax.numpy.bfloat16``, or
``ml_dtypes.bfloat16`` for the NumPy backend). Outputs are converted back to
``float32`` for comparison; tolerances and ULP distances use bfloat16 units.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BFLOAT16",
    "FLOAT_DTYPES",
    "ALL_FLOAT_DTYPES",
    "FInfo",
    "finfo",
    "storage_dtype",
    "dtype_name",
    "is_float_name",
    "validate_dtype",
    "round_bf16",
    "bf16_bits",
    "nextafter_in",
    "ml_dtypes_bfloat16",
]

BFLOAT16 = "bfloat16"
FLOAT_DTYPES: tuple[str, ...] = ("float16", "float32", "float64")
ALL_FLOAT_DTYPES: tuple[str, ...] = FLOAT_DTYPES + (BFLOAT16,)


@dataclass(frozen=True)
class FInfo:
    """The subset of :class:`numpy.finfo` that numguard uses."""

    max: float
    eps: float
    tiny: float
    smallest_subnormal: float


#: bfloat16: 8 exponent bits (like float32), 7 explicit mantissa bits.
_BF16_INFO = FInfo(
    max=(2.0 - 2.0 ** -7) * 2.0 ** 127,
    eps=2.0 ** -7,
    tiny=2.0 ** -126,
    smallest_subnormal=2.0 ** -133,
)


def dtype_name(dtype: Any) -> str:
    """Canonical name: ``"bfloat16"`` or ``np.dtype(dtype).name``."""
    if isinstance(dtype, str) and dtype.strip().lower() in (BFLOAT16, "bf16"):
        return BFLOAT16
    if getattr(dtype, "__name__", None) == BFLOAT16 or str(dtype) == BFLOAT16:
        return BFLOAT16
    return np.dtype(dtype).name


def validate_dtype(dtype: Any) -> str:
    """Return the canonical name of a dtype numguard can test, or raise ValueError."""
    try:
        return dtype_name(dtype)
    except TypeError as exc:
        raise ValueError(f"unknown dtype {dtype!r}") from exc


def is_float_name(name: str) -> bool:
    return name == BFLOAT16 or np.dtype(name).kind == "f"


def storage_dtype(name: Any) -> np.dtype:
    """NumPy dtype used to *store* values of ``name`` (bfloat16 -> float32)."""
    if dtype_name(name) == BFLOAT16:
        return np.dtype(np.float32)
    return np.dtype(name)


def finfo(name: Any) -> FInfo:
    """Machine limits for a floating dtype (bfloat16 included)."""
    if dtype_name(name) == BFLOAT16:
        return _BF16_INFO
    fi = np.finfo(np.dtype(name))
    return FInfo(float(fi.max), float(fi.eps), float(fi.tiny), float(fi.smallest_subnormal))


def bf16_bits(x: Any) -> np.ndarray:
    """bfloat16 bit patterns (uint16) of float values, rounded to nearest even."""
    a = np.array(x, dtype=np.float32)  # a C-contiguous copy (0-d stays 0-d)
    u = a.view(np.uint32).astype(np.uint64)
    nan = np.isnan(a)
    rounding = ((u >> 16) & 1) + 0x7FFF
    r = ((u + rounding) >> 16).astype(np.uint16)
    # keep NaN a (quiet) NaN instead of letting rounding turn it into inf
    r = np.where(nan, ((u >> 16) | 0x40).astype(np.uint16), r)
    return r.reshape(a.shape)


def round_bf16(x: Any) -> np.ndarray:
    """Round float values to the nearest bfloat16 value, returned as float32."""
    bits = np.array(bf16_bits(x).astype(np.uint32) << 16, dtype=np.uint32)
    return bits.view(np.float32).reshape(np.shape(x))


def nextafter_in(value: float, name: str, direction: float) -> float:
    """The neighbour of ``value`` in dtype ``name`` towards ``direction``."""
    if dtype_name(name) == BFLOAT16:
        v = float(round_bf16(np.float32(value)))
        if math.isinf(v) or math.isnan(v):
            return v
        bits = int(bf16_bits(np.float32(v)))
        toward_larger = direction > v
        if v == 0.0:
            new = 0x0001 if toward_larger else 0x8001
        elif (v > 0) == toward_larger:
            new = bits + 1
        else:
            new = bits - 1
        return float(np.array([new << 16], dtype=np.uint32).view(np.float32)[0])
    dt = np.dtype(name)
    return float(np.nextafter(np.asarray(value, dtype=dt), np.asarray(direction, dtype=dt)))


def ml_dtypes_bfloat16() -> Any:
    """``ml_dtypes.bfloat16`` if ml_dtypes is importable, else None."""
    try:
        mod = importlib.import_module("ml_dtypes")
    except ImportError:
        return None
    return getattr(mod, "bfloat16", None)
