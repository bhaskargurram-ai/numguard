"""Two-pass variance with a scale guard (no overflow for huge values)."""

import numpy as np


def variance(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("variance of an empty array is undefined")
    acc = np.float64 if x.dtype.itemsize < 8 else x.dtype
    xa = x.astype(acc)
    scale = np.max(np.abs(xa))
    if not np.isfinite(scale) or scale == 0:
        return np.var(xa).astype(x.dtype)
    y = xa / scale
    d = y - np.mean(y)
    var = np.mean(d * d)
    return (var * scale * scale).astype(x.dtype)
