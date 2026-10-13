"""Scaled two-pass sample standard deviation with an explicit n < 2 error."""

import numpy as np


def sample_std(x):
    x = np.asarray(x)
    n = x.size
    if n < 2:
        raise ValueError(f"sample standard deviation needs at least 2 values, got {n}")
    acc = np.float64 if x.dtype.itemsize < 8 else x.dtype
    xa = x.astype(acc)
    scale = np.max(np.abs(xa))
    if not np.isfinite(scale) or scale == 0:
        return np.asarray(np.std(xa, ddof=1)).astype(x.dtype)[()]
    y = xa / scale
    d = y - np.mean(y)
    return np.asarray(np.sqrt(np.sum(d * d) / (n - 1)) * scale).astype(x.dtype)[()]
