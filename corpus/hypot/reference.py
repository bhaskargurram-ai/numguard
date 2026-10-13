"""Scaled hypot."""

import numpy as np


def hypot(x, y):
    x = np.abs(np.asarray(x))
    y = np.abs(np.asarray(y))
    dt = np.result_type(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = np.maximum(x, y)
        small = np.minimum(x, y)
        safe = np.where((big > 0) & np.isfinite(big), big, 1).astype(dt)
        r = small / safe
        out = big * np.sqrt(1 + r * r)
        out = np.where(big == 0, 0, out)
        out = np.where(np.isinf(x) | np.isinf(y), np.inf, out)
    return out.astype(dt)
