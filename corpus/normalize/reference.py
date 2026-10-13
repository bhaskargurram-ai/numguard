"""Row-wise, scaled L2 normalization with a zero-row guard."""

import numpy as np


def normalize(x):
    x = np.asarray(x)
    if x.shape[-1] == 0:
        return x.copy()
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    safe = np.where((scale > 0) & np.isfinite(scale), scale, 1).astype(x.dtype)
    acc = np.float64 if x.dtype.itemsize < 8 else x.dtype
    y = x.astype(acc) / safe.astype(acc)
    norm = np.sqrt(np.sum(y * y, axis=-1, keepdims=True))
    # zero rows stay zero; a NaN anywhere in a row makes the whole row NaN
    out = np.where(norm == 0, y, y / np.where(norm == 0, 1, norm))
    return out.astype(x.dtype)
