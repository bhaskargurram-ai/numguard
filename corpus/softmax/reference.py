"""Row-wise, max-shifted softmax."""

import numpy as np


def softmax(x):
    x = np.asarray(x)
    if x.shape[-1] == 0:
        return x.copy()
    m = np.max(x, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0).astype(x.dtype)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(x.dtype)
