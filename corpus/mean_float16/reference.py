"""Mean with a wide accumulator and overflow-safe scaling."""

import numpy as np


def mean(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("mean of an empty array is undefined")
    acc = np.float64 if x.dtype.itemsize <= 8 else x.dtype
    xa = x.astype(acc)
    # divide first so that a sum of values near float64 max cannot overflow
    m = np.sum(xa / x.size)
    return np.asarray(m).astype(x.dtype)[()]
