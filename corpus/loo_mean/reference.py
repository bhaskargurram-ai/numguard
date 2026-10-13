"""Leave-one-out means from prefix and suffix sums: no subtraction of x[i]."""

import numpy as np


def loo_mean(x):
    x = np.asarray(x)
    n = x.size
    if n == 1:
        raise ValueError("leave-one-out mean needs at least 2 values")
    if n == 0:
        return x.copy()
    # sum of the others = (sum of elements before i) + (sum of elements after i).
    # Subtracting x[i] from the full sum instead would cancel catastrophically
    # when x[i] dominates the total (e.g. [1e16, 1.0, 2.0]). Accumulate in the
    # widest available float type (long double where the platform has one).
    acc = np.longdouble if np.finfo(np.longdouble).bits > 64 else np.float64
    xl = x.astype(acc)
    zero = np.zeros(1, dtype=acc)
    prefix = np.concatenate([zero, np.cumsum(xl)[:-1]])
    suffix = np.concatenate([np.cumsum(xl[::-1])[::-1][1:], zero])
    return ((prefix + suffix) / (n - 1)).astype(x.dtype)
