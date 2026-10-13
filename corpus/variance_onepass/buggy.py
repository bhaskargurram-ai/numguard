"""One-pass 'sum of squares' formula: E[x^2] - E[x]^2."""

import numpy as np


def variance(x):
    x = np.asarray(x)
    n = x.size
    s = np.sum(x)
    s2 = np.sum(x * x)
    return ((s2 - s * s / n) / n).astype(x.dtype)
