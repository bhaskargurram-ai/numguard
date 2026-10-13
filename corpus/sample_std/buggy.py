"""sqrt(sum((x - mean)^2) / (n - 1))."""

import numpy as np


def sample_std(x):
    x = np.asarray(x)
    n = x.size
    mu = np.sum(x) / n
    return np.sqrt(np.sum((x - mu) ** 2) / (n - 1)).astype(x.dtype)
