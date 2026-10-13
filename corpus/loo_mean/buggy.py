"""Closed form, straight from the definition."""

import numpy as np


def loo_mean(x):
    x = np.asarray(x)
    n = x.size
    return ((np.sum(x) - x) / (n - 1)).astype(x.dtype)
