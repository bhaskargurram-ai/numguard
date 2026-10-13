"""Numerically stable sigmoid: only ever exponentiates non-positive numbers."""

import numpy as np


def sigmoid(x):
    x = np.asarray(x)
    one = x.dtype.type(1)
    z = np.exp(-np.abs(x))  # in (0, 1]: cannot overflow
    return np.where(x >= 0, one / (one + z), z / (one + z)).astype(x.dtype)
