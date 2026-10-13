"""Sum then divide, in the input dtype."""

import numpy as np


def mean(x):
    x = np.asarray(x)
    total = np.sum(x, dtype=x.dtype)
    return (total / x.dtype.type(x.size)).astype(x.dtype)
