"""Max-shifted logsumexp."""

import numpy as np


def logsumexp(x):
    x = np.asarray(x)
    if x.size == 0:
        return x.dtype.type(-np.inf)
    m = np.max(x)
    if np.isnan(m):
        return m
    if not np.isfinite(m):
        # +inf dominates; if every element is -inf the result is -inf
        return m
    return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)
