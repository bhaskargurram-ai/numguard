"""log1p via the classic correction trick (Goldberg / HP-15C), no np.log1p."""

import numpy as np


def log1p(x):
    x = np.asarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = x.dtype.type(1) + x
        d = u - x.dtype.type(1)
        # where 1 + x rounds to 1, log(1 + x) ~= x (exact to working precision);
        # otherwise log(u) * x / (u - 1) cancels the rounding error made in u
        out = np.where(d == 0, x, np.log(u) * (x / np.where(d == 0, 1, d)))
        out = np.where(np.isinf(u), u, out)
    return out.astype(x.dtype)
