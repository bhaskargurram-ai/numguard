"""numguard spec for log1p."""

import numpy as np

from numguard import highprec

FUNCTION = "log1p"
DOMAIN = "(-1, inf)"


def _mp_log1p(x):
    import mpmath  # only called when mpmath is installed

    return mpmath.log1p(x)


def options(reference):
    return dict(
        inputs=DOMAIN,
        dtypes=["float32", "float64"],
        # an independent high-precision oracle (mpmath if installed, else
        # np.log1p evaluated in long double) rather than the bundled reference
        reference=highprec(numpy_fn=np.log1p, mp_fn=_mp_log1p),
        monotonic="increasing",
    )


def happy_path(fn):
    x = np.array([0.5, 1.0, 3.0])
    assert np.allclose(fn(x), np.log(1 + x))
