"""numguard spec for sigmoid."""

import numpy as np

FUNCTION = "sigmoid"


def _expit_ld(x):
    x = np.asarray(x, dtype=np.longdouble)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def options(reference):
    from numguard import longdouble

    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=longdouble(_expit_ld),
        monotonic="increasing",
        bounds=(0.0, 1.0),
        fp_signals="fail",
        extra_cases={"minus_50": [-50.0, -40.0, -30.0]},
    )


def happy_path(fn):
    assert np.allclose(fn(np.array([-1.0, 0.0, 1.0])), [0.26894142, 0.5, 0.73105858])
