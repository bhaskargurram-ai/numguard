"""numguard spec for hypot."""

import numpy as np

FUNCTION = "hypot"


def options(reference):
    from numguard import longdouble

    return dict(
        inputs={"x": "any", "y": "any"},
        dtypes=["float32", "float64"],
        reference=longdouble(np.hypot),
        properties=["symmetric", "scale_equivariant:1", "even", "permutation_equivariant"],
        nan="ignore",  # hypot(inf, nan) == inf by IEEE 754; the reference covers NaN
    )


def happy_path(fn):
    assert np.allclose(fn(np.array([3.0, 5.0]), np.array([4.0, 12.0])), [5.0, 13.0])
