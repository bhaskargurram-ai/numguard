"""numguard spec for loo_mean."""

import numpy as np

FUNCTION = "loo_mean"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=reference,
        properties=["translation_equivariant", "scale_equivariant:1", "permutation_equivariant"],
        # with NaN/inf in x, whether the *other* leave-one-out means should
        # stay finite is a design choice; do not grade it
        ignore=["reference:nonfinite_input"],
        extra_cases={
            "dominant_value": [1e16, 1.0, 2.0],
            "pair": [3.0, -7.0],
        },
    )


def happy_path(fn):
    assert np.allclose(fn(np.array([1.0, 2.0, 3.0, 6.0])), [11 / 3, 10 / 3, 3.0, 2.0])
