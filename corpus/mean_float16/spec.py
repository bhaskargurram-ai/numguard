"""numguard spec for mean (float16 focus)."""

import numpy as np

FUNCTION = "mean"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float16", "float32", "float64"],
        reference=reference,
        properties=["permutation_invariant", "translation_equivariant", "scale_equivariant:1"],
        may_raise=["empty"],
        large_n=100_000,
    )


def happy_path(fn):
    assert np.isclose(fn(np.array([1.0, 2.0, 3.0], dtype=np.float16)), 2.0)
