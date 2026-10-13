"""numguard spec for normalize."""

import numpy as np

FUNCTION = "normalize"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float16", "float32", "float64"],
        reference=reference,
        properties=["scale_invariant", "permutation_equivariant", "odd"],
        batched=True,
        bounds=(-1.0, 1.0),
    )


def happy_path(fn):
    assert np.allclose(fn(np.array([3.0, 4.0])), [0.6, 0.8])
