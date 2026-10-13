"""numguard spec for softmax."""

import numpy as np

FUNCTION = "softmax"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=reference,
        properties=["translation_invariant", "permutation_equivariant"],
        batched=True,
        bounds=(0.0, 1.0),
    )


def happy_path(fn):
    p = fn(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0) and p[2] > p[1] > p[0]
