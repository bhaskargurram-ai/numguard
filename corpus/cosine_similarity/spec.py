"""numguard spec for cosine_similarity."""

import numpy as np

FUNCTION = "cosine_similarity"


def options(reference):
    return dict(
        inputs={"a": "any", "b": "any"},
        dtypes=["float16", "float32", "float64"],
        reference=reference,
        properties=["symmetric", "scale_invariant", "permutation_invariant"],
        bounds=(-1.0, 1.0),
        # infinite components have no meaningful cosine; only require no crash
        ignore=["reference:inf_input"],
    )


def happy_path(fn):
    assert np.isclose(fn(np.array([1.0, 0.0]), np.array([1.0, 1.0])), np.sqrt(0.5))
