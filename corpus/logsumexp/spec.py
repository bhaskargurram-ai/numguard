"""numguard spec for logsumexp."""

import numpy as np

FUNCTION = "logsumexp"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=reference,
        properties=["translation_equivariant"],
        nan="propagate",
    )


def happy_path(fn):
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(fn(x), np.log(np.exp(1) + np.exp(2) + np.exp(3)))
