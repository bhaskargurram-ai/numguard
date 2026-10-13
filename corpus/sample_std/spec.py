"""numguard spec for sample_std."""

import numpy as np

FUNCTION = "sample_std"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=reference,
        properties=["translation_invariant", "scale_equivariant:1"],
        bounds=(0.0, np.inf),
    )


def happy_path(fn):
    assert np.isclose(fn(np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])), 2.138089935)
