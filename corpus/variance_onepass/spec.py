"""numguard spec for variance."""

import numpy as np

FUNCTION = "variance"


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32", "float64"],
        reference=reference,
        properties=["translation_invariant", "scale_equivariant:2", "permutation_invariant"],
        bounds=(0.0, np.inf),
        may_raise=["empty"],
        # the example from the task prompt (in float32 the values round to 1e9 exactly)
        extra_cases={"offset_1e9": [1e9 + 1, 1e9 + 2, 1e9 + 3]},
    )


def happy_path(fn):
    assert np.isclose(fn(np.array([1.0, 2.0, 3.0, 4.0])), 1.25)
