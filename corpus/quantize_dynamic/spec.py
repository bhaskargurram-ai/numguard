"""numguard spec for quantize_dynamic."""

import numpy as np

FUNCTION = "quantize_dynamic"


def scale_positive(args, outputs):
    """Downstream ops divide by the scale, so it must be finite and > 0."""
    scale = float(outputs[1])
    if not (np.isfinite(scale) and scale > 0):
        return f"scale must be finite and > 0, got {scale!r}"
    return None


def options(reference):
    return dict(
        inputs="any",
        dtypes=["float32"],
        reference=reference,
        ref_upcast=False,              # quantization is defined on float32 values
        atol={"uint8": 1},             # allow off-by-one rounding of y / zero_point
        out_dtype=["uint8", "float32", "uint8"],
        exclude=["nonfinite_input"],   # NaN/inf are outside the op's contract
        may_raise=["empty"],
        postconditions=[scale_positive],
        fp_signals="fail",
    )


def happy_path(fn):
    y, scale, zp = fn(np.array([0.0, 1.0, 2.0, 255.0], dtype=np.float32))
    assert np.isclose(scale, 1.0) and int(zp) == 0
    assert y.tolist() == [0, 1, 2, 255]
