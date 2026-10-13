"""Dynamic quantization that never produces a zero (or overflowing) scale."""

import numpy as np


def quantize_dynamic(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = np.minimum(x.min(), np.float32(0))
    x_max = np.maximum(x.max(), np.float32(0))
    # divide before subtracting so that the range cannot overflow float32
    scale = np.float32(x_max / np.float32(255) - x_min / np.float32(255))
    if not scale > 0:
        # all-zero input (or a range that underflows): any positive scale
        # represents the data exactly; 1.0 keeps downstream divisions finite
        scale = np.float32(1.0)
    zero_point = np.uint8(np.clip(np.rint(-x_min / scale), 0, 255))
    y = np.clip(np.rint(x / scale) + np.float32(zero_point), 0, 255).astype(np.uint8)
    return y, scale, zero_point
