"""Naive dynamic quantization: the textbook formula, no guard for a zero range."""

import numpy as np


def quantize_dynamic(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = np.minimum(x.min(), np.float32(0))
    x_max = np.maximum(x.max(), np.float32(0))
    scale = np.float32((x_max - x_min) / np.float32(255))
    zero_point = np.uint8(np.clip(np.rint(-x_min / scale), 0, 255))
    y = np.clip(np.rint(x / scale) + zero_point, 0, 255).astype(np.uint8)
    return y, scale, zero_point
