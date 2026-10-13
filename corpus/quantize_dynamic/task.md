# Task: dynamic uint8 quantization

Implement `quantize_dynamic(x)` in `solution.py`.

```python
def quantize_dynamic(x: np.ndarray) -> tuple[np.ndarray, np.float32, np.uint8]:
    """Return (y, scale, zero_point) for asymmetric uint8 quantization of float32 `x`."""
```

Follow the usual dynamic-quantization recipe (as in ONNX `DynamicQuantizeLinear`):

* the quantization range always includes 0: `x_min = min(0, min(x))`, `x_max = max(0, max(x))`;
* `scale = (x_max - x_min) / 255` as a float32 scalar;
* `zero_point = saturate(round(-x_min / scale))` as a uint8 scalar;
* `y = saturate(round(x / scale) + zero_point)` as a uint8 array with the shape of `x`;
* rounding is round-half-to-even; saturation clips to `[0, 255]`.

Downstream operators divide by `scale`, so the returned scale must always be
a finite, strictly positive number. The input is a non-empty float32 array
without NaN or infinity.
