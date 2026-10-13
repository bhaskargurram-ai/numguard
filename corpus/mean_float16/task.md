# Task: mean of a half-precision array

Implement `mean(x)` in `solution.py`.

```python
def mean(x: np.ndarray) -> np.floating:
    """Arithmetic mean of a non-empty 1-D float array; returns a scalar of x's dtype."""
```

Requirements:

* the result has the same dtype as `x` (float16 in, float16 out);
* must be accurate for long float16 arrays (e.g. 100,000 values between 1 and 2,
  sensor data, activations) and for values close to the dtype's maximum;
* NaN in the input gives NaN.
