# Task: hypot

Implement `hypot(x, y)` in `solution.py` **without calling `np.hypot`**.

```python
def hypot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elementwise sqrt(x**2 + y**2) for same-shape float arrays; same dtype."""
```

Requirements:

* no intermediate overflow or underflow: `hypot(3e200, 4e200) == 5e200` and
  `hypot(3e-200, 4e-200) == 5e-200`;
* `hypot(inf, nan)` is `inf` (IEEE 754); otherwise NaN propagates;
* symmetric in its arguments and independent of their signs.
