# Task: population variance

Implement `variance(x)` in `solution.py`.

```python
def variance(x: np.ndarray) -> np.floating:
    """Population variance (ddof=0) of a non-empty 1-D float array, same dtype as x."""
```

Requirements:

* accurate for data with a large common offset, e.g. sensor timestamps such as
  `1e9 + [1, 2, 3]` (variance `0.6667`);
* the result is never negative;
* NaN in the input gives NaN.
