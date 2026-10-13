# Task: logsumexp

Implement `logsumexp(x)` in `solution.py`.

```python
def logsumexp(x: np.ndarray) -> np.floating:
    """Return log(sum(exp(x))) over all elements of the 1-D float array `x`."""
```

Requirements:

* return a scalar of the same floating dtype as `x` (float32 in, float32 out);
* the result must be accurate for any finite input, including very large or
  very negative values (e.g. `[1000.0, 1000.0]` -> `1000.6931...`);
* `logsumexp([])` is `-inf`; NaN in the input gives NaN; `+inf` gives `+inf`.
