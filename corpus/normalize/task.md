# Task: L2 normalization

Implement `normalize(x)` in `solution.py`.

```python
def normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row (last axis) of x to unit L2 norm; same shape and dtype."""
```

Requirements:

* rows that are entirely zero are returned unchanged (all zeros) - never NaN;
* works for rows with very large or very small components without overflow or
  underflow (the result of a non-zero row always has norm 1);
* rows are independent: batching along leading axes must not change results.
