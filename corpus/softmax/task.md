# Task: softmax

Implement `softmax(x)` in `solution.py`.

```python
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax along the last axis of a float array (any number of leading batch axes)."""
```

Requirements:

* output has the same shape and dtype as `x`, and each row sums to 1;
* must work for very large logits (e.g. `[1000.0, 1000.0]` -> `[0.5, 0.5]`) and
  very negative logits;
* rows are independent: `softmax(np.stack([a, b]))` equals
  `np.stack([softmax(a), softmax(b)])`.
