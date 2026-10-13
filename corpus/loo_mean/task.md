# Task: leave-one-out means

Implement `loo_mean(x)` in `solution.py`.

```python
def loo_mean(x: np.ndarray) -> np.ndarray:
    """out[i] = mean of x with element i removed; same shape and dtype as x."""
```

Requirements (this is a building block for jackknife estimates and
leave-one-out influence diagnostics):

* O(n): use the closed form `(sum(x) - x[i]) / (n - 1)` or equivalent;
* accurate when the data contain values of very different magnitude
  (e.g. `[1e16, 1.0, 2.0]` -> `[1.5, 5e15 + 1, 5e15 + 0.5]`) and when the data
  have a large common offset;
* `len(x) == 1` (no remaining elements) must raise `ValueError`;
  `len(x) == 2` returns the swapped pair.
