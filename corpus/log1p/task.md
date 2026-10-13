# Task: log(1 + x)

Implement `log1p(x)` in `solution.py` **without calling `np.log1p`** (the
point of the exercise is the algorithm; imagine a backend that lacks it).

```python
def log1p(x: np.ndarray) -> np.ndarray:
    """Elementwise log(1 + x) for x > -1; same shape and dtype as x."""
```

Requirements:

* full relative accuracy for tiny `|x|` (e.g. `log1p(1e-10) == 9.9999999995e-11`);
* `log1p(-0.0)` is `-0.0`, `log1p(inf)` is `inf`, NaN propagates;
* elementwise: works for any shape, including empty arrays.
