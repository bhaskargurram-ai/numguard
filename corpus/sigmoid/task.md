# Task: logistic sigmoid

Implement `sigmoid(x)` in `solution.py` (do not use `scipy.special.expit`).

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Elementwise 1 / (1 + exp(-x)); same shape and dtype as x."""
```

Requirements:

* values in `[0, 1]`, monotonically non-decreasing;
* no floating-point overflow warnings or errors for any finite input
  (training loops run with `np.seterr(all="raise")`);
* accurate for very negative inputs where the result is tiny
  (`sigmoid(-50.0) == 1.9287498479639178e-22`);
* NaN propagates.
