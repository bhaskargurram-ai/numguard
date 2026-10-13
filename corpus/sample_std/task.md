# Task: sample standard deviation

Implement `sample_std(x)` in `solution.py`.

```python
def sample_std(x: np.ndarray) -> np.floating:
    """Sample standard deviation (ddof=1, Bessel's correction) of a 1-D float array."""
```

Requirements:

* same dtype as `x`;
* the sample standard deviation is undefined for fewer than two observations:
  for `len(x) < 2` raise `ValueError` (do not return 0, inf or a warning-laden NaN);
* accurate for huge and tiny magnitudes and for data with a large common offset;
* NaN in the input gives NaN.
