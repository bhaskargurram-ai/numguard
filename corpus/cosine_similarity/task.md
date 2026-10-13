# Task: cosine similarity

Implement `cosine_similarity(a, b)` in `solution.py`.

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.floating:
    """Cosine of the angle between two 1-D float vectors of equal length."""
```

Requirements:

* result in `[-1, 1]`, same dtype as the inputs;
* if either vector is all zeros (or empty) the similarity is defined as `0.0`
  (this is the convention of the retrieval system that consumes it) - never NaN;
* must not overflow or underflow for vectors with very large or very small
  components (embeddings stored in float16/float32 can have either);
* NaN in either input gives NaN.
