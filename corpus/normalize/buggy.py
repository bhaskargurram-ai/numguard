"""x / ||x||."""

import numpy as np


def normalize(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
