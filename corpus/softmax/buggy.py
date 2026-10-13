"""Textbook softmax."""

import numpy as np


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
