"""Direct translation of the formula."""

import numpy as np


def logsumexp(x):
    return np.log(np.sum(np.exp(x)))
