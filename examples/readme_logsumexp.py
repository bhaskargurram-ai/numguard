import numpy as np
import scipy.special
import numguard

def logsumexp(x):                       # what an assistant often writes first
    return np.log(np.sum(np.exp(x)))

assert np.isclose(logsumexp(np.array([1.0, 2.0, 3.0])), 3.40760596)  # happy path: passes

report = numguard.check(logsumexp, reference=scipy.special.logsumexp,
                        properties=["translation_equivariant"])
print(report.summary())
