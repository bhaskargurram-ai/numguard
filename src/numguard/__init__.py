"""numguard - numerical edge-case testing for array code.

Generate the inputs that numerical code gets wrong (zeros, empty arrays, NaN,
infinities, huge and subnormal magnitudes, signed zero, cancellation-prone
data, float16 precision, ...) and check a function against them::

    import numpy as np
    import numguard

    def logsumexp(x):
        return np.log(np.sum(np.exp(x)))

    report = numguard.check(logsumexp, reference=scipy.special.logsumexp,
                            properties=["translation_equivariant"])
    print(report.summary())
"""

from __future__ import annotations

from . import properties
from .api import Spec, check, check_spec, get_spec, guard
from .cases import (
    ANY,
    DEFAULT_DTYPES,
    FLOAT_DTYPES,
    KINDS,
    Array,
    ArraySpec,
    Case,
    Domain,
    generate_cases,
    integers,
    interval,
)
from .checks import compare
from .properties import Property
from .reference import highprec, longdouble, mp_scalar, ulp_distance
from .report import CheckResult, NumGuardError, Report

__version__ = "0.1.0"

__all__ = [
    "ANY",
    "Array",
    "ArraySpec",
    "Case",
    "CheckResult",
    "DEFAULT_DTYPES",
    "Domain",
    "FLOAT_DTYPES",
    "KINDS",
    "NumGuardError",
    "Property",
    "Report",
    "Spec",
    "__version__",
    "check",
    "check_spec",
    "compare",
    "generate_cases",
    "get_spec",
    "guard",
    "highprec",
    "integers",
    "interval",
    "longdouble",
    "mp_scalar",
    "properties",
    "ulp_distance",
]
