"""Reference evaluation helpers: ULP distances and high-precision references.

The reference check compares a function against a trusted callable. To make
that callable meaningfully *more* accurate than the code under test, numguard
can evaluate it in extended precision:

* :func:`longdouble` runs a NumPy-based callable on ``np.longdouble`` inputs
  (80-bit extended precision on most x86-64 Linux builds; on platforms where
  ``longdouble`` is just ``float64`` this degrades gracefully to float64).
* :func:`mp_scalar` runs a scalar function written with :mod:`mpmath`
  elementwise at a chosen number of decimal digits, if mpmath is installed.
* :func:`highprec` picks the best available of the two.
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any, Callable

import numpy as np

__all__ = [
    "HAS_MPMATH",
    "LONGDOUBLE_IS_EXTENDED",
    "HighPrecReference",
    "longdouble",
    "mp_scalar",
    "highprec",
    "ulp_distance",
    "ulp",
    "default_tolerance",
    "DEFAULT_RTOL",
]

HAS_MPMATH: bool = importlib.util.find_spec("mpmath") is not None
LONGDOUBLE_IS_EXTENDED: bool = bool(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant)

#: Default relative tolerances per floating dtype (roughly 10, 80 and 4500 ULP).
DEFAULT_RTOL: dict[str, float] = {
    "float16": 1e-2,
    "float32": 1e-5,
    "float64": 1e-12,
    "longdouble": 1e-15,
}


def default_tolerance(dtype: Any) -> tuple[float, float]:
    """Return default ``(rtol, tiny)`` for comparing values of ``dtype``.

    ``tiny`` is the smallest normal number of the dtype; relative errors of
    subnormal values are measured against it. Integer and boolean dtypes are
    compared exactly.
    """
    dt = np.dtype(dtype)
    if dt.kind == "f":
        name = "longdouble" if dt == np.longdouble and dt != np.float64 else dt.name
        rtol = DEFAULT_RTOL.get(name, 1e-12)
        return rtol, float(np.finfo(dt).tiny)
    if dt.kind == "c":
        return default_tolerance(np.finfo(dt).dtype)
    return 0.0, 0.0


# --------------------------------------------------------------------------
# ULPs
# --------------------------------------------------------------------------

_UINT_FOR_SIZE = {2: np.uint16, 4: np.uint32, 8: np.uint64}


def ulp(x: Any, dtype: Any = None) -> np.ndarray:
    """Size of one unit in the last place at ``x`` (``np.spacing`` magnitude)."""
    arr = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    return np.abs(np.spacing(arr))


def ulp_distance(a: Any, b: Any, dtype: Any = None) -> np.ndarray:
    """Number of representable values of ``dtype`` between ``a`` and ``b``.

    Both inputs are rounded to ``dtype`` (default: the result type of ``a``
    and ``b``, which must be float16, float32 or float64). The result is a
    float64 array: ``0`` for identical values (``+0`` and ``-0`` are 0 apart,
    two NaNs are 0 apart), ``inf`` if exactly one of the pair is NaN.
    Infinity is treated as one ULP beyond the largest finite value.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    dt = np.dtype(dtype) if dtype is not None else np.result_type(a, b)
    if dt.kind != "f" or dt.itemsize not in _UINT_FOR_SIZE:
        raise TypeError(f"ulp_distance supports float16/32/64, not {dt}")
    with np.errstate(all="ignore"):
        a = a.astype(dt)
        b = b.astype(dt)
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    a = np.array(a, copy=True).reshape(-1)
    b = np.array(b, copy=True).reshape(-1)
    ut = _UINT_FOR_SIZE[dt.itemsize]
    bits = dt.itemsize * 8
    sign_mask = ut(1 << (bits - 1))
    mag_mask = ut((1 << (bits - 1)) - 1)
    ia = a.view(ut)
    ib = b.view(ut)
    ma, mb = ia & mag_mask, ib & mag_mask
    sa, sb = (ia & sign_mask) != 0, (ib & sign_mask) != 0
    same = sa == sb
    hi = np.maximum(ma, mb)
    lo = np.minimum(ma, mb)
    dist = np.where(same, hi - lo, ma + mb).astype(np.float64)  # no overflow: ma, mb < 2**(bits-1)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    dist = np.where(nan_a & nan_b, 0.0, dist)
    dist = np.where(nan_a ^ nan_b, np.inf, dist)
    return dist.reshape(shape)


# --------------------------------------------------------------------------
# High-precision references
# --------------------------------------------------------------------------


class HighPrecReference:
    """A reference callable that computes in higher precision than its inputs.

    The check machinery passes the *original* case inputs (not upcast to
    float64) and compares the returned values against the function under
    test. ``precision`` is a human-readable label shown in reports.
    """

    def __init__(self, fn: Callable[..., Any], precision: str, name: str | None = None):
        self._fn = fn
        self.precision = precision
        functools.update_wrapper(self, fn, updated=())
        self.__name__ = self.__qualname__ = name or getattr(fn, "__name__", "reference")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<HighPrecReference {self.__name__} ({self.precision})>"


def _to_longdouble(arg: Any) -> Any:
    if isinstance(arg, np.ndarray) and arg.dtype.kind == "f":
        return arg.astype(np.longdouble)
    if isinstance(arg, (float, np.floating)):
        return np.longdouble(arg)
    return arg


def longdouble(fn: Callable[..., Any]) -> HighPrecReference:
    """Wrap a NumPy-based callable so that it runs on ``np.longdouble`` inputs.

    Floating-point array arguments are converted to ``np.longdouble`` before
    the call; the result is returned unchanged (typically longdouble).
    """

    def run(*args: Any, **kwargs: Any) -> Any:
        return fn(*(_to_longdouble(a) for a in args), **kwargs)

    label = "longdouble" if LONGDOUBLE_IS_EXTENDED else "longdouble(=float64)"
    return HighPrecReference(run, label, getattr(fn, "__name__", None))


def mp_scalar(fn: Callable[..., Any], dps: int = 50) -> HighPrecReference:
    """Vectorize a scalar :mod:`mpmath` function into an array reference.

    ``fn`` receives ``mpmath.mpf`` scalars and should return an mpf (or a
    number). Evaluation happens at ``dps`` decimal digits; results are
    converted to ``np.longdouble``. Raises :class:`ImportError` if mpmath is
    not installed - use :func:`highprec` for automatic fallback.
    """
    if not HAS_MPMATH:
        raise ImportError("mp_scalar requires mpmath (pip install mpmath)")
    import mpmath  # noqa: PLC0415 - optional dependency

    def scalar(*xs: Any) -> Any:
        with mpmath.workdps(dps):
            vals = []
            for x in xs:
                xf = float(x)
                vals.append(mpmath.mpf(xf) if xf == xf else mpmath.nan)
            r = fn(*vals)
            if isinstance(r, mpmath.mpc):
                # a complex result means the input is outside the real domain
                return np.longdouble("nan") if r.imag != 0 else np.longdouble(mpmath.nstr(r.real, 25))
            if not mpmath.isfinite(r):
                return np.longdouble(float(r))
            # keep more than float64 precision when longdouble is extended
            return np.longdouble(mpmath.nstr(r, 25))

    vec = np.vectorize(scalar, otypes=[np.longdouble])

    def run(*args: Any) -> Any:
        return vec(*args)

    return HighPrecReference(run, f"mpmath({dps} digits)", getattr(fn, "__name__", None))


def highprec(
    numpy_fn: Callable[..., Any] | None = None,
    mp_fn: Callable[..., Any] | None = None,
    *,
    dps: int = 50,
    prefer: str = "mpmath",
) -> HighPrecReference:
    """Best-available high-precision reference.

    ``mp_fn`` is an elementwise scalar function written with mpmath; it is
    used when mpmath is importable and ``prefer == "mpmath"``. Otherwise
    ``numpy_fn`` is evaluated in ``np.longdouble``. At least one is required.

    Example::

        ref = highprec(numpy_fn=np.log1p, mp_fn=lambda x: mpmath.log1p(x))
    """
    if numpy_fn is None and mp_fn is None:
        raise ValueError("highprec needs numpy_fn and/or mp_fn")
    if mp_fn is not None and HAS_MPMATH and (prefer == "mpmath" or numpy_fn is None):
        return mp_scalar(mp_fn, dps=dps)
    if numpy_fn is None:
        raise ImportError(
            "mpmath is not installed and no numpy_fn fallback was given; "
            "pip install 'numguard[mpmath]'"
        )
    return longdouble(numpy_fn)
