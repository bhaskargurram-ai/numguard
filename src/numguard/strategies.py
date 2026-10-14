"""Hypothesis strategies built from numguard specs (optional; needs ``hypothesis``).

numguard's edge-case catalogue is deterministic and finite; Hypothesis is
good at random search and shrinking. These strategies combine both: they draw
from the catalogue *and* from random arrays that respect each argument's
domain, dtype and shape, so property-based tests see NaN, infinities,
subnormals, cancellation-prone data and empty arrays as well as ordinary
values::

    from hypothesis import given
    import numguard.strategies as ngst

    @given(ngst.inputs_for(my_softmax))       # tuples of arrays, one per argument
    def test_softmax_sums_to_one(args):
        (x,) = args
        ...

    @given(ngst.cases(my_softmax, dtypes=["float32"]))   # numguard Case objects
    def test_with_case_tags(case):
        hypothesis.assume(case.finite_inputs)
        ...

The spec comes from a :class:`numguard.Spec`, a function decorated with
:func:`numguard.guard`, or the same keyword arguments as :func:`numguard.check`
(``inputs``, ``domain``, ``dtypes``, ...). Importing this module does not need
Hypothesis; calling a strategy function does (``pip install hypothesis``).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .api import Spec, get_spec, make_spec
from .cases import ArraySpec, Case, Domain
from .dtypes import BFLOAT16, dtype_name, finfo, round_bf16, storage_dtype

__all__ = ["cases", "edge_cases", "random_inputs", "inputs_for", "element_kwargs", "resolve_spec"]


def _hypothesis() -> tuple[Any, Any]:
    try:
        from hypothesis import strategies as st  # noqa: PLC0415 - optional dependency
        from hypothesis.extra import numpy as hnp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without hypothesis
        raise ImportError("numguard.strategies needs Hypothesis: pip install 'numguard[hypothesis]'") from exc
    return st, hnp


def resolve_spec(target: Callable[..., Any] | Spec | None = None, **options: Any) -> Spec:
    """A :class:`Spec` from a spec, a guarded function, or :func:`numguard.check` keywords."""
    if isinstance(target, Spec):
        return target.replace(**options) if options else target
    attached = get_spec(target) if target is not None else None
    if attached is not None and not ({"inputs", "domain"} & set(options)):
        overrides = dict(options)
        if "dtypes" in overrides and overrides["dtypes"] is not None:
            overrides["dtypes"] = tuple(overrides["dtypes"])
        return attached.replace(**overrides) if overrides else attached
    return make_spec(target, **options)


def cases(target: Callable[..., Any] | Spec | None = None, **options: Any) -> Any:
    """Strategy drawing :class:`~numguard.cases.Case` objects from the edge-case catalogue.

    Each drawn case's ``args`` are fresh copies, so the test may modify them.
    """
    st, _ = _hypothesis()
    spec = resolve_spec(target, **options)
    catalogue = spec.cases()
    if not catalogue:
        raise ValueError("the spec generates no cases")

    def fresh(c: Case) -> Case:
        return Case(c.name, c.kind, c.dtype, c.copy_args(), c.tags, c.size)

    return st.sampled_from(catalogue).map(fresh)


def edge_cases(target: Callable[..., Any] | Spec | None = None, **options: Any) -> Any:
    """Strategy drawing argument tuples from the edge-case catalogue."""
    return cases(target, **options).map(lambda c: c.args)


def element_kwargs(domain: Domain, dtype: str) -> dict[str, Any]:
    """Keyword arguments for ``hypothesis.strategies.floats`` / ``integers`` that
    stay inside ``domain`` for values stored in ``dtype``.

    Bounds are rounded *inwards* to the storage precision, so a domain like
    ``(0, 1]`` in float16 never yields 0. NaN is never drawn (the catalogue
    covers NaN handling); infinities only when the domain is unbounded.
    """
    name = dtype_name(dtype)
    st_dt = storage_dtype(name)
    if domain.integer or st_dt.kind in "iu":
        ii = np.iinfo(st_dt) if st_dt.kind in "iu" else np.iinfo(np.int64)
        lo = int(max(domain.lo, ii.min)) if math.isfinite(domain.lo) else int(ii.min)
        hi = int(min(domain.hi, ii.max)) if math.isfinite(domain.hi) else int(ii.max)
        if not domain.lo_closed:
            lo += 1
        if not domain.hi_closed:
            hi -= 1
        return {"kind": "int", "min_value": lo, "max_value": hi}
    width = {2: 16, 4: 32, 8: 64}[st_dt.itemsize]
    fmax = finfo(name).max

    def inward(v: float, up: bool) -> float:
        r = float(np.asarray(v, dtype=st_dt))
        if name == BFLOAT16:
            r = float(round_bf16(np.float32(r)))
        if (up and r < v) or (not up and r > v):
            from .dtypes import nextafter_in  # noqa: PLC0415

            r = nextafter_in(r, name, math.inf if up else -math.inf)
        return r

    kw: dict[str, Any] = {"kind": "float", "width": width, "allow_nan": False}
    lo = domain.lo if math.isfinite(domain.lo) else None
    hi = domain.hi if math.isfinite(domain.hi) else None
    if lo is not None:
        kw["min_value"] = inward(max(lo, -fmax), up=True)
        kw["exclude_min"] = not domain.lo_closed
    else:
        kw["min_value"] = -fmax
    if hi is not None:
        kw["max_value"] = inward(min(hi, fmax), up=False)
        kw["exclude_max"] = not domain.hi_closed
    else:
        kw["max_value"] = fmax
    kw["allow_infinity"] = False
    return kw


def _elements(st: Any, spec: ArraySpec, dtname: str) -> Any:
    kw = dict(element_kwargs(spec.domain, dtname))
    kind = kw.pop("kind")
    if kind == "int":
        return st.integers(**kw)
    floats = st.floats(**kw)
    if dtname == BFLOAT16:
        floats = floats.map(lambda v: float(round_bf16(np.float32(v))))
    return floats


def random_inputs(target: Callable[..., Any] | Spec | None = None, *, min_size: int = 0,
                  max_size: int = 32, **options: Any) -> Any:
    """Strategy drawing random in-domain argument tuples (sizes ``min_size..max_size``).

    Symbolic dimensions share one size per draw, as in the edge-case catalogue.
    """
    st, hnp = _hypothesis()
    spec = resolve_spec(target, **options)
    dtypes = list(spec.dtypes)

    def for_dtype(dtname: str) -> Any:
        def for_size(n: int) -> Any:
            parts = []
            for s in spec.inputs:
                arg_dt = dtype_name(s.dtype) if s.dtype is not None else dtname
                parts.append(hnp.arrays(storage_dtype(arg_dt), s.resolve_shape(n),
                                        elements=_elements(st, s, arg_dt)))
            return st.tuples(*parts)

        return st.integers(min_size, max_size).flatmap(for_size)

    return st.sampled_from(dtypes).flatmap(for_dtype)


def inputs_for(target: Callable[..., Any] | Spec | None = None, *, min_size: int = 0,
               max_size: int = 32, **options: Any) -> Any:
    """Edge cases from the catalogue *or* random in-domain arrays (argument tuples)."""
    st, _ = _hypothesis()
    return st.one_of(edge_cases(target, **options),
                     random_inputs(target, min_size=min_size, max_size=max_size, **options))
