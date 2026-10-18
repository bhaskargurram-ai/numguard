"""Public API: :class:`Spec`, :func:`check` and the :func:`guard` decorator."""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TypeVar, Union, overload

import numpy as np

from .cases import (
    DEFAULT_DTYPES,
    ArraySpec,
    Case,
    Domain,
    DomainLike,
    case_matches,
    generate_cases,
)
from .backends import BACKENDS
from .checks import run_checks
from .gradcheck import GRAD_BACKENDS
from .dtypes import BFLOAT16, dtype_name, round_bf16, storage_dtype, validate_dtype
from .properties import Property, PropertyLike, parse_property
from .report import Report

__all__ = ["Spec", "check", "check_spec", "guard", "get_spec", "InputsLike", "SPEC_ATTR"]

SPEC_ATTR = "__numguard__"

InputsLike = Union[
    None,
    ArraySpec,
    str,
    tuple,
    Sequence[Union[ArraySpec, str, tuple]],
    Mapping[str, Union[ArraySpec, str, tuple]],
]
F = TypeVar("F", bound=Callable[..., Any])

_NAN_POLICIES = ("propagate", "raise", "ignore")
_FP_POLICIES = ("ignore", "fail")


@dataclass(frozen=True)
class Spec:
    """Everything numguard needs to know to check a function.

    Most users build a spec implicitly through :func:`check` or :func:`guard`;
    see :func:`check` for the meaning of every field.
    """

    inputs: tuple[ArraySpec, ...] = ()
    reference: Callable[..., Any] | None = None
    properties: tuple[Property, ...] = ()
    dtypes: tuple[str, ...] = DEFAULT_DTYPES
    rtol: float | Mapping[str, float] | None = None
    atol: float | Mapping[str, float] | None = None
    finite: bool = True
    nan: str = "propagate"
    allow_nonfinite: tuple[str, ...] = ()
    may_raise: tuple[str, ...] = ()
    out_dtype: Any = "preserve"
    batched: bool = False
    batch_axis: int = 0
    monotonic: str | None = None
    monotonic_arg: int = 0
    bounds: tuple[float, float] | None = None
    fp_signals: str = "ignore"
    check_mutation: bool = True
    postconditions: tuple[Callable[..., Any], ...] = ()
    ignore: tuple[str, ...] = ()
    ref_upcast: bool = True
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    ref_kwargs: Mapping[str, Any] = field(default_factory=dict)
    n: int = 8
    large_n: int = 100_000
    seed: int = 0
    extra_cases: Mapping[str, Any] = field(default_factory=dict)
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    arg_names: tuple[str, ...] = ()
    backend: str = "numpy"
    elementwise: bool = False
    grad: Any = False
    grad_reference: Callable[..., Any] | None = None
    grad_rtol: float | None = None

    def __post_init__(self) -> None:
        if self.nan not in _NAN_POLICIES:
            raise ValueError(f"nan must be one of {_NAN_POLICIES}, got {self.nan!r}")
        if self.fp_signals not in _FP_POLICIES:
            raise ValueError(f"fp_signals must be one of {_FP_POLICIES}, got {self.fp_signals!r}")
        if self.monotonic not in (None, "increasing", "decreasing"):
            raise ValueError("monotonic must be None, 'increasing' or 'decreasing'")
        object.__setattr__(self, "dtypes", tuple(validate_dtype(d) for d in self.dtypes))
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.grad is True and self.backend not in GRAD_BACKENDS:
            raise ValueError("grad=True uses automatic differentiation, which needs backend='jax' "
                             "or backend='torch'; pass grad=<function> to check an analytic gradient")
        if self.grad not in (None, False, True) and not callable(self.grad):
            raise TypeError("grad must be True, False or a function returning the gradient")
        if self.grad_reference is not None and self.grad in (None, False):
            raise ValueError("grad_reference needs grad=True or grad=<function>")

    def replace(self, **changes: Any) -> "Spec":
        """Return a copy with some fields changed (``None`` values are ignored)."""
        changes = {k: v for k, v in changes.items() if v is not None}
        if "dtypes" in changes:
            changes["dtypes"] = tuple(changes["dtypes"])
        return dataclasses.replace(self, **changes)

    def cases(self) -> list[Case]:
        """Generate the edge cases this spec describes (plus ``extra_cases``)."""
        cases = generate_cases(
            self.inputs, self.dtypes, n=self.n, large_n=self.large_n, seed=self.seed,
            include=self.include, exclude=self.exclude, arg_names=self.arg_names or None,
        )
        return cases + self._extra_cases()

    def _extra_cases(self) -> list[Case]:
        out: list[Case] = []
        sweep = self.dtypes if any(s.dtype is None for s in self.inputs) else (
            dtype_name(self.inputs[0].dtype),)
        for name, values in self.extra_cases.items():
            if len(self.inputs) == 1:
                values = (values,)
            if not isinstance(values, (tuple, list)) or len(values) != len(self.inputs):
                raise ValueError(f"extra case {name!r} needs {len(self.inputs)} array(s)")
            for dt in sweep:
                arrays = []
                for spec_i, v in zip(self.inputs, values):
                    tname = dtype_name(spec_i.dtype) if spec_i.dtype is not None else dt
                    with np.errstate(all="ignore"):
                        arr = np.asarray(v, dtype=np.float64).astype(storage_dtype(tname))
                        arrays.append(round_bf16(arr) if tname == BFLOAT16 else arr)
                finite = all(np.all(np.isfinite(a)) for a in arrays if a.dtype.kind == "f")
                has_nan = any(np.isnan(a).any() for a in arrays if a.dtype.kind == "f")
                tags = {"extra", name, dt, "finite_input" if finite else "nonfinite_input"}
                if has_nan:
                    tags.add("nan_input")
                size = max((a.shape[-1] for a in arrays if a.ndim), default=0)
                case = Case(name=name, kind="extra", dtype=dt, args=tuple(arrays),
                            tags=frozenset(tags), size=size)
                if self.exclude and case_matches(case, self.exclude):
                    continue
                if self.include is not None and not case_matches(case, self.include):
                    continue
                out.append(case)
        return out


# --------------------------------------------------------------------------
# Building specs
# --------------------------------------------------------------------------


def _positional_params(fn: Callable[..., Any]) -> list[str]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ["x"]
    names = []
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            names.append(p.name)
    return names or ["x"]


def _as_arrayspec(item: Any, name: str | None, domain: DomainLike | None) -> ArraySpec:
    if isinstance(item, ArraySpec):
        if name and item.name is None:
            return dataclasses.replace(item, name=name)
        return item
    if isinstance(item, (str, Domain)):
        return ArraySpec(domain=Domain.parse(item), name=name)
    if isinstance(item, tuple):
        return ArraySpec(shape=item, domain=Domain.parse(domain or "any"), name=name)
    raise TypeError(f"cannot interpret {item!r} as an input spec")


def _build_inputs(fn: Callable[..., Any] | None, inputs: InputsLike,
                  domain: DomainLike | None) -> tuple[tuple[ArraySpec, ...], tuple[str, ...]]:
    names = _positional_params(fn) if fn is not None else []
    if inputs is None:
        if fn is None:
            raise ValueError("inputs must be given when no function is available")
        dom = Domain.parse(domain or "any")
        specs = tuple(ArraySpec(domain=dom, name=n) for n in names)
        return specs, tuple(names)
    if isinstance(inputs, Mapping):
        specs = tuple(_as_arrayspec(v, k, domain) for k, v in inputs.items())
        return specs, tuple(inputs.keys())
    if isinstance(inputs, (ArraySpec, str, Domain)) or (
        isinstance(inputs, tuple) and all(isinstance(d, (int, str)) for d in inputs)
        and not all(isinstance(d, str) and _looks_like_domain(d) for d in inputs)
    ):
        inputs = [inputs]
    specs_list = []
    for i, item in enumerate(inputs):  # type: ignore[arg-type]
        nm = names[i] if i < len(names) else None
        specs_list.append(_as_arrayspec(item, nm, domain))
    specs = tuple(specs_list)
    arg_names = tuple(s.name or f"arg{i}" for i, s in enumerate(specs))
    return specs, arg_names


def _looks_like_domain(s: str) -> bool:
    try:
        Domain.parse(s)
        return True
    except ValueError:
        return False


def _tuple(v: Any) -> tuple:
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(v)


def make_spec(fn: Callable[..., Any] | None = None, *, inputs: InputsLike = None,
              domain: DomainLike | None = None, reference: Callable[..., Any] | None = None,
              properties: Sequence[PropertyLike] | PropertyLike = (),
              dtypes: Sequence[str] | str | None = None, **options: Any) -> Spec:
    """Build a :class:`Spec` from user-friendly arguments (see :func:`check`)."""
    specs, names = _build_inputs(fn, inputs, domain)
    props = tuple(parse_property(p) for p in _tuple(properties))
    if "postconditions" in options:
        pc = options["postconditions"]
        options["postconditions"] = (pc,) if callable(pc) else tuple(pc or ())
    for key in ("allow_nonfinite", "may_raise", "exclude", "ignore"):
        if key in options:
            options[key] = _tuple(options[key])
    if options.get("include") is not None:
        options["include"] = _tuple(options["include"])
    if options.get("bounds") is not None:
        lo, hi = options["bounds"]
        options["bounds"] = (float(lo), float(hi))
    unknown = set(options) - {f.name for f in dataclasses.fields(Spec)}
    if unknown:
        raise TypeError(f"unknown numguard option(s): {', '.join(sorted(unknown))}")
    return Spec(inputs=specs, reference=reference, properties=props,
                dtypes=tuple(_tuple(dtypes)) or DEFAULT_DTYPES, arg_names=names, **options)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def check(
    fn: Callable[..., Any],
    inputs: InputsLike = None,
    reference: Callable[..., Any] | None = None,
    properties: Sequence[PropertyLike] | PropertyLike = (),
    dtypes: Sequence[str] | str | None = None,
    rtol: float | Mapping[str, float] | None = None,
    atol: float | Mapping[str, float] | None = None,
    *,
    domain: DomainLike | None = None,
    spec: Spec | None = None,
    name: str | None = None,
    **options: Any,
) -> Report:
    """Run numguard's edge cases and checks against ``fn``.

    Parameters
    ----------
    fn:
        The function under test. It is called positionally with one NumPy
        array per input spec (plus ``kwargs``).
    inputs:
        Input specs. ``None`` infers one 1-D array per required positional
        parameter, using ``domain``. Accepts an :class:`ArraySpec`
        (e.g. ``Array("n", domain="positive")``), a domain string, a shape
        tuple, a list of those, or a ``{name: spec}`` mapping.
    reference:
        Trusted implementation. Called with float64-upcast inputs, or with
        the original inputs if it is a :class:`~numguard.reference.HighPrecReference`
        (see :func:`numguard.highprec`).
    properties:
        Metamorphic properties such as ``"translation_invariant"``,
        ``"scale_equivariant:2"`` or :class:`~numguard.properties.Property` objects.
    dtypes:
        Floating dtypes to test (default ``float32`` and ``float64``).
    rtol, atol:
        Tolerances for value comparisons; floats or ``{dtype_name: value}``
        keyed by the compared output's dtype (``{"uint8": 1}`` allows
        off-by-one integer outputs). Defaults depend on the dtype (1e-2, 1e-5,
        1e-12 for float16/32/64; integer outputs are compared exactly).
    domain:
        Domain for inferred inputs: ``any``, ``positive``, ``nonneg``,
        ``unit_interval``, ``"[a, b)"``, ``"int[0, 255]"``...
    spec:
        A prebuilt :class:`Spec` (other arguments then override its fields).

    Notes
    -----
    Other options (keyword-only) map to :class:`Spec` fields:
    ``finite`` (bool), ``nan`` (``"propagate"``/``"raise"``/``"ignore"``),
    ``allow_nonfinite`` and ``may_raise`` (case names/kinds/tags/globs),
    ``out_dtype`` (``"preserve"``, ``None``, a dtype or a per-output list),
    ``batched``/``batch_axis``, ``monotonic`` (``"increasing"``/``"decreasing"``),
    ``bounds`` (``(lo, hi)``), ``fp_signals`` (``"ignore"``/``"fail"``),
    ``postconditions`` (callables ``post(args, outputs)`` returning ``None``/
    ``True`` when satisfied, or ``False``/a message), ``ignore`` (entries
    ``"check:pattern"`` that downgrade matching results to *skip*, e.g.
    ``"reference:inf_input"``),
    ``extra_cases`` (``{name: arrays}`` of hand-written inputs, cast to each
    dtype under test), ``check_mutation``, ``kwargs``/``ref_kwargs``, ``ref_upcast`` (call a
    plain reference with float64-upcast inputs, default True), ``n``, ``large_n``,
    ``seed``, ``include``/``exclude``, ``backend`` (``"numpy"`` (default),
    ``"torch"``, ``"jax"`` or ``"cupy"``: inputs are converted to that
    library's arrays, outputs back to NumPy; see :mod:`numguard.backends`),
    ``elementwise`` (bool: each output element depends only on the same input
    element; enables the ``isolation`` check and faster finite differences),
    ``grad`` (``True`` for JAX/PyTorch automatic differentiation, or a function
    returning the gradient of ``sum(f)`` per floating argument), ``grad_reference``
    (a float64 function returning the exact gradient; default: finite differences
    of ``reference`` or of ``fn`` in float64) and ``grad_rtol``.
    ``dtypes`` may include ``"bfloat16"`` (emulated; see :mod:`numguard.dtypes`).
    """
    if spec is None:
        spec = make_spec(fn, inputs=inputs, domain=domain, reference=reference,
                         properties=properties, dtypes=dtypes, rtol=rtol, atol=atol, **options)
    else:
        overrides: dict[str, Any] = dict(options)
        if inputs is not None or domain is not None:
            overrides["inputs"], overrides["arg_names"] = _build_inputs(fn, inputs, domain)
        if properties:
            overrides["properties"] = tuple(parse_property(p) for p in _tuple(properties))
        spec = spec.replace(reference=reference, dtypes=_tuple(dtypes) or None, rtol=rtol,
                            atol=atol, **overrides)
    return check_spec(fn, spec, name=name)


def check_spec(fn: Callable[..., Any], spec: Spec, *, name: str | None = None,
               **overrides: Any) -> Report:
    """Check ``fn`` against an existing :class:`Spec` (with optional overrides)."""
    if overrides:
        spec = spec.replace(**overrides)
    return run_checks(fn, spec, spec.cases(), name=name)


def get_spec(fn: Callable[..., Any]) -> Spec | None:
    """Return the :class:`Spec` attached by :func:`guard`, or ``None``."""
    return getattr(fn, SPEC_ATTR, None)


@overload
def guard(fn: F) -> F: ...


@overload
def guard(fn: None = None, **kwargs: Any) -> Callable[[F], F]: ...


def guard(fn: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Attach a numguard spec to a function without changing its behaviour.

    Usable bare (``@guard``) or with the same arguments as :func:`check`
    (``@guard(domain="positive", reference=np.log)``). The spec is stored as
    ``fn.__numguard__``; ``fn.numguard_check(**overrides)`` runs it and returns
    a :class:`Report`. The pytest plugin collects guarded functions from the
    modules passed with ``--numguard-modules``.
    """

    def decorate(f: F) -> F:
        spec = make_spec(f, **kwargs)
        setattr(f, SPEC_ATTR, spec)

        def numguard_check(**overrides: Any) -> Report:
            return check_spec(f, spec, **overrides)

        f.numguard_check = numguard_check  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorate(fn)
    return decorate
