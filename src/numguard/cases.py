"""Edge-case generation.

Given a description of a function's array inputs (shape, dtype, value domain),
:func:`generate_cases` produces a deterministic list of named :class:`Case`
objects covering the inputs that numerical code most often gets wrong:
zeros, empty and single-element arrays, NaN and infinities, values near the
overflow and underflow thresholds, subnormals, signed zero, catastrophic
cancellation, large ``n`` and more.

Every case carries *tags* so that checks know what to expect (for example a
case tagged ``nonfinite_input`` is not required to produce finite output).
"""

from __future__ import annotations

import fnmatch
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Union

import numpy as np

from .dtypes import BFLOAT16, FLOAT_DTYPES, dtype_name, finfo, nextafter_in, round_bf16, storage_dtype

__all__ = [
    "ANY",
    "FLOAT_DTYPES",
    "DEFAULT_DTYPES",
    "KINDS",
    "REGULAR_KINDS",
    "ArraySpec",
    "Array",
    "Case",
    "Domain",
    "integers",
    "interval",
    "generate_cases",
    "case_matches",
]

DEFAULT_DTYPES: tuple[str, ...] = ("float32", "float64")

#: Case kinds that produce "ordinary" finite inputs of moderate magnitude.
#: Property checks (invariances, batching, ...) run on these.
REGULAR_KINDS: frozenset[str] = frozenset(
    {
        "random",
        "zeros",
        "ones",
        "constant",
        "sorted",
        "reverse_sorted",
        "duplicates",
        "alternating",
        "single",
        "pair",
    }
)
_EXTREME_KINDS = frozenset(
    {"huge", "large", "tiny", "subnormal", "small", "wide_range", "cancellation"}
)
_SIZE_KINDS = frozenset({"empty", "single", "pair", "large_n"})
_NONFINITE_KINDS = frozenset({"nan", "posinf", "neginf"})


# --------------------------------------------------------------------------
# Domains
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """A set of allowed input values: an interval, optionally integer-valued.

    NaN is never part of a domain, but NaN cases are still generated (they
    test NaN handling, which is governed by the ``nan=`` option of a check).
    """

    lo: float = -math.inf
    hi: float = math.inf
    lo_closed: bool = True
    hi_closed: bool = True
    integer: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.lo <= self.hi:
            raise ValueError(f"empty domain: lo={self.lo} > hi={self.hi}")

    def contains(self, value: float) -> bool:
        """Return True if ``value`` lies in the domain."""
        v = float(value)
        if math.isnan(v):
            return False
        if self.integer and math.isfinite(v) and v != math.floor(v):
            return False
        lo_ok = v >= self.lo if self.lo_closed else v > self.lo
        hi_ok = v <= self.hi if self.hi_closed else v < self.hi
        if math.isinf(v):
            # infinities are only "in" a domain whose bound is infinite
            return (v > 0 and self.hi == math.inf) or (v < 0 and self.lo == -math.inf)
        return lo_ok and hi_ok

    def contains_all(self, values: Iterable[float]) -> bool:
        return all(self.contains(v) for v in values)

    @property
    def symmetric(self) -> bool:
        """True if ``x in D`` implies ``-x in D``."""
        return self.lo == -self.hi and self.lo_closed == self.hi_closed

    @property
    def allows_negative(self) -> bool:
        return self.lo < 0

    @property
    def allows_positive(self) -> bool:
        return self.hi > 0

    def describe(self) -> str:
        if self.name:
            return self.name
        lb = "[" if self.lo_closed and math.isfinite(self.lo) else "("
        rb = "]" if self.hi_closed and math.isfinite(self.hi) else ")"
        prefix = "int" if self.integer else ""
        return f"{prefix}{lb}{_fmt_bound(self.lo)}, {_fmt_bound(self.hi)}{rb}"

    @classmethod
    def parse(cls, spec: "DomainLike") -> "Domain":
        """Parse a domain from a name or interval notation.

        Accepted forms: a :class:`Domain`; names ``any``/``real``, ``positive``,
        ``nonneg``, ``negative``, ``nonpos``, ``unit_interval``; interval
        notation such as ``"[-1, inf)"`` or ``"(0, 1]"``; integer ranges such
        as ``"int[0, 255]"``.
        """
        if isinstance(spec, Domain):
            return spec
        if not isinstance(spec, str):
            raise TypeError(f"cannot interpret {spec!r} as a domain")
        key = spec.strip().lower().replace("-", "_")
        if key in _NAMED_DOMAINS:
            return _NAMED_DOMAINS[key]
        m = re.fullmatch(
            r"(int)?\s*([\[(])\s*([^,\s]+)\s*,\s*([^,\s]+)\s*([\])])", spec.strip()
        )
        if not m:
            raise ValueError(
                f"unknown domain {spec!r}; use one of {sorted(_NAMED_DOMAINS)} "
                "or interval notation like '[-1, inf)' or 'int[0, 255]'"
            )
        is_int, lb, lo, hi, rb = m.groups()
        lo_v, hi_v = _parse_bound(lo), _parse_bound(hi)
        if is_int:
            return integers(int(lo_v), int(hi_v))
        return Domain(lo_v, hi_v, lb == "[", rb == "]")


DomainLike = Union[Domain, str]


def _fmt_bound(v: float) -> str:
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:g}"


def _parse_bound(s: str) -> float:
    s = s.strip().lower()
    if s in ("inf", "+inf", "infinity"):
        return math.inf
    if s in ("-inf", "-infinity"):
        return -math.inf
    return float(s)


ANY = Domain(name="any")
_NAMED_DOMAINS: dict[str, Domain] = {
    "any": ANY,
    "real": ANY,
    "positive": Domain(0.0, math.inf, lo_closed=False, name="positive"),
    "nonneg": Domain(0.0, math.inf, name="nonneg"),
    "nonnegative": Domain(0.0, math.inf, name="nonneg"),
    "negative": Domain(-math.inf, 0.0, hi_closed=False, name="negative"),
    "nonpos": Domain(-math.inf, 0.0, name="nonpos"),
    "unit_interval": Domain(0.0, 1.0, name="unit_interval"),
}


def interval(lo: float, hi: float, *, closed: str = "both") -> Domain:
    """Real interval domain. ``closed`` is one of both/left/right/neither."""
    if closed not in ("both", "left", "right", "neither"):
        raise ValueError("closed must be 'both', 'left', 'right' or 'neither'")
    return Domain(
        float(lo),
        float(hi),
        lo_closed=closed in ("both", "left"),
        hi_closed=closed in ("both", "right"),
    )


def integers(lo: int, hi: int) -> Domain:
    """Closed integer range ``[lo, hi]`` (inputs use an integer dtype)."""
    return Domain(float(lo), float(hi), integer=True, name=f"int[{lo}, {hi}]")


# --------------------------------------------------------------------------
# Input specs and cases
# --------------------------------------------------------------------------

Dim = Union[int, str]


@dataclass(frozen=True)
class ArraySpec:
    """Description of one array argument.

    ``shape`` entries are ints (fixed) or strings (symbolic sizes, all of which
    take the case's size ``n``). ``dtype=None`` means "the dtype under test";
    a fixed dtype pins the argument (integer domains default to ``int64``).
    """

    shape: tuple[Dim, ...] = ("n",)
    domain: Domain = ANY
    dtype: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", Domain.parse(self.domain))
        if self.domain.integer and self.dtype is None:
            object.__setattr__(self, "dtype", "int64")
        for d in self.shape:
            if not isinstance(d, (int, str)) or isinstance(d, bool):
                raise TypeError(f"shape entries must be int or str, got {d!r}")

    @property
    def symbolic(self) -> bool:
        return any(isinstance(d, str) for d in self.shape)

    def resolve_shape(self, size: int) -> tuple[int, ...]:
        return tuple(size if isinstance(d, str) else int(d) for d in self.shape)

    @property
    def is_integer(self) -> bool:
        return self.dtype is not None and storage_dtype(self.dtype).kind in "iub"


def Array(*shape: Dim, domain: DomainLike = "any", dtype: str | None = None,
          name: str | None = None) -> ArraySpec:
    """Convenience constructor: ``Array("n", domain="positive")``.

    With no shape the argument is a 1-D array of symbolic length ``n``.
    For 0-d (scalar) inputs use ``ArraySpec(shape=())``.
    """
    return ArraySpec(shape=tuple(shape) if shape else ("n",),
                     domain=Domain.parse(domain), dtype=dtype, name=name)


@dataclass
class Case:
    """One generated input set for the function under test."""

    name: str
    kind: str
    dtype: str
    args: tuple[np.ndarray, ...]
    tags: frozenset[str] = field(default_factory=frozenset)
    size: int = 0

    @property
    def finite_inputs(self) -> bool:
        return "nonfinite_input" not in self.tags

    @property
    def regular(self) -> bool:
        return "regular" in self.tags

    def copy_args(self) -> tuple[np.ndarray, ...]:
        return tuple(np.array(a, copy=True) for a in self.args)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        shapes = ", ".join(str(a.shape) for a in self.args)
        return f"Case({self.name!r}, {self.dtype}, shapes=[{shapes}])"


def case_matches(case: Case, patterns: Iterable[str]) -> bool:
    """True if any pattern matches the case name (glob), its kind, or a tag."""
    for p in patterns:
        if p == case.kind or p in case.tags or fnmatch.fnmatchcase(case.name, p):
            return True
    return False


# --------------------------------------------------------------------------
# Value generators
# --------------------------------------------------------------------------


@dataclass
class _Ctx:
    domain: Domain
    dtype: np.dtype
    rng: np.random.Generator
    count: int
    baseline: np.ndarray
    dtname: str = ""

    @property
    def fi(self):  # noqa: ANN201 - numguard.dtypes.FInfo
        return finfo(self.dtname or self.dtype)


def _signs(ctx: _Ctx, count: int) -> np.ndarray | None:
    """Random signs consistent with the domain (None if no sign is allowed)."""
    d = ctx.domain
    if d.allows_negative and d.allows_positive:
        return ctx.rng.choice(np.array([-1.0, 1.0]), size=count)
    if d.allows_positive:
        return np.ones(count)
    if d.allows_negative:
        return -np.ones(count)
    return None


def _magnitude_case(ctx: _Ctx, mags: np.ndarray) -> np.ndarray | None:
    s = _signs(ctx, mags.size)
    if s is None:
        return None
    vals = s * mags
    # flip the sign of elements that fall outside the domain (e.g. -1e308 for
    # a domain like (-1, inf)); give up if neither sign fits
    ok = np.array([ctx.domain.contains(v) for v in vals.tolist()], dtype=bool)
    vals = np.where(ok, vals, -vals)
    if not ctx.domain.contains_all(vals.tolist()):
        return None
    return vals


def _baseline(domain: Domain, dtype: np.dtype, rng: np.random.Generator,
              count: int, dtname: str = "") -> np.ndarray:
    """Typical in-domain values (float64 array, later cast)."""
    if domain.integer:
        lo = max(domain.lo, -1000.0)
        hi = min(domain.hi, 1000.0)
        return rng.integers(int(lo), int(hi), size=count, endpoint=True).astype(np.float64)
    lo, hi = domain.lo, domain.hi
    if math.isinf(lo) and math.isinf(hi):
        vals = rng.standard_normal(count)
    elif math.isinf(hi):
        vals = lo + rng.lognormal(0.0, 0.5, count)
    elif math.isinf(lo):
        vals = hi - rng.lognormal(0.0, 0.5, count)
    else:
        vals = rng.uniform(lo, hi, count)
        if not domain.lo_closed:
            vals = np.where(vals == lo, (lo + hi) / 2, vals)
    if dtype.kind == "f":
        fmax = finfo(dtname or dtype).max
        vals = np.clip(vals, -fmax, fmax)
    return vals


def _constant_value(d: Domain) -> float:
    for c in (3.25, 0.375, -3.25, 7.0):
        if d.contains(c):
            return c
    if math.isfinite(d.lo) and math.isfinite(d.hi):
        return (d.lo + d.hi) / 2
    return d.lo + 1.0 if math.isfinite(d.lo) else d.hi - 1.0


def _g_random(ctx: _Ctx) -> np.ndarray:
    return ctx.baseline


def _g_zeros(ctx: _Ctx) -> np.ndarray | None:
    return np.zeros(ctx.count) if ctx.domain.contains(0.0) else None


def _g_neg_zero(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer or not ctx.domain.contains(0.0):
        return None
    return np.full(ctx.count, -0.0)


def _g_ones(ctx: _Ctx) -> np.ndarray | None:
    return np.ones(ctx.count) if ctx.domain.contains(1.0) else None


def _g_constant(ctx: _Ctx) -> np.ndarray:
    return np.full(ctx.count, _constant_value(ctx.domain))


def _g_size(ctx: _Ctx) -> np.ndarray:
    return ctx.baseline


def _g_large_n(ctx: _Ctx) -> np.ndarray:
    # positive values so that naive accumulators keep growing
    if not ctx.domain.integer and ctx.domain.contains(1.0) and ctx.domain.contains(2.0):
        return ctx.rng.uniform(1.0, 2.0, ctx.count)
    return ctx.baseline


def _inject(ctx: _Ctx, value: float) -> np.ndarray | None:
    if ctx.count == 0 or ctx.domain.integer:
        return None
    vals = ctx.baseline.copy()
    vals[ctx.count // 2] = value
    return vals


def _g_nan(ctx: _Ctx) -> np.ndarray | None:
    return _inject(ctx, math.nan)


def _g_posinf(ctx: _Ctx) -> np.ndarray | None:
    return _inject(ctx, math.inf) if ctx.domain.hi == math.inf else None


def _g_neginf(ctx: _Ctx) -> np.ndarray | None:
    return _inject(ctx, -math.inf) if ctx.domain.lo == -math.inf else None


def _g_huge(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    mags = float(ctx.fi.max) * ctx.rng.uniform(0.25, 0.5, ctx.count)
    return _magnitude_case(ctx, mags)


def _g_large(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    mags = math.sqrt(float(ctx.fi.max)) * ctx.rng.uniform(2.0, 8.0, ctx.count)
    return _magnitude_case(ctx, mags)


def _g_small(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    mags = float(ctx.fi.eps) * ctx.rng.uniform(1.0, 100.0, ctx.count)
    return _magnitude_case(ctx, mags)


def _g_tiny(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    mags = float(ctx.fi.tiny) * ctx.rng.uniform(1.0, 8.0, ctx.count)
    return _magnitude_case(ctx, mags)


def _g_subnormal(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    sub = float(ctx.fi.smallest_subnormal)
    mags = sub * ctx.rng.integers(1, 100, ctx.count).astype(np.float64)
    return _magnitude_case(ctx, mags)


def _g_alternating(ctx: _Ctx) -> np.ndarray | None:
    d = ctx.domain
    if not (d.allows_negative and d.allows_positive):
        return None
    k = np.arange(ctx.count, dtype=np.float64)
    vals = (1.0 + k) * np.where(k % 2 == 0, 1.0, -1.0)
    if d.integer or not d.contains_all(vals.tolist()):
        vals = np.where(k % 2 == 0, 1.0, -1.0) * min(1.0, d.hi, -d.lo)
        if not d.contains_all(vals.tolist()):
            return None
    return vals


def _g_cancellation(ctx: _Ctx) -> np.ndarray | None:
    if ctx.domain.integer:
        return None
    offset = float(round(16.0 / math.sqrt(float(ctx.fi.eps))))
    vals = offset + (np.arange(ctx.count) % 5).astype(np.float64)
    if not ctx.domain.contains_all([offset, offset + 4.0]):
        return None
    return vals


def _g_duplicates(ctx: _Ctx) -> np.ndarray | None:
    if ctx.count < 2:
        return None
    pool = ctx.baseline[: min(3, ctx.count)]
    return ctx.rng.choice(pool, size=ctx.count)


def _g_sorted(ctx: _Ctx) -> np.ndarray | None:
    return np.sort(ctx.baseline) if ctx.count >= 2 else None


def _g_reverse_sorted(ctx: _Ctx) -> np.ndarray | None:
    return np.sort(ctx.baseline)[::-1].copy() if ctx.count >= 2 else None


def _g_wide_range(ctx: _Ctx) -> np.ndarray | None:
    """Geometric sweep from the smallest normal to the largest finite value."""
    if ctx.domain.integer or ctx.count == 0:
        return None
    d = ctx.domain
    lo_mag = float(ctx.fi.tiny)
    hi_mag = float(ctx.fi.max)
    if math.isfinite(d.lo) and math.isfinite(d.hi):
        bound = max(abs(d.lo), abs(d.hi))
        if bound > 0:
            hi_mag = min(hi_mag, bound)
    with np.errstate(all="ignore"):
        mags = np.minimum(np.geomspace(lo_mag, hi_mag, ctx.count), hi_mag)
    if d.allows_negative and d.allows_positive:
        vals = mags * np.where(np.arange(ctx.count) % 2 == 0, 1.0, -1.0)
    elif d.allows_positive:
        vals = mags
    else:
        vals = -mags
    keep = np.array([d.contains(v) for v in vals.tolist()], dtype=bool)
    vals = vals[keep]
    if vals.size < 2:
        return None
    return np.resize(vals, ctx.count)


def _g_boundary(ctx: _Ctx) -> np.ndarray | None:
    d = ctx.domain
    if d.name == "any" or (math.isinf(d.lo) and math.isinf(d.hi)):
        return None
    pts: list[float] = []
    for b, closed, inward in ((d.lo, d.lo_closed, math.inf), (d.hi, d.hi_closed, -math.inf)):
        if math.isfinite(b):
            if closed:
                pts.append(b)
            elif not d.integer:
                pts.append(nextafter_in(b, ctx.dtname or ctx.dtype.name, inward))
    if not pts:
        return None
    return np.resize(np.array(pts, dtype=np.float64), ctx.count)


def _g_int_min(ctx: _Ctx) -> np.ndarray | None:
    if not ctx.domain.integer:
        return None
    lo = max(ctx.domain.lo, float(np.iinfo(ctx.dtype).min))
    return np.full(ctx.count, lo)


def _g_int_max(ctx: _Ctx) -> np.ndarray | None:
    if not ctx.domain.integer:
        return None
    hi = min(ctx.domain.hi, float(np.iinfo(ctx.dtype).max))
    return np.full(ctx.count, hi)


_Gen = Callable[[_Ctx], Union[np.ndarray, None]]

#: All case kinds, in generation order, with their generator and extra tags.
KINDS: dict[str, tuple[_Gen, frozenset[str]]] = {
    "random": (_g_random, frozenset()),
    "zeros": (_g_zeros, frozenset()),
    "neg_zero": (_g_neg_zero, frozenset({"signed_zero"})),
    "ones": (_g_ones, frozenset()),
    "constant": (_g_constant, frozenset()),
    "empty": (_g_size, frozenset()),
    "single": (_g_size, frozenset()),
    "pair": (_g_size, frozenset()),
    "nan": (_g_nan, frozenset({"nonfinite_input", "nan_input"})),
    "posinf": (_g_posinf, frozenset({"nonfinite_input", "inf_input"})),
    "neginf": (_g_neginf, frozenset({"nonfinite_input", "inf_input"})),
    "huge": (_g_huge, frozenset({"overflow_prone"})),
    "large": (_g_large, frozenset({"overflow_prone"})),
    "small": (_g_small, frozenset({"precision"})),
    "tiny": (_g_tiny, frozenset({"underflow_prone"})),
    "subnormal": (_g_subnormal, frozenset({"underflow_prone"})),
    "alternating": (_g_alternating, frozenset({"mixed_sign"})),
    "cancellation": (_g_cancellation, frozenset({"precision"})),
    "duplicates": (_g_duplicates, frozenset()),
    "sorted": (_g_sorted, frozenset()),
    "reverse_sorted": (_g_reverse_sorted, frozenset()),
    "wide_range": (_g_wide_range, frozenset({"overflow_prone", "underflow_prone"})),
    "boundary": (_g_boundary, frozenset()),
    "int_min": (_g_int_min, frozenset({"boundary"})),
    "int_max": (_g_int_max, frozenset({"boundary"})),
    "large_n": (_g_large_n, frozenset()),
}


def _kind_tags(kind: str) -> frozenset[str]:
    tags = {kind} | set(KINDS[kind][1])
    if kind in REGULAR_KINDS:
        tags.add("regular")
    if kind in _EXTREME_KINDS:
        tags.add("extreme")
    if kind in _SIZE_KINDS:
        tags.add("size")
    if kind not in _NONFINITE_KINDS:
        tags.add("finite_input")
    return frozenset(tags)


def _seed_for(seed: int, *parts: object) -> np.random.Generator:
    words = [int(seed) & 0xFFFFFFFF]
    words += [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(words)


def _cast(vals: np.ndarray, dtype: np.dtype, shape: tuple[int, ...],
          dtname: str = "") -> np.ndarray:
    with np.errstate(all="ignore"):
        if dtype.kind in "iu":
            ii = np.iinfo(dtype)
            vals = np.clip(np.nan_to_num(vals), ii.min, ii.max)
            out = np.asarray(np.rint(vals), dtype=dtype)
        elif dtype.kind == "b":
            out = np.asarray(vals != 0)
        else:
            out = np.asarray(vals, dtype=np.float64).astype(dtype)
            if dtname == BFLOAT16:
                out = round_bf16(out)
    return out.reshape(shape)


def generate_cases(
    inputs: Sequence[ArraySpec],
    dtypes: Sequence[str] | None = None,
    *,
    n: int = 8,
    large_n: int = 100_000,
    wide_n: int = 64,
    seed: int = 0,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    arg_names: Sequence[str] | None = None,
) -> list[Case]:
    """Generate named edge cases for a function with the given array inputs.

    For each dtype in ``dtypes`` (applied to every argument whose spec has
    ``dtype=None``) and each case kind in :data:`KINDS`, a *joint* case applies
    the kind to every argument. When there are several array arguments, each
    value kind is additionally applied to one argument at a time while the
    others keep baseline values (named e.g. ``zeros[b]``).

    Cases whose inputs fall outside an argument's domain are not generated.
    ``include``/``exclude`` take glob patterns, kinds or tags.
    Generation is fully deterministic for a given ``seed``.
    """
    if not inputs:
        raise ValueError("at least one array input is required")
    dtypes = tuple(dtypes) if dtypes else DEFAULT_DTYPES
    names = list(arg_names) if arg_names else []
    names = [
        (spec.name or (names[i] if i < len(names) else f"arg{i}"))
        for i, spec in enumerate(inputs)
    ]
    sweeps = any(spec.dtype is None for spec in inputs)
    dtypes = tuple(dtype_name(d) for d in dtypes)
    if not sweeps:
        dtypes = (dtype_name(inputs[0].dtype),)
    any_symbolic = any(spec.symbolic for spec in inputs)
    multi = len(inputs) > 1

    include = list(include) if include is not None else None
    exclude = list(exclude)
    cases: list[Case] = []
    seen: set[tuple] = set()

    for dt_name in dtypes:
        work = storage_dtype(dt_name)
        arg_names_dt = [dtype_name(s.dtype) if s.dtype is not None else dt_name for s in inputs]
        arg_dtypes = [storage_dtype(n) for n in arg_names_dt]

        def sized(kind: str) -> int:
            return {"empty": 0, "single": 1, "pair": 2, "large_n": large_n,
                    "wide_range": wide_n}.get(kind, n)

        baselines: dict[tuple[int, int], np.ndarray] = {}

        def baseline(i: int, size: int) -> np.ndarray:
            key = (i, size)
            if key not in baselines:
                spec = inputs[i]
                count = int(np.prod(spec.resolve_shape(size), dtype=np.int64))
                rng = _seed_for(seed, "baseline", i, dt_name, size)
                baselines[key] = _baseline(spec.domain, arg_dtypes[i], rng, count, arg_names_dt[i])
            return baselines[key]

        def gen_arg(kind: str, i: int, size: int) -> np.ndarray | None:
            spec = inputs[i]
            count = int(np.prod(spec.resolve_shape(size), dtype=np.int64))
            ctx = _Ctx(spec.domain, arg_dtypes[i], _seed_for(seed, kind, i, dt_name),
                       count, baseline(i, size), arg_names_dt[i])
            if arg_dtypes[i].kind != "f" and not spec.domain.integer:
                return None
            return KINDS[kind][0](ctx)

        def emit(name: str, kind: str, values: list[np.ndarray], size: int) -> None:
            arrays = tuple(
                _cast(v, arg_dtypes[i], inputs[i].resolve_shape(size), arg_names_dt[i])
                for i, v in enumerate(values)
            )
            key = (dt_name,) + tuple((a.dtype.str, a.shape, a.tobytes()) for a in arrays)
            if key in seen:
                return
            seen.add(key)
            case = Case(name=name, kind=kind, dtype=dt_name, args=arrays,
                        tags=_kind_tags(kind) | {dt_name}, size=size)
            if include is not None and not case_matches(case, include):
                return
            if exclude and case_matches(case, exclude):
                return
            cases.append(case)

        for kind in KINDS:
            size = sized(kind)
            if kind in _SIZE_KINDS:
                if not any_symbolic:
                    continue  # fixed shapes: size variations do not apply
                vals = [gen_arg(kind, i, size) for i in range(len(inputs))]
                vals = [v if v is not None else baseline(i, size) for i, v in enumerate(vals)]
                emit(kind, kind, vals, size)
                continue
            # joint case (with several arguments: only if every argument
            # supports the kind; otherwise the per-argument variants cover it)
            gen = [gen_arg(kind, i, size) for i in range(len(inputs))]
            if all(v is not None for v in gen) or (not multi and gen[0] is not None):
                vals = [v if v is not None else baseline(i, size) for i, v in enumerate(gen)]
                emit(kind, kind, vals, size)
            # per-argument variants
            if multi and kind != "random":
                for i, v in enumerate(gen):
                    if v is None:
                        continue
                    vals = [baseline(j, size) for j in range(len(inputs))]
                    vals[i] = v
                    emit(f"{kind}[{names[i]}]", kind, vals, size)
    return cases
