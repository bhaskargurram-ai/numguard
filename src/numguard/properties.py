"""Metamorphic properties: relations between ``f(x)`` and ``f(T(x))``.

A :class:`Property` transforms the inputs of a case (``T``) and states what
the output on the transformed inputs should be, given the original output.
For example translation invariance of a variance says ``f(x + c) == f(x)``;
scale equivariance of a norm says ``f(2 x) == 2 f(x)``.

Properties can be given as objects or as strings: ``"translation_invariant"``,
``"scale_equivariant:2"`` (exponent 2), ``"symmetric"``, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np

from .cases import ArraySpec

__all__ = [
    "Property",
    "PropertyLike",
    "permutation_invariant",
    "permutation_equivariant",
    "translation_invariant",
    "translation_equivariant",
    "scale_invariant",
    "scale_equivariant",
    "symmetric",
    "odd",
    "even",
    "parse_property",
    "PROPERTY_FACTORIES",
]

Args = tuple[np.ndarray, ...]
Outputs = list[np.ndarray]
Transform = Callable[[Args, Sequence[ArraySpec]], Union[Args, None]]
Expect = Callable[[Outputs, Args], Union[Outputs, None]]


@dataclass(frozen=True)
class Property:
    """A metamorphic relation ``f(transform(x)) == expect(f(x))``.

    ``transform`` returns ``None`` when it does not apply to a case (for
    example when the transformed inputs would leave the declared domain);
    such cases are skipped. ``cases`` lists case kinds/tags/globs the
    property is checked on (default: regular, moderate-magnitude cases).
    """

    name: str
    transform: Transform
    expect: Expect
    description: str = ""
    cases: tuple[str, ...] = ("regular",)
    rtol_factor: float = 4.0

    def __str__(self) -> str:
        return self.name


PropertyLike = Union[Property, str]


def _in_domain(arrays: Args, specs: Sequence[ArraySpec], idx: Sequence[int]) -> bool:
    for i in idx:
        a = arrays[i]
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            return False
        if not specs[i].domain.contains_all(a.ravel().tolist()):
            return False
    return True


def _float_args(args: Args, which: Sequence[int] | None) -> list[int]:
    idx = range(len(args)) if which is None else which
    return [i for i in idx if i < len(args) and args[i].dtype.kind == "f"]


def _perm(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    p = rng.permutation(n)
    if n > 1 and np.array_equal(p, np.arange(n)):
        p = np.roll(p, 1)
    return p


def _permute_args(args: Args, args_idx: Sequence[int] | None) -> tuple[Args, np.ndarray] | None:
    idx = list(range(len(args))) if args_idx is None else list(args_idx)
    lengths = {args[i].shape[-1] for i in idx if args[i].ndim >= 1}
    if len(lengths) != 1:
        return None
    n = lengths.pop()
    if n < 2:
        return None
    p = _perm(n)
    new = list(args)
    for i in idx:
        if args[i].ndim >= 1:
            new[i] = args[i][..., p]
    return tuple(new), p


def permutation_invariant(args: Sequence[int] | None = None) -> Property:
    """``f(x[..., p]) == f(x)`` for a permutation ``p`` of the last axis.

    The same permutation is applied to every argument in ``args``
    (default: all), so paired inputs stay aligned.
    """

    def transform(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
        r = _permute_args(a, args)
        return None if r is None else r[0]

    return Property("permutation_invariant", transform, lambda out, a: out,
                    "f(permute(x)) == f(x)")


def permutation_equivariant(args: Sequence[int] | None = None) -> Property:
    """``f(x[..., p]) == f(x)[..., p]`` (e.g. softmax, elementwise functions)."""

    def transform(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
        r = _permute_args(a, args)
        return None if r is None else r[0]

    def expect(out: Outputs, a: Args) -> Outputs | None:
        r = _permute_args(a, args)
        if r is None:
            return None
        p = r[1]
        res = []
        for o in out:
            if o.ndim == 0 or o.shape[-1] != p.size:
                return None
            res.append(o[..., p])
        return res

    return Property("permutation_equivariant", transform, expect,
                    "f(permute(x)) == permute(f(x))")


def _shifted(shift: float, args_idx: Sequence[int] | None) -> Transform:
    def transform(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
        idx = _float_args(a, args_idx)
        if not idx:
            return None
        new = list(a)
        with np.errstate(all="ignore"):
            for i in idx:
                new[i] = (a[i] + a[i].dtype.type(shift)).astype(a[i].dtype)
        out = tuple(new)
        return out if _in_domain(out, specs, idx) else None

    return transform


def translation_invariant(shift: float = 1.0, args: Sequence[int] | None = (0,)) -> Property:
    """``f(x + c) == f(x)`` (variance, standard deviation, softmax, ...)."""
    return Property(f"translation_invariant(c={shift:g})", _shifted(shift, args),
                    lambda out, a: out, "f(x + c) == f(x)")


def translation_equivariant(shift: float = 1.0, args: Sequence[int] | None = (0,)) -> Property:
    """``f(x + c) == f(x) + c`` (mean, median, logsumexp, max, ...)."""

    def expect(out: Outputs, a: Args) -> Outputs:
        with np.errstate(all="ignore"):
            return [(o + o.dtype.type(shift)) if o.dtype.kind == "f" else o + shift for o in out]

    return Property(f"translation_equivariant(c={shift:g})", _shifted(shift, args),
                    expect, "f(x + c) == f(x) + c")


def scale_equivariant(exponent: float = 1.0, factor: float = 2.0,
                      args: Sequence[int] | None = None) -> Property:
    """``f(a * x) == a**k * f(x)``; the default factor 2 is exact in binary FP."""

    def transform(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
        idx = _float_args(a, args)
        if not idx:
            return None
        new = list(a)
        with np.errstate(all="ignore"):
            for i in idx:
                new[i] = (a[i] * a[i].dtype.type(factor)).astype(a[i].dtype)
        out = tuple(new)
        return out if _in_domain(out, specs, idx) else None

    def expect(out: Outputs, a: Args) -> Outputs:
        scale = float(factor) ** float(exponent)
        with np.errstate(all="ignore"):
            return [(o * o.dtype.type(scale)) if o.dtype.kind == "f" else o * scale for o in out]

    name = "scale_invariant" if exponent == 0 else f"scale_equivariant(k={exponent:g})"
    return Property(name, transform, expect, f"f({factor:g}*x) == {factor:g}**{exponent:g} * f(x)")


def scale_invariant(factor: float = 2.0, args: Sequence[int] | None = None) -> Property:
    """``f(a * x) == f(x)`` (cosine similarity, normalization direction, ...)."""
    return scale_equivariant(0.0, factor, args)


def symmetric(i: int = 0, j: int = 1) -> Property:
    """``f(..., x_i, ..., x_j, ...) == f(..., x_j, ..., x_i, ...)``."""

    def transform(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
        if max(i, j) >= len(a) or a[i].shape != a[j].shape or a[i].dtype != a[j].dtype:
            return None
        new = list(a)
        new[i], new[j] = a[j], a[i]
        out = tuple(new)
        return out if _in_domain(out, specs, (i, j)) else None

    return Property("symmetric", transform, lambda out, a: out,
                    f"f swaps argument {i} and {j} without changing the result")


def _negated(a: Args, specs: Sequence[ArraySpec]) -> Args | None:
    idx = _float_args(a, None)
    if not idx:
        return None
    new = list(a)
    for i in idx:
        new[i] = -a[i]
    out = tuple(new)
    return out if _in_domain(out, specs, idx) else None


def odd() -> Property:
    """``f(-x) == -f(x)``."""
    return Property("odd", _negated, lambda out, a: [-o for o in out], "f(-x) == -f(x)")


def even() -> Property:
    """``f(-x) == f(x)``."""
    return Property("even", _negated, lambda out, a: out, "f(-x) == f(x)")


PROPERTY_FACTORIES: dict[str, Callable[..., Property]] = {
    "permutation_invariant": permutation_invariant,
    "permutation_equivariant": permutation_equivariant,
    "translation_invariant": translation_invariant,
    "translation_equivariant": translation_equivariant,
    "scale_invariant": scale_invariant,
    "scale_equivariant": scale_equivariant,
    "symmetric": symmetric,
    "odd": odd,
    "even": even,
}


def parse_property(prop: PropertyLike) -> Property:
    """Turn ``"name"`` or ``"name:value"`` into a :class:`Property`.

    The optional value is the first parameter of the factory: the shift for
    translation properties, the exponent for ``scale_equivariant`` and the
    factor for ``scale_invariant``.
    """
    if isinstance(prop, Property):
        return prop
    if not isinstance(prop, str):
        raise TypeError(f"cannot interpret {prop!r} as a property")
    name, _, value = prop.partition(":")
    name = name.strip().lower().replace("-", "_")
    if name not in PROPERTY_FACTORIES:
        raise ValueError(
            f"unknown property {name!r}; available: {', '.join(sorted(PROPERTY_FACTORIES))}"
        )
    factory = PROPERTY_FACTORIES[name]
    if value.strip():
        return factory(float(value))
    return factory()

