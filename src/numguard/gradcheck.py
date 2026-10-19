"""Gradient checks: automatic differentiation against finite differences.

numguard checks the gradient of ``sum(f(x))`` with respect to every floating
argument, at the same edge cases as the values:

* **reverse mode** (``jax.grad``, ``torch.autograd.grad``) or a gradient
  function you supply (``grad=callable``) is compared with a float64
  reference: ``grad_reference`` if given, otherwise Richardson-extrapolated
  central differences of the value reference (``reference=``) or of the
  function itself evaluated in float64;
* **primal consistency** (JAX and PyTorch): the primal value returned by a
  forward-mode ``jvp`` must equal ``f(x)``. A custom JVP rule that returns a
  different primal is a silent bug (see ``jax-ml/jax#41017``);
* **forward vs reverse** (``elementwise=True`` only): the JVP with an all-ones
  tangent must equal the reverse-mode gradient, element by element.

Finite differences are taken only where they mean something: points whose
step would leave the declared input domain, or where the estimates at three
step sizes disagree (a kink such as ``abs`` at 0), are skipped rather than
reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .backends import Backend, BackendCall
from .cases import ArraySpec
from .dtypes import BFLOAT16

__all__ = ["AutodiffResult", "autodiff", "finite_difference_grad", "grad_tolerance",
           "GRAD_BACKENDS"]

#: Backends whose automatic differentiation numguard can drive (``grad=True``).
GRAD_BACKENDS = ("jax", "torch")

_GRAD_RTOL = {"float16": 5e-2, BFLOAT16: 5e-2, "float32": 2e-3, "float64": 1e-6}


def grad_tolerance(dtname: str, override: float | None = None) -> float:
    """Relative tolerance for a gradient computed in ``dtname``."""
    if override is not None:
        return float(override)
    return _GRAD_RTOL.get(dtname, 1e-6)


@dataclass
class AutodiffResult:
    """Gradients of ``sum(f(x))`` (one per differentiated argument) and JVP outputs."""

    grads: list[np.ndarray | None]
    jvp_primal: np.ndarray | None = None
    jvp_tangent: np.ndarray | None = None
    jvp_error: str | None = None


def _first(out: Any) -> Any:
    if isinstance(out, (tuple, list)):
        if len(out) != 1:
            raise ValueError(f"gradient checks need a single output, got {len(out)}")
        return out[0]
    if isinstance(out, dict):
        if len(out) != 1:
            raise ValueError(f"gradient checks need a single output, got {len(out)}")
        return next(iter(out.values()))
    return out


def autodiff(backend: Backend, fn: Callable[..., Any], args: Sequence[np.ndarray],
             dtnames: Sequence[str], idx: Sequence[int],
             kwargs: dict[str, Any]) -> AutodiffResult:
    """Run ``fn``'s automatic differentiation in ``backend`` at ``args``.

    ``idx`` lists the arguments to differentiate. Raises whatever the backend
    raises for the reverse pass; a failing forward pass is recorded in
    ``jvp_error`` instead, because not every function supports forward mode.
    """
    if backend.name == "jax":
        return _jax_autodiff(backend, fn, args, dtnames, idx, kwargs)
    if backend.name == "torch":
        return _torch_autodiff(backend, fn, args, dtnames, idx, kwargs)
    raise ValueError(f"automatic differentiation is not available for backend {backend.name!r}")


def _jax_autodiff(backend: Backend, fn: Callable[..., Any], args: Sequence[np.ndarray],
                  dtnames: Sequence[str], idx: Sequence[int],
                  kwargs: dict[str, Any]) -> AutodiffResult:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    xs = [backend.to_backend(a, d) for a, d in zip(args, dtnames)]

    def partial(*diff: Any) -> Any:
        full = list(xs)
        for i, v in zip(idx, diff):
            full[i] = v
        return _first(fn(*full, **kwargs))

    primals = tuple(xs[i] for i in idx)
    grads = jax.grad(lambda *d: jnp.sum(partial(*d)), argnums=tuple(range(len(idx))))(*primals)
    res = AutodiffResult(grads=[_to_np(backend, g) for g in grads])
    try:
        p, t = jax.jvp(partial, primals, tuple(jnp.ones_like(v) for v in primals))
        res.jvp_primal, res.jvp_tangent = _to_np(backend, p), _to_np(backend, t)
    except Exception as exc:  # noqa: BLE001 - forward mode is optional
        res.jvp_error = f"{type(exc).__name__}: {exc}"
    return res


def _torch_autodiff(backend: Backend, fn: Callable[..., Any], args: Sequence[np.ndarray],
                    dtnames: Sequence[str], idx: Sequence[int],
                    kwargs: dict[str, Any]) -> AutodiffResult:
    import torch  # noqa: PLC0415

    xs = [backend.to_backend(a, d) for a, d in zip(args, dtnames)]
    leaves = []
    for i in idx:
        xs[i] = xs[i].detach().clone().requires_grad_(True)
        leaves.append(xs[i])
    out = _first(fn(*xs, **kwargs))
    grads = torch.autograd.grad(out.sum(), leaves, allow_unused=True)
    res = AutodiffResult(grads=[None if g is None else _to_np(backend, g) for g in grads])

    base = [x.detach() for x in xs]

    def partial(*diff: Any) -> Any:
        full = list(base)
        for i, v in zip(idx, diff):
            full[i] = v
        return _first(fn(*full, **kwargs))

    try:
        primals = tuple(base[i] for i in idx)
        p, t = torch.func.jvp(partial, primals, tuple(torch.ones_like(v) for v in primals))
        res.jvp_primal, res.jvp_tangent = _to_np(backend, p), _to_np(backend, t)
    except Exception as exc:  # noqa: BLE001 - forward mode is optional
        res.jvp_error = f"{type(exc).__name__}: {exc}"
    return res


def _to_np(backend: Backend, x: Any) -> np.ndarray:
    return np.asarray(backend.to_numpy(x)[0])


def _value64(valfn: Callable[..., Any], args: list[np.ndarray], kwargs: dict[str, Any]) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.asarray(_first(valfn(*args, **kwargs)), dtype=np.float64)
    return out


def _in_domain(spec_i: ArraySpec, v: np.ndarray) -> np.ndarray:
    dom = spec_i.domain
    with np.errstate(invalid="ignore"):
        lo_ok = v >= dom.lo if dom.lo_closed else v > dom.lo
        hi_ok = v <= dom.hi if dom.hi_closed else v < dom.hi
    return np.asarray(lo_ok & hi_ok & np.isfinite(v))


def finite_difference_grad(valfn: Callable[..., Any], args64: Sequence[np.ndarray], i: int,
                           spec_i: ArraySpec | None, kwargs: dict[str, Any], *,
                           elementwise: bool, max_points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Float64 estimate of ``d sum(f) / d args[i]`` and its uncertainty.

    Central differences at steps ``h``, ``h/2`` and ``h/4`` are combined by
    Richardson extrapolation into two estimates; their difference plus a
    rounding-error bound is the uncertainty. A point is kept only if the
    function is locally linear at the smallest step and a central difference
    at an unrelated step (``0.0921 h``) agrees, so that estimates which agree
    by coincidence are not trusted. The base step is
    ``2**-10 * max(1, |x|)`` for elementwise functions; otherwise it is also
    capped by the spread of the argument's values, the scale on which
    functions such as ``logsumexp`` or ``softmax`` vary.

    Returns ``(estimate, uncertainty)``. Both are NaN where the estimate is
    not trustworthy: the function is not finite at the point, every value
    sampled is an integer (a step function such as ``round``, whose
    derivative is zero almost everywhere), the step would leave the declared
    domain or overflow,
    the estimates disagree by more than 10% (a kink, or a function varying
    faster than the step can resolve), or, without ``elementwise=True``, the
    element is beyond the first ``max_points`` perturbed one at a time.
    """
    x = np.asarray(args64[i], dtype=np.float64)
    est = np.full(x.shape, np.nan)
    unc = np.full(x.shape, np.nan)
    if x.size == 0:
        return est, unc
    eps = np.finfo(np.float64).eps
    with np.errstate(all="ignore"):
        scale = np.maximum(1.0, np.abs(x))
        if not elementwise and x.size > 1:
            fin = x[np.isfinite(x)]
            spread = float(np.ptp(fin)) if fin.size else 0.0
            scale = np.maximum(1.0, np.minimum(np.abs(x), spread))
        h = 2.0 ** -10 * scale

    def reduce(vals: np.ndarray) -> np.ndarray:
        return vals if elementwise else np.asarray(np.sum(vals))

    def central(step: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plus, minus = list(args64), list(args64)
        d = step if mask is None else np.where(mask, step, 0.0)
        plus[i], minus[i] = x + d, x - d
        fp, fm = _value64(valfn, plus, kwargs), _value64(valfn, minus, kwargs)
        if elementwise and (fp.shape != x.shape or fm.shape != x.shape):
            raise ValueError("elementwise=True but the output shape differs from the input shape")
        sp, sm = reduce(fp), reduce(fm)
        hh = step if mask is None else step[mask]
        integral = (sp == np.round(sp)) & (sm == np.round(sm))
        return (sp - sm) / (2.0 * hh), np.maximum(np.abs(sp), np.abs(sm)), sp + sm, integral

    def estimate(mask: np.ndarray | None, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (d1, a1, _, i1), (d2, a2, _, i2), (d3, a3, s3, i3), (d4, _, _, i4) = (
            central(h * s, mask) for s in (1.0, 0.5, 0.25, 0.0921))
        hh = h if mask is None else h[mask]
        r1, r2 = (4 * d2 - d1) / 3, (4 * d3 - d2) / 3
        rounding = 16 * eps * np.maximum(np.maximum(a1, a2), a3) / (hh / 4)
        unc = np.abs(r1 - r2) + rounding
        # the smallest step must be locally linear: curvature small next to slope
        curvature = np.abs(s3 - 2 * f0)
        slope = np.abs(d3) * 2 * (hh / 4)
        linear = np.isfinite(f0) & (curvature <= 0.25 * slope + 64 * eps * np.maximum(a3, np.abs(f0)))
        # integer values at every step: a step function (round, floor, sign), whose derivative is
        # zero almost everywhere; differences across its steps say nothing about it
        steps = i1 & i2 & i3 & i4 & (f0 == np.round(f0))
        linear &= ~steps
        # a non-dyadic step guards against estimates that agree by coincidence
        # (a periodic function sampled at multiples of its period)
        consistent = np.abs(d4 - r2) <= 0.1 * np.abs(r2) + unc
        return r2, np.where(linear & consistent, unc, np.nan)

    with np.errstate(all="ignore"):
        ok = np.isfinite(x + 2 * h) & np.isfinite(x - 2 * h)
        if spec_i is not None:
            ok &= _in_domain(spec_i, x - 2 * h) & _in_domain(spec_i, x + 2 * h)
        f0 = reduce(_value64(valfn, list(args64), kwargs))
        if elementwise:
            e, u = estimate(None, f0)
            good = ok & np.isfinite(e) & np.isfinite(u) & (u <= 0.1 * np.abs(e))
            est[good], unc[good] = e[good], u[good]
            return est, unc
        flat_ok = ok.ravel()
        points = [k for k in range(x.size) if flat_ok[k]][:max_points]
        for k in points:
            mask = np.zeros(x.size, bool)
            mask[k] = True
            e, u = estimate(mask.reshape(x.shape), f0)
            e0, u0 = float(np.ravel(e)[0]), float(np.ravel(u)[0])
            if np.isfinite(e0) and np.isfinite(u0) and u0 <= 0.1 * abs(e0):
                est.ravel()[k], unc.ravel()[k] = e0, u0
    return est, unc


def unwrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The user's function behind a :class:`BackendCall`."""
    return fn.fn if isinstance(fn, BackendCall) else fn
