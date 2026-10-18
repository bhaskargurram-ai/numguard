"""The checks numguard runs on every generated case.

Each check returns :class:`~numguard.report.CheckResult` objects. Checks are
driven by :func:`run_checks`, which executes the function under test (and
the reference, if any) once per case and then evaluates:

``error``
    The function raised an exception where none was expected.
``mutation``
    The function modified its input arrays in place.
``dtype``
    Floating outputs must keep the input dtype (or match a declared dtype).
``finite``
    Finite inputs must give finite outputs (unless the reference agrees the
    output is non-finite, or the case is explicitly allowed).
``nan``
    NaN inputs must propagate (``nan="propagate"``) or raise (``nan="raise"``).
``fp_signals``
    Optional: no overflow / invalid / divide-by-zero signals on finite inputs.
``reference``
    Agreement with a reference callable within ``rtol``/``atol``, with ULP
    error reporting.
``bounds``
    Optional: finite outputs must lie in a declared interval (with a slack of
    one default ``rtol`` for floating outputs).
``post:<name>``
    Optional user postconditions ``post(args, outputs) -> None | bool | str``
    evaluated on every finite-input case.
``batch``
    ``f(stack(xs)) == stack(f(x) for x in xs)`` for batched functions.
``monotonic``
    Optional: an elementwise function is non-decreasing / non-increasing.
``prop:<name>``
    User-declared metamorphic properties (see :mod:`numguard.properties`).
``isolation``
    ``elementwise=True``: each element's result is the same alone, in the
    full array, and next to every other case's inputs.
``grad``, ``grad:primal``, ``grad:forward``
    ``grad=True`` (JAX, PyTorch) or ``grad=callable``: the gradient of
    ``sum(f)`` agrees with a float64 reference, a forward-mode ``jvp``
    returns ``f(x)`` as its primal, and forward and reverse mode agree (see
    :mod:`numguard.gradcheck`).
"""

from __future__ import annotations

import fnmatch
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from .backends import BackendCall, wrap
from .cases import Case, case_matches
from .dtypes import BFLOAT16, dtype_name, is_float_name, round_bf16
from .reference import HighPrecReference, default_tolerance, ulp_distance
from .report import FAIL, PASS, SKIP, CheckResult, Report

if TYPE_CHECKING:  # pragma: no cover
    from .api import Spec

__all__ = ["Run", "Comparison", "call", "compare", "run_checks", "FP_SIGNALS"]

#: Floating-point exception kinds that numguard records (underflow is ignored).
FP_SIGNALS = ("overflow", "invalid value", "divide by zero")


# --------------------------------------------------------------------------
# Running a function on a case
# --------------------------------------------------------------------------


@dataclass
class Run:
    """The outcome of calling a function once on a set of inputs."""

    outputs: list[np.ndarray] | None = None
    weak: list[bool] = field(default_factory=list)
    error: BaseException | None = None
    fp_events: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mutated: list[int] = field(default_factory=list)
    #: dtype name of every output as the function produced it (for backends:
    #: the backend dtype, e.g. ``bfloat16``, before conversion to NumPy).
    out_dtypes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.outputs is not None

    def error_text(self) -> str:
        if self.error is None:
            return ""
        msg = str(self.error).strip().splitlines()
        return f"{type(self.error).__name__}: {msg[0] if msg else ''}".strip().rstrip(":")


def _normalize(out: Any) -> tuple[list[np.ndarray], list[bool]]:
    if isinstance(out, Mapping):
        items = [out[k] for k in sorted(out)]
    elif isinstance(out, tuple):
        items = list(out)
    else:
        items = [out]
    arrays, weak = [], []
    for it in items:
        weak.append(isinstance(it, (int, float, complex)) and not isinstance(it, np.generic))
        arrays.append(np.asarray(it))
    return arrays, weak


def call(fn: Callable[..., Any], args: Sequence[np.ndarray],
         kwargs: Mapping[str, Any] | None = None, *, check_mutation: bool = True) -> Run:
    """Call ``fn(*args, **kwargs)`` capturing outputs, exceptions and FP signals.

    The function receives copies of ``args``; when ``check_mutation`` is set,
    the copies are compared afterwards to detect in-place modification.
    """
    run = Run()
    work = [np.array(a, copy=True) for a in args]
    events: list[str] = []

    def handler(err: str, flag: int) -> None:
        events.append(err)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="call", invalid="call", divide="call", under="ignore",
                         call=handler):
            try:
                raw = fn(*work, **(kwargs or {}))
                run.outputs, run.weak = _normalize(raw)
                names = fn.last_out_dtypes() if isinstance(fn, BackendCall) else None
                if names is None or len(names) != len(run.outputs):
                    names = [o.dtype.name for o in run.outputs]
                run.out_dtypes = list(names)
            except Exception as exc:  # noqa: BLE001 - we report every failure
                run.error = exc
    run.fp_events = sorted({e for e in events if e in FP_SIGNALS})
    run.warnings = [str(w.message) for w in caught]
    if check_mutation:
        for i, (a, w) in enumerate(zip(args, work)):
            if a.shape != w.shape or a.tobytes() != w.tobytes():
                run.mutated.append(i)
    return run


# --------------------------------------------------------------------------
# Numerical comparison
# --------------------------------------------------------------------------


@dataclass
class Comparison:
    ok: bool
    message: str = ""
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    max_ulp: float | None = None
    n_bad: int = 0
    example: str | None = None


def _tol_value(tol: float | Mapping[str, float] | None, dtype: Any,
               default: float) -> float:
    if tol is None:
        return default
    if isinstance(tol, Mapping):
        return float(tol.get(dtype_name(dtype), tol.get("default", default)))
    return float(tol)


def _fmt_val(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return repr(v)
    return repr(f) if not np.isfinite(f) else f"{f:.6g}"


def compare(actual: np.ndarray, expected: np.ndarray, *, rtol: Any = None, atol: Any = None,
            tol_dtype: np.dtype | None = None, rtol_factor: float = 1.0,
            label: str = "expected") -> Comparison:
    """Compare ``actual`` with ``expected`` elementwise.

    ``expected`` is first rounded to ``actual``'s dtype (so that a value
    beyond the dtype's range is expected to be ``inf``). NaN matches NaN,
    infinities must match exactly, and finite values must satisfy
    ``|a - e| <= atol + rtol * max(|e|, tiny)`` where ``tiny`` is the
    smallest normal number of the dtype (so subnormal results are judged
    relative to ``tiny`` rather than to themselves). ``rtol``/``atol`` may be floats or dicts keyed
    by dtype name; defaults come from :func:`numguard.reference.default_tolerance`.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        try:
            np.broadcast_shapes(actual.shape, expected.shape)
            if actual.size != expected.size or actual.ndim != expected.ndim:
                raise ValueError
        except ValueError:
            return Comparison(False, f"shape {actual.shape} != {label} {expected.shape}", n_bad=1)
    adt = actual.dtype
    bf16 = tol_dtype is not None and dtype_name(tol_dtype) == BFLOAT16
    tdt = np.dtype(np.float32) if bf16 else (np.dtype(tol_dtype) if tol_dtype is not None else adt)
    d_rtol, d_floor = default_tolerance(BFLOAT16 if bf16 else tdt)
    if bf16 and adt.kind == "f":
        d_rtol = max(d_rtol, default_tolerance(adt)[0])
        d_floor = max(d_floor, default_tolerance(adt)[1])
        with np.errstate(all="ignore"):
            # a value beyond bfloat16's range is expected to be +-inf
            expected = round_bf16(np.asarray(expected, dtype=np.float64).astype(np.float32))
    elif adt.kind == "f" and tdt.kind == "f":
        d_rtol = max(d_rtol, default_tolerance(adt)[0])
        d_floor = max(d_floor, default_tolerance(adt)[1])
    tol_key = BFLOAT16 if bf16 else tdt
    rt = _tol_value(rtol, tol_key, d_rtol) * rtol_factor
    at = _tol_value(atol, tol_key, 0.0)

    if adt.kind in "iub" and expected.dtype.kind in "iub":
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        bad = diff > at
        n_bad = int(bad.sum())
        max_abs = float(diff.max()) if diff.size else 0.0
        ex = None
        if n_bad:
            i = np.unravel_index(int(np.argmax(diff)), diff.shape)
            where = f" at {tuple(int(k) for k in i)}" if actual.ndim else ""
            ex = f"got {actual[i]}, {label} {expected[i]}{where}"
        msg = "" if not n_bad else (
            f"{ex} ({n_bad}/{actual.size} elements differ)" if actual.size > 1 else str(ex))
        return Comparison(n_bad == 0, msg,
                          max_abs_err=max_abs, n_bad=n_bad, example=ex)

    if adt.kind not in "fiub":
        eq = np.array_equal(actual, expected)
        return Comparison(bool(eq), "" if eq else "values differ", n_bad=0 if eq else 1)

    cmp_dt = adt if adt.kind == "f" else np.dtype(np.float64)
    with np.errstate(all="ignore"):
        a = actual.astype(cmp_dt)
        e = expected.astype(cmp_dt)
        a_l = a.astype(np.longdouble)
        e_l = e.astype(np.longdouble)
        nan_a, nan_e = np.isnan(a), np.isnan(e)
        inf_a, inf_e = np.isinf(a), np.isinf(e)
        fin = ~(nan_a | nan_e | inf_a | inf_e)
        diff = np.where(fin, np.abs(a_l - e_l), np.longdouble(0))
        mismatch_nonfinite = (nan_a != nan_e) | ((inf_a | inf_e) & ~nan_a & ~nan_e & (a != e))
        # relative error is measured against max(|expected|, smallest normal):
        # in the subnormal range errors count relative to the smallest normal
        denom = np.maximum(np.abs(e_l), np.longdouble(d_floor) if d_floor else np.longdouble(0))
        tol = at + rt * denom
        bad_fin = fin & (diff > tol)
        bad = bad_fin | mismatch_nonfinite
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(fin & (denom > 0), diff / np.where(denom > 0, denom, 1), np.longdouble(0))
        rel = np.where(mismatch_nonfinite, np.longdouble(np.inf), rel)
    n_bad = int(bad.sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    if np.any(mismatch_nonfinite):
        max_abs = float("inf")
    max_rel = float(rel.max()) if rel.size else 0.0
    max_ulp = None
    if cmp_dt.kind == "f" and cmp_dt.itemsize in (2, 4, 8) and a.size:
        u = ulp_distance(a, e, BFLOAT16 if bf16 and cmp_dt == np.float32 else cmp_dt)
        u = np.where(np.isnan(a) | np.isnan(e), np.where(nan_a == nan_e, 0.0, np.inf), u)
        max_ulp = float(u.max())
    ex = None
    if n_bad:
        score = np.where(bad, np.where(mismatch_nonfinite, np.inf, rel.astype(np.float64)), -1.0)
        i = np.unravel_index(int(np.argmax(score)), score.shape)
        where = f" at {tuple(int(k) for k in i)}" if a.ndim else ""
        ex = f"got {_fmt_val(a[i])}, {label} {_fmt_val(e[i])}{where}"
    msg = ""
    if n_bad:
        msg = ex or ""
        if a.size > 1:
            msg += f" ({n_bad}/{a.size} elements off)"
    return Comparison(n_bad == 0, msg, max_abs, max_rel, max_ulp, n_bad, ex)


# --------------------------------------------------------------------------
# Check driver
# --------------------------------------------------------------------------


class _Ctx:
    """Shared state for one ``run_checks`` invocation."""

    def __init__(self, spec: "Spec", report: Report):
        self.spec = spec
        self.report = report

    def ignored(self, check: str, case: Case) -> bool:
        """Does an ``ignore`` entry (``"check"`` or ``"check:case-pattern"``) match?

        Check names may themselves contain colons (``prop:...``, ``post:...``),
        so every split point is tried; the check part is a glob.
        """
        for entry in self.spec.ignore:
            if fnmatch.fnmatchcase(check, entry):
                return True
            for i, ch in enumerate(entry):
                if ch == ":" and fnmatch.fnmatchcase(check, entry[:i]) and \
                        case_matches(case, [entry[i + 1:] or "*"]):
                    return True
        return False

    def add(self, check: str, case: Case, status: str, message: str = "", *,
            run: Run | None = None, cmp: Comparison | None = None) -> None:
        if status != SKIP and self.ignored(check, case):
            status, message = SKIP, f"ignored ({status}): {message}" if message else "ignored"
        r = CheckResult(check=check, case=case.name, dtype=case.dtype, status=status,
                        message=message, fp_events=list(run.fp_events) if run else [])
        if cmp is not None:
            r.max_abs_err, r.max_rel_err, r.max_ulp = cmp.max_abs_err, cmp.max_rel_err, cmp.max_ulp
            r.example = cmp.example
        self.report.results.append(r)


def _ref_args(reference: Callable[..., Any], case: Case,
              upcast: bool = True) -> tuple[np.ndarray, ...]:
    if isinstance(reference, HighPrecReference) or not upcast:
        return case.args
    out = []
    for a in case.args:
        if a.dtype.kind == "f" and a.dtype.itemsize < 8:
            out.append(a.astype(np.float64))
        else:
            out.append(a)
    return tuple(out)


def _tol_dtype(out: np.ndarray, weak: bool, case: Case) -> Any:
    if case.dtype == BFLOAT16:
        if out.dtype.kind in "iub" and not weak:
            return out.dtype
        # outputs of bfloat16 cases are float32 after conversion (or the
        # function upcast): judge them with bfloat16 tolerances
        return BFLOAT16 if (weak or out.dtype.kind != "f" or out.dtype.itemsize <= 4) else out.dtype
    case_dt = np.dtype(case.dtype)
    if out.dtype.kind in "iub" and not weak:
        return out.dtype  # integer outputs: exact unless atol keyed by dtype name
    if weak or out.dtype.kind != "f":
        return case_dt if case_dt.kind == "f" else out.dtype
    if case_dt.kind == "f" and default_tolerance(case_dt)[0] > default_tolerance(out.dtype)[0]:
        return case_dt
    return out.dtype


def _round_like(ref: Any, out: np.ndarray, case: Case) -> np.ndarray:
    """Round a reference output to the dtype the function's output has (bfloat16-aware)."""
    r = np.asarray(ref)
    if out.dtype.kind not in "fc":
        return r
    r = r.astype(out.dtype)
    if case.dtype == BFLOAT16 and out.dtype == np.float32:
        r = round_bf16(r)
    return r


def _nonfinite_mask(arrs: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [~np.isfinite(a) if a.dtype.kind in "fc" else np.zeros(a.shape, bool) for a in arrs]


def _check_case(ctx: _Ctx, fn: Callable[..., Any], case: Case) -> Run:
    spec = ctx.spec
    run = call(fn, case.args, spec.kwargs)
    ref_run: Run | None = None
    if spec.reference is not None:
        ref_run = call(spec.reference, _ref_args(spec.reference, case, spec.ref_upcast),
                       spec.ref_kwargs,
                       check_mutation=False)
    may_raise = case_matches(case, spec.may_raise)
    nan_case = "nan_input" in case.tags

    # ---- exceptions -------------------------------------------------------
    if run.error is not None:
        if nan_case and spec.nan == "raise":
            ctx.add("nan", case, PASS, f"raised {run.error_text()}", run=run)
        elif may_raise:
            ctx.add("error", case, PASS, f"allowed: {run.error_text()}", run=run)
        elif ref_run is not None and ref_run.error is not None:
            ctx.add("error", case, PASS, "reference raises too", run=run)
        else:
            ctx.add("error", case, FAIL, f"raised {run.error_text()}", run=run)
        return run
    ctx.add("error", case, PASS, run=run)
    outs = run.outputs or []

    # ---- in-place mutation -----------------------------------------------
    if spec.check_mutation:
        if run.mutated:
            ctx.add("mutation", case, FAIL,
                    f"modified input argument(s) {run.mutated} in place", run=run)
        else:
            ctx.add("mutation", case, PASS, run=run)

    # ---- dtype -------------------------------------------------------------
    if spec.out_dtype is not None:
        _check_dtype(ctx, case, run)

    # ---- reference outputs (needed by the finite check) ---------------
    ref_outs: list[np.ndarray] | None = None
    if ref_run is not None and ref_run.error is None:
        ref_outs = ref_run.outputs

    # ---- finite outputs ----------------------------------------------------
    allowed_nonfinite = case_matches(case, spec.allow_nonfinite)
    if spec.finite and case.finite_inputs and not allowed_nonfinite:
        masks = _nonfinite_mask(outs)
        ref_masks = None
        if ref_outs is not None and len(ref_outs) == len(outs):
            # round the reference to each output's dtype: a value beyond the
            # dtype's range is legitimately +-inf in that dtype
            with np.errstate(all="ignore"):
                ref_masks = _nonfinite_mask([_round_like(r, o, case) for r, o in zip(ref_outs, outs)])
        problems = []
        for k, m in enumerate(masks):
            if not m.any():
                continue
            if ref_masks is not None and ref_masks[k].shape == m.shape and not np.any(m & ~ref_masks[k]):
                continue
            idx = np.unravel_index(int(np.argmax(m)), m.shape) if m.ndim else ()
            val = outs[k][idx] if m.ndim else outs[k][()]
            label = f"output {k} " if len(outs) > 1 else ""
            problems.append(f"{label}{int(m.sum())}/{m.size} non-finite (e.g. {_fmt_val(val)})")
        if problems:
            ctx.add("finite", case, FAIL, "finite input gave " + "; ".join(problems), run=run)
        else:
            ctx.add("finite", case, PASS, run=run)

    # ---- NaN policy --------------------------------------------------------
    if nan_case and spec.nan in ("propagate", "raise"):
        if spec.nan == "raise":
            ctx.add("nan", case, FAIL, "NaN input did not raise", run=run)
        else:
            float_outs = [o for o in outs if o.dtype.kind in "fc" and o.size]
            if float_outs:
                if any(np.isnan(o).any() for o in float_outs):
                    ctx.add("nan", case, PASS, run=run)
                else:
                    ctx.add("nan", case, FAIL,
                            "NaN input silently produced an all-finite/non-NaN output", run=run)

    # ---- reference --------------------------------------------------------
    if ref_run is not None:
        if ref_run.error is not None:
            ctx.add("reference", case, FAIL,
                    f"reference raised {ref_run.error_text()} but function returned a value",
                    run=run)
        else:
            _check_reference(ctx, case, run, ref_outs or [])

    # ---- floating-point signals ------------------------------------------
    if spec.fp_signals == "fail" and case.finite_inputs and not allowed_nonfinite:
        ref_nonfinite = False
        if ref_outs is not None:
            with np.errstate(all="ignore"):
                ref_nonfinite = any(
                    m.any() for m in _nonfinite_mask(
                        [_round_like(r, o, case) for r, o in zip(ref_outs, outs)]))
        if run.fp_events and not ref_nonfinite:
            ctx.add("fp_signals", case, FAIL,
                    f"raised FP signal(s) on finite input: {', '.join(run.fp_events)}", run=run)
        else:
            ctx.add("fp_signals", case, PASS, run=run)

    # ---- postconditions ------------------------------------------------------
    if spec.postconditions and case.finite_inputs:
        for post in spec.postconditions:
            pname = f"post:{getattr(post, '__name__', 'postcondition')}"
            try:
                with np.errstate(all="ignore"):
                    verdict = post(case.args, list(outs))
            except Exception as exc:  # noqa: BLE001
                ctx.add(pname, case, FAIL, f"postcondition raised {type(exc).__name__}: {exc}",
                        run=run)
                continue
            if verdict is None or verdict is True:
                ctx.add(pname, case, PASS, run=run)
            else:
                msg = verdict if isinstance(verdict, str) else "postcondition violated"
                ctx.add(pname, case, FAIL, msg, run=run)

    # ---- bounds -------------------------------------------------------------
    if spec.bounds is not None:
        lo, hi = spec.bounds
        bad = []
        for k, o in enumerate(outs):
            if o.dtype.kind not in "fiu" or not o.size:
                continue
            slack = default_tolerance(o.dtype)[0] if o.dtype.kind == "f" else 0.0
            with np.errstate(all="ignore"):
                v = o.astype(np.float64)
                out_of = np.isfinite(v) & ((v < lo - slack * max(1.0, abs(lo)))
                                           | (v > hi + slack * max(1.0, abs(hi))))
            if out_of.any():
                i = np.unravel_index(int(np.argmax(out_of)), out_of.shape)
                bad.append(f"{int(out_of.sum())} value(s) outside [{lo:g}, {hi:g}], "
                           f"e.g. {float(v[i])!r}")
        ctx.add("bounds", case, FAIL if bad else PASS, "; ".join(bad), run=run)
    return run


def _check_dtype(ctx: _Ctx, case: Case, run: Run) -> None:
    spec = ctx.spec
    outs = run.outputs or []
    names = run.out_dtypes if len(run.out_dtypes) == len(outs) else [o.dtype.name for o in outs]
    case_name = case.dtype
    problems = []
    if spec.out_dtype == "preserve":
        if not is_float_name(case_name):
            return
        for k, (o, weak, got) in enumerate(zip(outs, run.weak, names)):
            if weak or o.dtype.kind != "f":
                continue
            if got != case_name:
                problems.append(f"output {k}: {got} (expected {case_name})" if len(outs) > 1
                                else f"got {got}, expected {case_name}")
    else:
        expected = spec.out_dtype
        exp_list = list(expected) if isinstance(expected, (list, tuple)) else [expected] * len(outs)
        for k, (e, got) in enumerate(zip(exp_list, names)):
            if e is None:
                continue
            want = case_name if e == "preserve" else dtype_name(e)
            if got != want:
                problems.append(f"output {k}: got {got}, expected {want}")
    ctx.add("dtype", case, FAIL if problems else PASS, "; ".join(problems), run=run)


def _check_reference(ctx: _Ctx, case: Case, run: Run, ref_outs: list[np.ndarray]) -> None:
    spec = ctx.spec
    outs = run.outputs or []
    if len(ref_outs) != len(outs):
        ctx.add("reference", case, FAIL,
                f"function returned {len(outs)} output(s), reference {len(ref_outs)}", run=run)
        return
    worst: Comparison | None = None
    failed: list[str] = []
    agg = Comparison(True, max_abs_err=0.0, max_rel_err=0.0, max_ulp=None)
    for k, (o, r, weak) in enumerate(zip(outs, ref_outs, run.weak)):
        c = compare(o, r, rtol=spec.rtol, atol=spec.atol, tol_dtype=_tol_dtype(o, weak, case),
                    label="reference")
        for attr in ("max_abs_err", "max_rel_err", "max_ulp"):
            v = getattr(c, attr)
            if v is not None:
                cur = getattr(agg, attr)
                setattr(agg, attr, v if cur is None else max(cur, v))
        if not c.ok:
            failed.append((f"output {k}: " if len(outs) > 1 else "") + c.message)
            worst = worst or c
    agg.example = worst.example if worst else None
    agg.ok = not failed
    ctx.add("reference", case, PASS if agg.ok else FAIL, "; ".join(failed), run=run, cmp=agg)


def _compare_outputs(got: list[np.ndarray], want: list[np.ndarray], case: Case,
                     spec: "Spec", rtol_factor: float, label: str) -> Comparison:
    if len(got) != len(want):
        return Comparison(False, f"{len(got)} output(s) vs {len(want)}")
    agg = Comparison(True, max_abs_err=0.0, max_rel_err=0.0)
    msgs = []
    for k, (g, w) in enumerate(zip(got, want)):
        c = compare(g, w, rtol=spec.rtol, atol=spec.atol, tol_dtype=_tol_dtype(g, False, case),
                    rtol_factor=rtol_factor, label=label)
        agg.max_abs_err = max(agg.max_abs_err or 0.0, c.max_abs_err or 0.0)
        agg.max_rel_err = max(agg.max_rel_err or 0.0, c.max_rel_err or 0.0)
        if c.max_ulp is not None:
            agg.max_ulp = c.max_ulp if agg.max_ulp is None else max(agg.max_ulp, c.max_ulp)
        if not c.ok:
            msgs.append((f"output {k}: " if len(got) > 1 else "") + c.message)
            agg.example = agg.example or c.example
    agg.ok = not msgs
    agg.message = "; ".join(msgs)
    return agg


def _all_finite(arrs: Sequence[np.ndarray]) -> bool:
    return all(a.dtype.kind not in "fc" or bool(np.all(np.isfinite(a))) for a in arrs)


def _check_batch(ctx: _Ctx, fn: Callable[..., Any], case: Case, run: Run) -> None:
    spec = ctx.spec
    axis = spec.batch_axis
    if any(a.ndim == 0 for a in case.args) or not run.ok:
        return
    variants = [case.args, tuple(a[..., ::-1].copy() for a in case.args),
                tuple(np.roll(a, 1, axis=-1) for a in case.args)]
    singles = [run] + [call(fn, v, spec.kwargs) for v in variants[1:]]
    if any(not s.ok for s in singles):
        bad = next(s for s in singles if not s.ok)
        ctx.add("batch", case, FAIL, f"unbatched call raised {bad.error_text()}", run=run)
        return
    try:
        stacked = tuple(np.stack([v[i] for v in variants], axis=axis)
                        for i in range(len(case.args)))
    except ValueError as exc:  # pragma: no cover - defensive
        ctx.add("batch", case, SKIP, f"cannot stack inputs: {exc}")
        return
    b = call(fn, stacked, spec.kwargs)
    if not b.ok:
        ctx.add("batch", case, FAIL, f"batched call raised {b.error_text()}", run=b)
        return
    n_out = len(run.outputs or [])
    try:
        want = [np.stack([s.outputs[k] for s in singles], axis=axis)  # type: ignore[index]
                for k in range(n_out)]
    except ValueError as exc:
        ctx.add("batch", case, FAIL, f"per-item outputs cannot be stacked: {exc}", run=b)
        return
    c = _compare_outputs(b.outputs or [], want, case, spec, 4.0, "unbatched")
    ctx.add("batch", case, PASS if c.ok else FAIL,
            "" if c.ok else "f(stack(xs)) != stack(f(x)): " + c.message, run=b, cmp=c)


def _check_monotonic(ctx: _Ctx, fn: Callable[..., Any], case: Case) -> None:
    spec = ctx.spec
    i = spec.monotonic_arg
    if i >= len(case.args):
        return
    x = case.args[i]
    if x.dtype.kind != "f" or x.size < 2 or not np.all(np.isfinite(x)):
        return
    xs = np.sort(x.ravel())
    args = list(case.args)
    args[i] = xs
    if any(a.shape != xs.shape for j, a in enumerate(args) if j != i and a.ndim):
        return
    r = call(fn, args, spec.kwargs)
    if not r.ok or len(r.outputs or []) != 1:
        return
    y = r.outputs[0]  # type: ignore[index]
    if y.shape != xs.shape or y.dtype.kind != "f":
        return
    with np.errstate(all="ignore"):
        y64 = y.astype(np.float64)
        if spec.monotonic == "decreasing":
            y64 = -y64
        rt = _tol_value(spec.rtol, y.dtype, default_tolerance(y.dtype)[0])
        floor = default_tolerance(y.dtype)[1]
        prev, nxt = y64[:-1], y64[1:]
        ok_pair = np.isnan(prev) | np.isnan(nxt) | (nxt >= prev - rt * np.abs(prev) - floor)
    if ok_pair.all():
        ctx.add("monotonic", case, PASS, run=r)
    else:
        k = int(np.argmin(ok_pair))
        ctx.add("monotonic", case, FAIL,
                f"not {spec.monotonic}: f({xs[k]!r})={_fmt_val(y[k])} then "
                f"f({xs[k + 1]!r})={_fmt_val(y[k + 1])}", run=r)


def _check_properties(ctx: _Ctx, fn: Callable[..., Any], case: Case, run: Run) -> None:
    spec = ctx.spec
    if not run.ok or not _all_finite(run.outputs or []) or not case.finite_inputs:
        return
    for prop in spec.properties:
        if not case_matches(case, prop.cases):
            continue
        targs = prop.transform(case.args, spec.inputs)
        if targs is None:
            continue
        expected = prop.expect(list(run.outputs or []), case.args)
        if expected is None:
            continue
        r = call(fn, targs, spec.kwargs)
        name = f"prop:{prop.name}"
        if not r.ok:
            ctx.add(name, case, FAIL, f"transformed input raised {r.error_text()}", run=r)
            continue
        c = _compare_outputs(r.outputs or [], expected, case, spec, prop.rtol_factor,
                             "expected")
        ctx.add(name, case, PASS if c.ok else FAIL,
                "" if c.ok else f"{prop.description}: {c.message}", run=r, cmp=c)


def _isolation_pool(spec: "Spec", done: Sequence[tuple[Case, Run]]) -> dict[str, list[tuple[str, tuple[np.ndarray, ...]]]]:
    """Per dtype: the inputs of every case that ran, to place next to other cases."""
    pool: dict[str, list[tuple[str, tuple[np.ndarray, ...]]]] = {}
    for case, run in done:
        if not run.ok or not _isolatable(case) or case.size > spec.n * 64:
            continue
        pool.setdefault(case.dtype, []).append((case.name, case.args))
    return pool


def _isolatable(case: Case) -> bool:
    if not case.args or any(a.ndim != 1 for a in case.args):
        return False
    return len({a.shape[0] for a in case.args}) == 1 and case.args[0].shape[0] >= 1


def _check_isolation(ctx: _Ctx, fn: Callable[..., Any], case: Case, run: Run,
                     pool: Mapping[str, list[tuple[str, tuple[np.ndarray, ...]]]]) -> None:
    """An elementwise result must not depend on the other elements of the array.

    Each element is recomputed (a) alone, as a length-1 array, and (b) inside a
    larger array that also holds the inputs of every other case of the same
    dtype (NaN, infinities, huge, tiny and subnormal values included). Both
    must agree with the original call. A function that shares work across the
    batch, such as one convergence loop for every element (``jax-ml/jax#41440``),
    or that normalizes by a batch statistic, fails here.
    """
    spec = ctx.spec
    if not run.ok or not _isolatable(case) or case.size > spec.n * 64:
        return
    outs = run.outputs or []
    n = case.args[0].shape[0]
    if not outs or any(o.shape != (n,) for o in outs):
        ctx.add("isolation", case, FAIL,
                "elementwise=True but the output shape differs from the input shape", run=run)
        return
    neighbours = [args for name, args in pool.get(case.dtype, []) if name != case.name]
    if spec.nan == "raise":
        neighbours = [a for a in neighbours if not any(np.isnan(x).any() for x in a if x.dtype.kind == "f")]
    problems: list[str] = []
    worst: Comparison | None = None
    if neighbours:
        others = [np.concatenate([nb[i] for nb in neighbours]) for i in range(len(case.args))]
        m = others[0].shape[0]
        for label, before in (("after", 0), ("before", m)):
            mixed = tuple(np.concatenate([a, o]) if before == 0 else np.concatenate([o, a])
                          for a, o in zip(case.args, others))
            r = call(fn, mixed, spec.kwargs)
            if not r.ok:
                problems.append(f"raised {r.error_text()} when other cases' inputs were placed {label} it")
                continue
            got = [o[before:before + n] if o.ndim == 1 and o.shape[0] == n + m else o for o in r.outputs or []]
            c = _compare_outputs(got, outs, case, spec, 4.0, "alone")
            if not c.ok:
                problems.append(f"with other cases' inputs placed {label} it: {c.message}")
                worst = worst or c
    k_max = min(n, 16)
    singles = []
    for k in range(k_max):
        r = call(fn, tuple(a[k:k + 1] for a in case.args), spec.kwargs)
        if not r.ok:
            problems.append(f"element {k} computed alone raised {r.error_text()}")
            break
        singles.append(r.outputs or [])
    if len(singles) == k_max and singles:
        try:
            got = [np.concatenate([s[j] for s in singles]) for j in range(len(outs))]
        except ValueError:
            got = None
        if got is not None:
            c = _compare_outputs(got, [o[:k_max] for o in outs], case, spec, 4.0, "in the full array")
            if not c.ok:
                problems.append(f"computed one element at a time: {c.message}")
                worst = worst or c
    if problems:
        ctx.add("isolation", case, FAIL, "result depends on the other elements: " + "; ".join(problems),
                run=run, cmp=worst)
    else:
        ctx.add("isolation", case, PASS, run=run)


def _grad_dtnames(spec: "Spec", case: Case) -> list[str]:
    return [dtype_name(s.dtype) if s.dtype is not None else case.dtype for s in spec.inputs] \
        or [case.dtype] * len(case.args)


def _check_grad(ctx: _Ctx, fn: Callable[..., Any], grad_fn: Callable[..., Any] | None,
                case: Case, run: Run) -> None:
    """Gradients of ``sum(f)`` against a float64 reference (see :mod:`numguard.gradcheck`)."""
    from .gradcheck import autodiff, finite_difference_grad, grad_tolerance, unwrap  # noqa: PLC0415

    spec = ctx.spec
    if not run.ok or not case.finite_inputs:
        return
    outs = run.outputs or []
    if len(outs) != 1 or outs[0].dtype.kind != "f":
        return
    if not np.all(np.isfinite(outs[0])) and not spec.elementwise:
        return
    idx = [i for i, a in enumerate(case.args) if a.dtype.kind == "f"]
    if not idx:
        return
    dtnames = _grad_dtnames(spec, case)
    ad = None
    if spec.grad is True:
        backend = fn.backend if isinstance(fn, BackendCall) else None
        if backend is None:
            return
        try:
            ad = autodiff(backend, unwrap(fn), case.args, dtnames, idx, dict(spec.kwargs))
        except Exception as exc:  # noqa: BLE001 - reported as a failure
            ctx.add("grad", case, FAIL, f"automatic differentiation raised {type(exc).__name__}: "
                    f"{str(exc).strip().splitlines()[0] if str(exc).strip() else ''}".rstrip(": "), run=run)
            return
        grads = ad.grads
    else:
        assert grad_fn is not None
        if isinstance(grad_fn, BackendCall):
            grad_fn.set_case(dtnames)
        r = call(grad_fn, case.args, spec.kwargs)
        if not r.ok:
            ctx.add("grad", case, FAIL, f"gradient function raised {r.error_text()}", run=r)
            return
        grads = list(r.outputs or [])
        if len(grads) != len(idx):
            ctx.add("grad", case, FAIL, f"gradient function returned {len(grads)} array(s) for "
                    f"{len(idx)} floating argument(s)", run=r)
            return

    # ---- primal and forward-mode consistency ---------------------------------
    if ad is not None and ad.jvp_primal is not None:
        c = _compare_outputs([np.asarray(ad.jvp_primal)], outs, case, spec, 1.0, "f(x)")
        ctx.add("grad:primal", case, PASS if c.ok else FAIL,
                "" if c.ok else f"the primal value returned by jvp differs from f(x): {c.message}",
                run=run, cmp=c)
        if spec.elementwise and len(idx) == 1 and ad.jvp_tangent is not None and grads[0] is not None:
            tol = grad_tolerance(dtnames[idx[0]], spec.grad_rtol)
            c = compare(np.asarray(ad.jvp_tangent), np.asarray(grads[0], dtype=np.float64),
                        rtol=tol, atol=0.0, label="reverse mode")
            ctx.add("grad:forward", case, PASS if c.ok else FAIL,
                    "" if c.ok else f"forward-mode derivative differs from reverse mode: {c.message}",
                    run=run, cmp=c)

    # ---- the float64 reference gradient -------------------------------------
    args64 = [a.astype(np.float64) if a.dtype.kind == "f" else a for a in case.args]
    expected: list[np.ndarray]
    if spec.grad_reference is not None:
        r = call(spec.grad_reference, args64, spec.ref_kwargs, check_mutation=False)
        if not r.ok:
            ctx.add("grad", case, FAIL, f"grad_reference raised {r.error_text()}", run=run)
            return
        expected = [np.asarray(e, dtype=np.float64) for e in (r.outputs or [])]
        uncert = [np.zeros_like(e) for e in expected]
        if len(expected) != len(idx):
            ctx.add("grad", case, FAIL, f"grad_reference returned {len(expected)} array(s) for "
                    f"{len(idx)} floating argument(s)", run=run)
            return
    else:
        if spec.reference is not None:
            valfn, vkw = spec.reference, dict(spec.ref_kwargs)
        else:
            valfn, vkw = fn, dict(spec.kwargs)
            if isinstance(fn, BackendCall):
                fn.set_case(["float64" if a.dtype.kind == "f" else dtnames[i]
                             for i, a in enumerate(case.args)])
                probe = call(fn, args64, vkw, check_mutation=False)
                fn.set_case(dtnames)
                if not probe.ok or any(np.asarray(o).dtype != np.float64 for o in probe.outputs or []):
                    ctx.add("grad", case, SKIP, "no float64 evaluation for finite differences: give "
                            "reference= or grad_reference= (JAX needs JAX_ENABLE_X64=1 for float64)")
                    return
        expected, uncert = [], []
        try:
            for i in idx:
                spec_i = spec.inputs[i] if i < len(spec.inputs) else None
                est, unc = finite_difference_grad(valfn, args64, i, spec_i, vkw,
                                                  elementwise=spec.elementwise)
                expected.append(est)
                uncert.append(unc)
        except Exception as exc:  # noqa: BLE001
            ctx.add("grad", case, SKIP, f"finite differences failed: {type(exc).__name__}: {exc}")
            return
        if isinstance(fn, BackendCall):
            fn.set_case(dtnames)

    # ---- compare --------------------------------------------------------------
    problems: list[str] = []
    worst: Comparison | None = None
    checked = 0
    for pos, i in enumerate(idx):
        g = grads[pos]
        e = expected[pos]
        if g is None:
            if np.any(np.isfinite(e) & (e != 0)):
                problems.append(f"argument {spec.arg_names[i] if i < len(spec.arg_names) else i}: "
                                "no gradient (unused), but the function depends on it")
            continue
        g = np.asarray(g)
        if g.shape != case.args[i].shape:
            problems.append(f"gradient of argument {i} has shape {g.shape}, input {case.args[i].shape}")
            continue
        mask = np.isfinite(e)
        if spec.elementwise and outs[0].shape == g.shape:
            mask &= np.isfinite(outs[0])
        if not mask.any():
            continue
        checked += int(mask.sum())
        tol = grad_tolerance(dtnames[i], spec.grad_rtol)
        gm = g[mask].astype(np.float64)
        em = e[mask]
        um = np.asarray(uncert[pos])[mask] if np.shape(uncert[pos]) == e.shape else np.zeros_like(em)
        scale = float(np.max(np.abs(em)))
        with np.errstate(all="ignore"):
            # a reference beyond the gradient dtype's range is expected to be +-inf there
            e_round = em.astype(g.dtype).astype(np.float64) if g.dtype.kind == "f" else em
            limit = tol * np.abs(em) + tol * 1e-3 * max(1.0, scale) + um
            diff = np.abs(gm - em)
            bad = np.where(np.isinf(e_round), gm != e_round, ~np.isfinite(gm) | (diff > limit))
        if bad.any():
            with np.errstate(all="ignore"):
                rel = np.where(bad, np.where(np.isfinite(diff), diff / np.maximum(np.abs(em), 1e-300),
                                             np.inf), -1.0)
            j = int(np.argmax(rel))
            k = int(np.flatnonzero(mask.ravel())[j])
            xval = case.args[i].ravel()[k]
            name = spec.arg_names[i] if i < len(spec.arg_names) else f"argument {i}"
            problems.append(f"d/d{name} at {name}={_fmt_val(xval)}: got {_fmt_val(gm[j])}, "
                            f"expected {_fmt_val(em[j])} ({int(bad.sum())}/{gm.size} off)")
            c = Comparison(False, problems[-1], float(np.nanmax(np.where(np.isfinite(diff), diff, np.nan)))
                           if np.isfinite(diff).any() else float("inf"), float(np.max(rel)), None,
                           int(bad.sum()), problems[-1])
            worst = worst or c
    if not checked and not problems:
        return
    source = ("grad_reference" if spec.grad_reference is not None else
              "finite differences of the reference" if spec.reference is not None else
              "finite differences in float64")
    ctx.add("grad", case, FAIL if problems else PASS,
            f"gradient differs from {source}: " + "; ".join(problems) if problems else "",
            run=run, cmp=worst)


def run_checks(fn: Callable[..., Any], spec: "Spec", cases: Sequence[Case],
               name: str | None = None) -> Report:
    """Run every applicable check of ``spec`` on ``cases`` and build a :class:`Report`."""
    t0 = time.perf_counter()
    report = Report(function=name or getattr(fn, "__qualname__", repr(fn)))
    report.dtypes = list(dict.fromkeys(c.dtype for c in cases))
    report.n_cases = len(cases)
    if spec.reference is not None:
        report.reference = getattr(spec.reference, "__qualname__",
                                   getattr(spec.reference, "__name__", "reference"))
        if isinstance(spec.reference, HighPrecReference):
            report.reference += f" [{spec.reference.precision}]"
    ctx = _Ctx(spec, report)
    grad_dtypes = report.dtypes if spec.grad not in (None, False) else []
    fn = wrap(fn, spec.backend, report.dtypes)
    grad_fn = None
    if callable(spec.grad):
        grad_fn = wrap(spec.grad, spec.backend, grad_dtypes)
    if spec.backend != "numpy":
        report.backend = spec.backend
    done: list[tuple[Case, Run]] = []
    for case in cases:
        if isinstance(fn, BackendCall):
            fn.set_case(_grad_dtnames(spec, case))
        run = _check_case(ctx, fn, case)
        done.append((case, run))
        if spec.batched and case.regular and case.size >= 1:
            _check_batch(ctx, fn, case, run)
        if spec.monotonic and case.finite_inputs:
            _check_monotonic(ctx, fn, case)
        if spec.properties:
            _check_properties(ctx, fn, case, run)
        if spec.grad not in (None, False):
            _check_grad(ctx, fn, grad_fn, case, run)
    if spec.elementwise:
        pool = _isolation_pool(spec, done)
        for case, run in done:
            if isinstance(fn, BackendCall):
                fn.set_case(_grad_dtnames(spec, case))
            _check_isolation(ctx, fn, case, run, pool)
    report.duration_s = time.perf_counter() - t0
    return report
