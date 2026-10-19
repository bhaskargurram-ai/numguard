"""Scan mode: run numguard over every public numerical function of a module.

``numguard scan MODULE`` imports the module, finds its public callables and,
for each one, probes a few calls to work out what it takes:

* the **arity** (``ufunc.nin``, else the number of required positional
  parameters; functions needing more than three are skipped);
* a **domain** where it returns mostly finite values: ``any``, then
  ``positive``, then ``unit_interval``;
* whether it is **elementwise**: the output has the input's shape and an
  element computed alone matches the same element computed in the array.
  A NumPy ufunc, or a function whose reference is elementwise, counts as
  elementwise even if the probe disagrees, since that disagreement may be
  the bug.

Each function is then checked with the usual edge-case catalogue. With
``--ref MODULE``, a function of the same name in the reference module (for
example ``scipy.special`` for ``jax.scipy.special``) is used as the
reference; without one, the ``finite`` check is off, because nothing says
whether an inf on a huge input is right.

Scan results are *candidates*: each finding needs triage (is it documented
behaviour? does it reproduce on the latest release?) before it is reported
anywhere.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import multiprocessing
import queue as _queue
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .api import check
from .backends import BackendError, wrap
from .report import Report

__all__ = ["ScanEntry", "ScanResult", "ScanTimeout", "scan_module", "public_functions", "probe"]

#: Name fragments of functions that are not pure numerical functions.
_SKIP_NAME = re.compile(
    r"(^|_)(seed|random|rand|randn|load|save|savez|read|write|print|show|plot|open|close|set|get|"
    r"config|errstate|geterr|seterr|info|test|bench|register|deprecat|vectorize|frompyfunc|"
    r"jit|grad|vmap|pmap|device|block|array|asarray|empty|zeros|ones|full|eye|identity|arange|"
    r"linspace|logspace|geomspace|meshgrid|mgrid|ogrid|copy|astype|dtype|shape|ndim|size)($|_)"
)

_DOMAINS = (("any", np.linspace(-3.0, 3.0, 8) + 0.0625),
            ("positive", np.linspace(0.125, 3.0, 8)),
            ("unit_interval", np.linspace(0.05, 0.95, 8)))


@dataclass
class ScanEntry:
    """What the scan found out about one function, and its report."""

    name: str
    status: str  # "checked", "skipped" or "error"
    reason: str = ""
    arity: int = 0
    domain: str = ""
    elementwise: bool = False
    reference: str = ""
    report: Report | None = None
    seconds: float = 0.0

    @property
    def failed_checks(self) -> list[str]:
        return list(self.report.failed_checks()) if self.report is not None else []

    def first_failure(self) -> str:
        if self.report is None or not self.report.failures:
            return ""
        r = self.report.failures[0]
        return f"{r.check} [{r.case}, {r.dtype}]: {r.message}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "status": self.status, "reason": self.reason,
                             "arity": self.arity, "domain": self.domain,
                             "elementwise": self.elementwise, "reference": self.reference,
                             "seconds": round(self.seconds, 3)}
        if self.report is not None:
            d["failed_checks"] = self.failed_checks
            d["failures"] = [{"check": r.check, "case": r.case, "dtype": r.dtype, "message": r.message}
                             for r in self.report.failures[:20]]
            d["n_results"] = len(self.report.results)
        return d


@dataclass
class ScanResult:
    module: str
    backend: str
    reference_module: str | None
    entries: list[ScanEntry] = field(default_factory=list)

    @property
    def checked(self) -> list[ScanEntry]:
        return [e for e in self.entries if e.status == "checked"]

    @property
    def flagged(self) -> list[ScanEntry]:
        return [e for e in self.checked if e.failed_checks]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "numguard/scan", "version": 1, "module": self.module,
                "backend": self.backend, "reference_module": self.reference_module,
                "n_functions": len(self.entries), "n_checked": len(self.checked),
                "n_flagged": len(self.flagged), "entries": [e.to_dict() for e in self.entries]}

    def summary(self, max_rows: int = 50) -> str:
        lines = [f"numguard scan {self.module} (backend {self.backend}"
                 + (f", reference {self.reference_module}" if self.reference_module else "") + ")",
                 f"{len(self.entries)} public callables, {len(self.checked)} checked, "
                 f"{len(self.flagged)} with failing checks (candidates: triage before reporting)", ""]
        rows = sorted(self.checked, key=lambda e: (not e.failed_checks, e.name))
        for e in rows[:max_rows]:
            mark = "FLAG" if e.failed_checks else "ok  "
            extra = f"  {e.first_failure()[:160]}" if e.failed_checks else ""
            ref = f" ref={e.reference}" if e.reference else ""
            lines.append(f"  {mark} {e.name}({e.arity}) domain={e.domain}"
                         f"{' elementwise' if e.elementwise else ''}{ref}"
                         f"  failed: {', '.join(e.failed_checks) or '-'}{extra}")
        skipped = [e for e in self.entries if e.status != "checked"]
        if skipped:
            lines += ["", f"skipped {len(skipped)}: "
                      + ", ".join(f"{e.name} ({e.reason})" for e in skipped[:40])
                      + (" ..." if len(skipped) > 40 else "")]
        return "\n".join(lines)


def public_functions(module: Any) -> list[tuple[str, Callable[..., Any]]]:
    """Public callables of ``module`` (``__all__`` if defined), sorted by name."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    out = []
    for n in sorted(set(names)):
        try:
            obj = getattr(module, n)
        except AttributeError:
            continue
        if inspect.isclass(obj) or inspect.ismodule(obj) or not callable(obj):
            continue
        out.append((n, obj))
    return out


def _arity(fn: Callable[..., Any]) -> int | None:
    nin = getattr(fn, "nin", None)
    if isinstance(nin, int):
        return nin
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    required = [p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()) and not required:
        return None
    return len(required)


def _infer_arity(fn: Callable[..., Any], ref: Callable[..., Any] | None, backend: str) -> int | None:
    """Arity of a function whose signature can't be read (C builtins, e.g. ``torch.special``).

    The reference function's arity is tried first, then 1, 2 and 3: the
    first number of arguments that does not raise ``TypeError`` wins.
    """
    first = _arity(ref) if ref is not None else None
    tries = ([first] if first and 1 <= first <= 3 else []) + [k for k in (1, 2, 3) if k != first]
    call = wrap(fn, backend)
    base = np.linspace(0.1, 0.9, 8)
    for k in tries:
        try:
            with np.errstate(all="ignore"):
                call(*(base.copy() for _ in range(k)))
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - right arity, other trouble; probe() reports it
            return k
        return k
    return None


def _to_np(out: Any) -> np.ndarray | None:
    if isinstance(out, (tuple, list, dict)):
        return None
    try:
        arr = np.asarray(out)
    except Exception:  # noqa: BLE001
        return None
    return arr if arr.dtype.kind == "f" else None


def probe(fn: Callable[..., Any], arity: int, backend: str = "numpy") -> tuple[str, bool] | str:
    """Find a domain and whether ``fn`` is elementwise; a string is a reason to skip."""
    call = wrap(fn, backend)
    last = "no domain gives mostly finite output"
    for name, base in _DOMAINS:
        args = tuple(np.roll(base, k).astype(np.float64) for k in range(arity))
        try:
            with np.errstate(all="ignore"):
                out = _to_np(call(*args))
        except Exception as exc:  # noqa: BLE001 - probing
            last = f"raises on {name} inputs: {type(exc).__name__}"
            continue
        if out is None:
            return "output is not a single floating array"
        if out.size == 0 or np.mean(np.isfinite(out)) < 0.75:
            continue
        elementwise = out.shape == base.shape
        if elementwise:
            for k in (0, 3, 7):
                try:
                    with np.errstate(all="ignore"):
                        one = _to_np(call(*(a[k:k + 1] for a in args)))
                except Exception:  # noqa: BLE001
                    one = None
                if one is None or one.shape != (1,) or not np.allclose(one[0], out[k], rtol=1e-6,
                                                                        atol=1e-12, equal_nan=True):
                    elementwise = False
                    break
        return name, elementwise
    return last


class ScanTimeout(BaseException):
    """A function took longer than the scan's per-function limit.

    A ``BaseException`` (like ``KeyboardInterrupt``) so that the ``except
    Exception`` handlers that record a function's errors as results do not
    swallow it.
    """


@contextlib.contextmanager
def _time_limit(seconds: float):  # noqa: ANN202
    """Raise :class:`ScanTimeout` after ``seconds`` (POSIX main thread only; elsewhere no limit)."""
    usable = (seconds and seconds > 0 and hasattr(signal, "setitimer")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def handler(signum, frame):  # noqa: ANN001, ARG001
        raise ScanTimeout(f"took longer than {seconds:g} s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def scan_module(module_name: str, *, reference_module: str | None = None, backend: str = "numpy",
                dtypes: Sequence[str] = ("float32", "float64"), include: Sequence[str] = (),
                exclude: Sequence[str] = (), budget_s: float = 30.0, large_n: int = 4096,
                timeout_s: float = 300.0, isolate: bool = False,
                progress: Callable[[str], None] | None = None) -> ScanResult:
    """Check every public numerical function of ``module_name``.

    ``include``/``exclude`` are regular expressions on function names.
    ``budget_s`` bounds the time spent on one function: a function whose
    first dtype already took longer is not checked in the other dtypes.
    ``timeout_s`` is a hard limit per function (probe and checks), enforced
    with a timer signal on POSIX systems; a function that exceeds it is
    recorded as ``error`` and the scan moves on. Libraries that compile per
    input shape, such as JAX, can need minutes per function.

    A timer cannot interrupt native code that never returns to Python (a C
    loop over a degree of 1e30, say). With ``isolate=True`` each function
    is checked in its own worker process, which is killed when it overruns
    ``timeout_s`` (plus a grace period for imports); a worker that crashes
    (a segfault, an abort) is recorded as an error too. Slower to start,
    but one function can no longer stall or end the scan.
    """
    if isolate:
        return _scan_isolated(module_name, reference_module=reference_module, backend=backend, dtypes=dtypes,
                              include=include, exclude=exclude, budget_s=budget_s, large_n=large_n,
                              timeout_s=timeout_s, progress=progress)
    mod = importlib.import_module(module_name)
    ref_mod = importlib.import_module(reference_module) if reference_module else None
    result = ScanResult(module=module_name, backend=backend, reference_module=reference_module)
    inc = [re.compile(p) for p in include]
    exc = [re.compile(p) for p in exclude]
    for name, fn in public_functions(mod):
        if inc and not any(p.search(name) for p in inc):
            continue
        if any(p.search(name) for p in exc):
            continue
        entry = ScanEntry(name=name, status="skipped")
        result.entries.append(entry)
        if _SKIP_NAME.search(name.lower()):
            entry.reason = "not a numerical function (by name)"
            continue
        ref = getattr(ref_mod, name, None) if ref_mod is not None else None
        ref = ref if callable(ref) else None
        arity = _arity(fn)
        if arity is None:
            try:
                with _time_limit(timeout_s):
                    arity = _infer_arity(fn, ref, backend)
            except (BackendError, ScanTimeout):
                arity = None
        if arity is None or not 1 <= arity <= 3:
            entry.reason = "needs 1-3 positional array arguments"
            continue
        entry.arity = arity
        t0 = time.perf_counter()
        try:
            with _time_limit(timeout_s):
                found = probe(fn, arity, backend)
        except BackendError as err:
            entry.status, entry.reason = "error", str(err)
            continue
        except ScanTimeout as err:
            entry.status, entry.reason = "error", f"probe {err}"
            continue
        if isinstance(found, str):
            entry.reason = found
            continue
        entry.domain, entry.elementwise = found
        if ref is not None:
            entry.reference = f"{reference_module}.{name}"
            # A function whose result depends on its neighbours looks
            # non-elementwise when probed, which is exactly the bug the
            # isolation check is for: trust the reference's shape instead.
            ref_found = probe(ref, arity)
            if not isinstance(ref_found, str) and ref_found[1]:
                entry.elementwise = True
        if isinstance(fn, np.ufunc):
            entry.elementwise = True
        if progress:
            progress(name)
        reports: list[Report] = []
        try:
            with _time_limit(max(1.0, timeout_s - (time.perf_counter() - t0))):
                for i, dt in enumerate(dtypes):
                    if i and time.perf_counter() - t0 > budget_s:
                        break
                    reports.append(check(
                        fn, inputs=[entry.domain] * arity, reference=ref if entry.reference else None,
                        dtypes=[dt], backend=backend, elementwise=entry.elementwise,
                        finite=bool(entry.reference), large_n=large_n, name=f"{module_name}.{name}"))
        except ScanTimeout as err:
            if not reports:
                entry.status, entry.reason = "error", str(err)
                entry.seconds = time.perf_counter() - t0
                continue
        except Exception as err:  # noqa: BLE001 - one bad function must not stop the scan
            entry.status, entry.reason = "error", f"{type(err).__name__}: {err}"[:300]
            continue
        entry.seconds = time.perf_counter() - t0
        entry.status = "checked"
        entry.report = reports[0]
        for extra in reports[1:]:
            entry.report.results.extend(extra.results)
            entry.report.dtypes = list(dict.fromkeys(entry.report.dtypes + extra.dtypes))
    return result


#: Seconds a worker process gets on top of ``timeout_s`` for starting and importing.
WORKER_GRACE_S = 60.0


def _worker(out: Any, module_name: str, name: str, kwargs: dict[str, Any]) -> None:
    try:
        res = scan_module(module_name, include=[f"^{re.escape(name)}$"], **kwargs)
        entry = next((e for e in res.entries if e.name == name), None)
        out.put(entry if entry is not None else ScanEntry(name=name, status="error", reason="not found by worker"))
    except BaseException as err:  # noqa: BLE001 - report everything to the parent
        out.put(ScanEntry(name=name, status="error", reason=f"{type(err).__name__}: {err}"[:300]))


def _scan_isolated(module_name: str, *, timeout_s: float, progress: Callable[[str], None] | None,
                   **kwargs: Any) -> ScanResult:
    mod = importlib.import_module(module_name)
    result = ScanResult(module=module_name, backend=kwargs["backend"],
                        reference_module=kwargs["reference_module"])
    inc = [re.compile(p) for p in kwargs.pop("include")]
    exc = [re.compile(p) for p in kwargs.pop("exclude")]
    ctx = multiprocessing.get_context("spawn")
    limit = timeout_s + WORKER_GRACE_S
    for name, _ in public_functions(mod):
        if (inc and not any(p.search(name) for p in inc)) or any(p.search(name) for p in exc):
            continue
        if _SKIP_NAME.search(name.lower()):
            result.entries.append(ScanEntry(name=name, status="skipped", reason="not a numerical function (by name)"))
            continue
        if progress:
            progress(name)
        out = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(out, module_name, name, dict(kwargs, timeout_s=timeout_s)),
                           daemon=True)
        t0 = time.perf_counter()
        proc.start()
        entry = None
        try:
            while entry is None and time.perf_counter() - t0 < limit:
                try:
                    entry = out.get(timeout=0.5)
                except _queue.Empty:
                    if not proc.is_alive():  # ended without a result: give a late put a moment
                        try:
                            entry = out.get(timeout=1.0)
                        except _queue.Empty:
                            break
        finally:
            if proc.is_alive():
                proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        if entry is None:
            if proc.exitcode not in (None, 0) and time.perf_counter() - t0 < limit:
                reason = f"worker process died (exit code {proc.exitcode})"
            else:
                reason = (f"killed after {limit:g} s: did not return (native code a timer cannot "
                          "interrupt, or a hang)")
            entry = ScanEntry(name=name, status="error", reason=reason, seconds=time.perf_counter() - t0)
        result.entries.append(entry)
    return result
