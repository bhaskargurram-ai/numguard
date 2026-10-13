"""Eval mode: grade implementations of the bundled bug-corpus tasks.

The corpus is a directory of task folders. Each folder contains

``task.md``
    The prompt a coding agent receives (function name, signature, contract).
``buggy.py``
    A plausible naive implementation that passes a happy-path test.
``reference.py``
    A correct implementation.
``spec.py``
    The numguard spec: ``FUNCTION`` (name of the function to grade),
    ``options(reference) -> dict`` (keyword arguments for :func:`numguard.check`)
    and optionally ``happy_path(fn)`` (a simple example-based test).

The harness is agent-agnostic: it only grades files. Point it at a directory
containing ``<task>/solution.py`` for each task (written by any coding agent or
by hand) with ``numguard eval --candidate-dir DIR``, or grade the bundled
versions with ``--baseline buggy`` / ``--baseline reference``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

from .api import check
from .report import Report, format_table

__all__ = [
    "Task",
    "TaskResult",
    "EvalResult",
    "corpus_dir",
    "list_tasks",
    "load_task",
    "grade_task",
    "run_eval",
    "BASELINES",
]

BASELINES = ("buggy", "reference")
_MODULE_COUNTER = [0]


def corpus_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Locate the corpus directory.

    Order: explicit ``override``, the ``NUMGUARD_CORPUS`` environment variable,
    the installed ``numguard._corpus`` package data, then ``corpus/`` at the
    root of a source checkout.
    """
    if override:
        return Path(override)
    env = os.environ.get("NUMGUARD_CORPUS")
    if env:
        return Path(env)
    spec = importlib.util.find_spec("numguard._corpus")
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "corpus"
        if (cand / "logsumexp" / "spec.py").is_file():
            return cand
    raise FileNotFoundError("numguard corpus not found; set NUMGUARD_CORPUS or pass --corpus-dir")


def _load_module(path: Path, label: str) -> ModuleType:
    _MODULE_COUNTER[0] += 1
    name = f"_numguard_corpus_{label}_{_MODULE_COUNTER[0]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass
class Task:
    """One corpus task folder."""

    name: str
    path: Path
    function: str
    spec_module: ModuleType

    @property
    def prompt(self) -> str:
        return (self.path / "task.md").read_text(encoding="utf-8")

    def title(self) -> str:
        first = self.prompt.strip().splitlines()[0]
        return first.lstrip("# ").removeprefix("Task:").strip()

    def options(self, reference: Callable[..., Any] | None) -> dict[str, Any]:
        return dict(self.spec_module.options(reference))

    def load_impl(self, path: Path) -> Callable[..., Any]:
        mod = _load_module(path, f"{self.name}_{path.stem}")
        fn = getattr(mod, self.function, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"{path} does not define a function named {self.function!r}")
        return fn

    def reference(self) -> Callable[..., Any]:
        return self.load_impl(self.path / "reference.py")

    def happy_path(self, fn: Callable[..., Any]) -> bool | None:
        test = getattr(self.spec_module, "happy_path", None)
        if test is None:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test(fn)
            return True
        except Exception:  # noqa: BLE001
            return False


def list_tasks(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Names of the task folders in the corpus (sorted)."""
    base = corpus_dir(root)
    return sorted(p.name for p in base.iterdir() if (p / "spec.py").is_file())


def load_task(name: str, root: str | os.PathLike[str] | None = None) -> Task:
    """Load a task's spec module."""
    path = corpus_dir(root) / name
    if not (path / "spec.py").is_file():
        raise FileNotFoundError(f"no corpus task named {name!r} in {path.parent}")
    mod = _load_module(path / "spec.py", f"{name}_spec")
    fn = getattr(mod, "FUNCTION", None)
    if not isinstance(fn, str):
        raise ValueError(f"{path / 'spec.py'} must define FUNCTION = '<name>'")
    return Task(name=name, path=path, function=fn, spec_module=mod)


@dataclass
class TaskResult:
    """Grade of one implementation of one task."""

    task: str
    status: str  # "pass" | "fail" | "missing" | "error"
    happy_path: bool | None = None
    failed_checks: list[str] = field(default_factory=list)
    failed_cases: list[str] = field(default_factory=list)
    n_failures: int = 0
    n_results: int = 0
    error: str | None = None
    report: Report | None = None
    duration_s: float = 0.0

    def to_dict(self, include_report: bool = False) -> dict[str, Any]:
        d = {
            "task": self.task,
            "status": self.status,
            "happy_path": self.happy_path,
            "failed_checks": self.failed_checks,
            "failed_cases": self.failed_cases,
            "n_failures": self.n_failures,
            "n_results": self.n_results,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }
        if include_report and self.report is not None:
            d["report"] = self.report.to_dict()
        return d


@dataclass
class EvalResult:
    """Scoreboard for one candidate (a directory or a bundled baseline)."""

    candidate: str
    results: list[TaskResult]

    @property
    def n_pass(self) -> int:
        return sum(r.status == "pass" for r in self.results)

    @property
    def n_tasks(self) -> int:
        return len(self.results)

    @property
    def n_happy(self) -> int:
        return sum(bool(r.happy_path) for r in self.results)

    def scoreboard(self, max_cases: int = 3) -> str:
        rows = []
        for r in self.results:
            hp = {True: "pass", False: "FAIL", None: "-"}[r.happy_path]
            ng = {"pass": "pass", "fail": "FAIL", "missing": "missing", "error": "ERROR"}[r.status]
            detail = ""
            if r.status == "fail":
                cases = r.failed_cases[:max_cases]
                more = len(r.failed_cases) - len(cases)
                detail = ", ".join(cases) + (f" +{more}" if more > 0 else "")
            elif r.error:
                detail = r.error
            rows.append([r.task, hp, ng, ", ".join(r.failed_checks), detail])
        table = format_table(["task", "happy-path test", "numguard", "failed checks",
                              "failing cases"], rows, max_widths=[18, 15, 8, 50, 40])
        footer = (f"{self.candidate}: numguard passed {self.n_pass}/{self.n_tasks} tasks; "
                  f"happy-path tests passed {self.n_happy}/{self.n_tasks}")
        return table + "\n" + footer

    def to_dict(self, include_reports: bool = False) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "n_tasks": self.n_tasks,
            "n_pass": self.n_pass,
            "n_happy_path_pass": self.n_happy,
            "tasks": [r.to_dict(include_reports) for r in self.results],
        }

    def to_json(self, indent: int | None = 2, include_reports: bool = False) -> str:
        return json.dumps(self.to_dict(include_reports), indent=indent)


def _short_error(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}".rstrip(": ")


def grade_task(task: Task, impl_path: Path, **overrides: Any) -> TaskResult:
    """Grade one implementation file against a task's spec."""
    t0 = time.perf_counter()
    if not impl_path.is_file():
        return TaskResult(task.name, "missing", error=f"{impl_path} not found")
    try:
        fn = task.load_impl(impl_path)
    except Exception as exc:  # noqa: BLE001 - candidate code may do anything
        return TaskResult(task.name, "error", error=_short_error(exc),
                          duration_s=time.perf_counter() - t0)
    try:
        opts = task.options(task.reference())
        opts.update({k: v for k, v in overrides.items() if v is not None})
        report = check(fn, name=f"{task.name}:{task.function}", **opts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return TaskResult(task.name, "error", error=_short_error(exc),
                          duration_s=time.perf_counter() - t0)
    hp = task.happy_path(fn)
    return TaskResult(
        task=task.name,
        status="pass" if report.ok else "fail",
        happy_path=hp,
        failed_checks=report.failed_checks(),
        failed_cases=report.failed_cases(),
        n_failures=len(report.failures),
        n_results=len(report.results),
        report=report,
        duration_s=time.perf_counter() - t0,
    )


def run_eval(
    candidate_dir: str | os.PathLike[str] | None = None,
    baseline: str | None = None,
    tasks: Sequence[str] | None = None,
    corpus: str | os.PathLike[str] | None = None,
    solution_name: str = "solution.py",
    **overrides: Any,
) -> EvalResult:
    """Grade a candidate directory (``DIR/<task>/solution.py``) or a baseline.

    Exactly one of ``candidate_dir`` and ``baseline`` must be given.
    ``overrides`` are passed to :func:`numguard.check` (e.g. ``large_n``).
    """
    if (candidate_dir is None) == (baseline is None):
        raise ValueError("give exactly one of candidate_dir and baseline")
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}")
    names = list(tasks) if tasks else list_tasks(corpus)
    results = []
    with np.errstate(all="ignore"):
        for name in names:
            task = load_task(name, corpus)
            if baseline is not None:
                impl = task.path / f"{baseline}.py"
            else:
                impl = Path(candidate_dir) / name / solution_name  # type: ignore[arg-type]
            results.append(grade_task(task, impl, **overrides))
    label = f"baseline:{baseline}" if baseline else str(candidate_dir)
    return EvalResult(candidate=label, results=results)
