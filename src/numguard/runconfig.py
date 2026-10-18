"""Run every target of a ``numguard.toml`` (and the configured pytest run)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .api import check
from .config import Config, PytestConfig
from .report import Report
from .resolve import resolve_target
from .specdoc import spec_kwargs

__all__ = ["TargetOutcome", "PytestOutcome", "ConfigRun", "run_target", "run_pytest", "run_config",
           "build_check_kwargs"]


@dataclass
class TargetOutcome:
    name: str
    function: str
    status: str  # pass | fail | error
    report: Report | None = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class PytestOutcome:
    status: str  # pass | fail | error
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    message: str | None = None
    output_tail: str = ""


@dataclass
class ConfigRun:
    targets: list[TargetOutcome] = field(default_factory=list)
    pytest: PytestOutcome | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = [t.status for t in self.targets] + ([self.pytest.status] if self.pytest else [])
        if "error" in statuses:
            return "error"
        if "fail" in statuses:
            return "fail"
        return "pass"


def _short(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}".rstrip(": ")[:500]


def build_check_kwargs(doc: Mapping[str, Any], fn: Callable[..., Any], root: Path) -> dict[str, Any]:
    """``numguard.check`` keyword arguments for a (trusted, validated) target document."""
    reference = None
    if "reference_function" in doc:
        reference = resolve_target(doc["reference_function"], root)
    return spec_kwargs(doc, fn, reference=reference)


def run_target(doc: Mapping[str, Any], root: Path, **overrides: Any) -> TargetOutcome:
    """Import and check one target. Errors (import, spec) become ``status="error"``."""
    t0 = time.perf_counter()
    name, function = doc["name"], doc["function"]
    try:
        fn = resolve_target(function, root)
        kwargs = build_check_kwargs(doc, fn, root)
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        report = check(fn, name=name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - user code and user config
        if os.environ.get("NUMGUARD_DEBUG"):
            traceback.print_exc()
        return TargetOutcome(name, function, "error", error=_short(exc), duration_s=time.perf_counter() - t0)
    return TargetOutcome(name, function, "pass" if report.ok else "fail", report=report,
                         duration_s=time.perf_counter() - t0)


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()  # noqa: S314 - produced by the local pytest run
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = failures = errors = skipped = 0
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    return tests - failures - errors - skipped, failures, errors, skipped


def run_pytest(conf: PytestConfig, root: Path, python: str | None = None) -> PytestOutcome:
    """Run ``python -m pytest <args>`` in ``root`` and summarize the JUnit XML."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="numguard-pytest-") as tmp:
        junit = Path(tmp) / "junit.xml"
        cmd = [python or sys.executable, "-m", "pytest", *conf.args, f"--junitxml={junit}", "-q",
               "-p", "no:cacheprovider"]
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=conf.timeout,
                                  check=False, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return PytestOutcome("error", message=f"pytest timed out after {conf.timeout}s",
                                 duration_s=time.perf_counter() - t0)
        except OSError as exc:
            return PytestOutcome("error", message=_short(exc), duration_s=time.perf_counter() - t0)
        tail = (proc.stdout + proc.stderr)[-4000:]
        dur = time.perf_counter() - t0
        if not junit.is_file():
            return PytestOutcome("error", message=f"pytest exited with status {proc.returncode} "
                                 "and wrote no report", duration_s=dur, output_tail=tail)
        try:
            passed, failed, errors, skipped = _junit_counts(junit)
        except (ET.ParseError, ValueError) as exc:
            return PytestOutcome("error", message=f"unreadable JUnit report: {exc}", duration_s=dur, output_tail=tail)
    if proc.returncode == 5:
        return PytestOutcome("error", message="pytest collected no tests", duration_s=dur, output_tail=tail)
    if proc.returncode not in (0, 1):
        return PytestOutcome("error", passed, failed, errors, skipped, dur,
                             message=f"pytest exited with status {proc.returncode}", output_tail=tail)
    status = "pass" if proc.returncode == 0 and not failed and not errors else "fail"
    return PytestOutcome(status, passed, failed, errors, skipped, dur, output_tail=tail)


def run_config(config: Config, names: Sequence[str] | None = None, *, run_pytest_too: bool = True,
               log: Callable[[str], None] = lambda s: None, **overrides: Any) -> ConfigRun:
    """Run the selected targets (default: all) and, if configured, pytest."""
    root = config.root
    targets = [config.target(n) for n in names] if names else list(config.targets)
    run = ConfigRun()
    for doc in targets:
        log(f"numguard: checking {doc['name']} ({doc['function']}) ...")
        run.targets.append(run_target(doc, root, **overrides))
    if run_pytest_too and config.pytest is not None and not names:
        log("numguard: running pytest " + " ".join(config.pytest.args))
        run.pytest = run_pytest(config.pytest, root)
    if not run.targets and run.pytest is None:
        run.notes.append("numguard.toml has no [[target]] tables (and no [pytest] run): nothing was checked")
    return run
