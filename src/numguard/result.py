"""The ``result.json`` document: what CI produces and the comment workflow trusts *after validation*.

``result.json`` is written in the unprivileged job that runs pull-request code,
so the privileged comment job treats it as hostile: :func:`validate_result`
checks every type, enum, count and length, drops unknown keys, and the card is
re-rendered from the validated copy.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from . import __version__

__all__ = [
    "RESULT_SCHEMA",
    "RESULT_VERSION",
    "STATUSES",
    "FIX_STATUSES",
    "MAX_RESULT_BYTES",
    "ResultValidationError",
    "build_result",
    "target_entry",
    "validate_result",
    "overall_status",
]

RESULT_SCHEMA = "numguard-result"
RESULT_VERSION = 1
STATUSES = ("pass", "fail", "error")
FIX_STATUSES = ("verified", "rejected", "refused", "no_submission", "error", "skipped")
MAX_RESULT_BYTES = 5_000_000
MAX_TARGETS = 200
MAX_FAILURE_ROWS = 50
MAX_DIFF_CHARS = 20_000
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")


class ResultValidationError(ValueError):
    pass


def overall_status(statuses: Sequence[str]) -> str:
    if "error" in statuses:
        return "error"
    if "fail" in statuses:
        return "fail"
    return "pass"


def target_entry(outcome: Any, max_rows: int = MAX_FAILURE_ROWS) -> dict[str, Any]:
    """A result.json target entry from a :class:`numguard.runconfig.TargetOutcome`."""
    rep = outcome.report
    entry: dict[str, Any] = {
        "name": outcome.name,
        "function": outcome.function,
        "status": outcome.status,
        "error": outcome.error,
        "n_cases": rep.n_cases if rep is not None else 0,
        "n_results": len(rep.results) if rep is not None else 0,
        "n_failures": len(rep.failures) if rep is not None else 0,
        "dtypes": list(rep.dtypes) if rep is not None else [],
        "reference": rep.reference if rep is not None else None,
        "backend": rep.backend if rep is not None else None,
        "failed_checks": rep.failed_checks() if rep is not None else [],
        "failures": rep.grouped_failures()[:max_rows] if rep is not None else [],
        "duration_s": round(float(outcome.duration_s), 3),
    }
    return entry


def build_result(run: Any, *, pr_number: int | None = None, head: str | None = None, base: str | None = None,
                 config_path: str | None = None, fixes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The result document for a :class:`numguard.runconfig.ConfigRun`."""
    py = None
    if run.pytest is not None:
        p = run.pytest
        py = {"status": p.status, "passed": p.passed, "failed": p.failed, "errors": p.errors,
              "skipped": p.skipped, "duration_s": round(p.duration_s, 3), "message": p.message}
    doc = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "tool_version": __version__,
        "status": run.status,
        "pr_number": pr_number,
        "head": head,
        "base": base,
        "config": config_path,
        "targets": [target_entry(t) for t in run.targets],
        "pytest": py,
        "fixes": fixes,
        "notes": list(run.notes),
    }
    return doc


# --------------------------------------------------------------------------- validation


def _fail(path: str, msg: str) -> None:
    raise ResultValidationError(f"{path}: {msg}")


def _dict(v: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(v, Mapping):
        _fail(path, "must be an object")
    return v


def _str(v: Any, path: str, limit: int, *, optional: bool = False) -> str | None:
    if v is None and optional:
        return None
    if not isinstance(v, str):
        _fail(path, "must be a string")
    if len(v) > limit:
        _fail(path, f"longer than {limit} characters")
    return v


def _int(v: Any, path: str, *, optional: bool = False, maximum: int = 10 ** 12) -> int | None:
    if v is None and optional:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= maximum:
        _fail(path, "must be a non-negative integer")
    return v


def _num(v: Any, path: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
        _fail(path, "must be a finite non-negative number")
    return float(v)


def _enum(v: Any, path: str, choices: Sequence[str]) -> str:
    if v not in choices:
        _fail(path, f"must be one of {', '.join(choices)}")
    return v


def _list(v: Any, path: str, limit: int) -> list[Any]:
    if not isinstance(v, list):
        _fail(path, "must be a list")
    if len(v) > limit:
        _fail(path, f"more than {limit} entries")
    return v


def _sha(v: Any, path: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or not _SHA_RE.match(v):
        _fail(path, "must be a lowercase hex commit SHA")
    return v


def _failure_row(v: Any, path: str) -> dict[str, str]:
    d = _dict(v, path)
    limits = {"check": 80, "case": 80, "dtypes": 60, "rel_err": 20, "ulp": 20, "detail": 400}
    return {k: _str(d.get(k, ""), f"{path}.{k}", lim) or "" for k, lim in limits.items()}


def _target(v: Any, path: str) -> dict[str, Any]:
    d = _dict(v, path)
    return {
        "name": _str(d.get("name"), f"{path}.name", 80),
        "function": _str(d.get("function"), f"{path}.function", 300),
        "status": _enum(d.get("status"), f"{path}.status", STATUSES),
        "error": _str(d.get("error"), f"{path}.error", 600, optional=True),
        "n_cases": _int(d.get("n_cases", 0), f"{path}.n_cases"),
        "n_results": _int(d.get("n_results", 0), f"{path}.n_results"),
        "n_failures": _int(d.get("n_failures", 0), f"{path}.n_failures"),
        "dtypes": [_str(x, f"{path}.dtypes[]", 20) for x in _list(d.get("dtypes", []), f"{path}.dtypes", 8)],
        "reference": _str(d.get("reference"), f"{path}.reference", 200, optional=True),
        "backend": _str(d.get("backend"), f"{path}.backend", 20, optional=True),
        "failed_checks": [_str(x, f"{path}.failed_checks[]", 80)
                          for x in _list(d.get("failed_checks", []), f"{path}.failed_checks", 60)],
        "failures": [_failure_row(x, f"{path}.failures[{i}]")
                     for i, x in enumerate(_list(d.get("failures", []), f"{path}.failures", MAX_FAILURE_ROWS))],
        "duration_s": _num(d.get("duration_s", 0), f"{path}.duration_s"),
    }


def _pytest(v: Any, path: str) -> dict[str, Any] | None:
    if v is None:
        return None
    d = _dict(v, path)
    out: dict[str, Any] = {"status": _enum(d.get("status"), f"{path}.status", STATUSES)}
    for k in ("passed", "failed", "errors", "skipped"):
        out[k] = _int(d.get(k, 0), f"{path}.{k}")
    out["duration_s"] = _num(d.get("duration_s", 0), f"{path}.duration_s")
    out["message"] = _str(d.get("message"), f"{path}.message", 300, optional=True)
    return out


def _fix(v: Any, path: str) -> dict[str, Any]:
    d = _dict(v, path)
    return {
        "target": _str(d.get("target"), f"{path}.target", 80),
        "status": _enum(d.get("status"), f"{path}.status", FIX_STATUSES),
        "rounds": _int(d.get("rounds", 0), f"{path}.rounds", maximum=100),
        "before_failures": _int(d.get("before_failures", 0), f"{path}.before_failures"),
        "after_failures": _int(d.get("after_failures"), f"{path}.after_failures", optional=True),
        "fixed_checks": [_str(x, f"{path}.fixed_checks[]", 80)
                         for x in _list(d.get("fixed_checks", []), f"{path}.fixed_checks", 60)],
        "diff": _str(d.get("diff", ""), f"{path}.diff", MAX_DIFF_CHARS) or "",
        "reasons": [_str(x, f"{path}.reasons[]", 400) for x in _list(d.get("reasons", []), f"{path}.reasons", 10)],
        "model": _str(d.get("model"), f"{path}.model", 100, optional=True),
        "rationale": _str(d.get("rationale"), f"{path}.rationale", 2000, optional=True),
    }


def validate_result(doc: Any) -> dict[str, Any]:
    """Return a validated, normalized copy of ``doc`` (raises :class:`ResultValidationError`)."""
    d = _dict(doc, "result")
    if d.get("schema") != RESULT_SCHEMA:
        _fail("result.schema", f"must be {RESULT_SCHEMA!r}")
    if d.get("version") != RESULT_VERSION:
        _fail("result.version", f"unsupported version {d.get('version')!r}")
    targets = [_target(t, f"result.targets[{i}]") for i, t in enumerate(_list(d.get("targets", []), "result.targets",
                                                                                MAX_TARGETS))]
    pytest = _pytest(d.get("pytest"), "result.pytest")
    fixes_raw = d.get("fixes")
    fixes = None if fixes_raw is None else [_fix(f, f"result.fixes[{i}]")
                                            for i, f in enumerate(_list(fixes_raw, "result.fixes", 50))]
    out = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "tool_version": _str(d.get("tool_version", ""), "result.tool_version", 40),
        "status": _enum(d.get("status"), "result.status", STATUSES),
        "pr_number": _int(d.get("pr_number"), "result.pr_number", optional=True, maximum=9_999_999_999),
        "head": _sha(d.get("head"), "result.head"),
        "base": _sha(d.get("base"), "result.base"),
        "config": _str(d.get("config"), "result.config", 300, optional=True),
        "targets": targets,
        "pytest": pytest,
        "fixes": fixes,
        "notes": [_str(n, "result.notes[]", 500) for n in _list(d.get("notes", []), "result.notes", 20)],
    }
    # the overall status must agree with its parts (a forged "pass" cannot hide failures)
    expected = overall_status([t["status"] for t in targets] + ([pytest["status"]] if pytest else []))
    if out["status"] != expected:
        _fail("result.status", f"is {out['status']!r} but the targets say {expected!r}")
    for t in targets:
        if (t["status"] == "fail") != (t["n_failures"] > 0) and t["status"] != "error":
            _fail(f"result.targets[{t['name']}]", "status and n_failures disagree")
    return out
