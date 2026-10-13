"""Structured check results and the :class:`Report` that aggregates them."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = ["CheckResult", "Report", "NumGuardError", "PASS", "FAIL", "SKIP", "format_table"]

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


class NumGuardError(AssertionError):
    """Raised by :meth:`Report.raise_if_failed`; the message is the summary table."""

    def __init__(self, report: "Report"):
        self.report = report
        super().__init__(report.summary())


@dataclass
class CheckResult:
    """Outcome of one check on one case (or one aggregated property check).

    ``max_abs_err``/``max_rel_err``/``max_ulp`` are filled in by comparing
    checks (reference, batch, properties); ``fp_events`` lists floating-point
    exceptions (``overflow``, ``invalid value``, ``divide by zero``) raised
    while the function under test ran on the case.
    """

    check: str
    case: str
    dtype: str
    status: str
    message: str = ""
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    max_ulp: float | None = None
    fp_events: list[str] = field(default_factory=list)
    example: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("max_abs_err", "max_rel_err", "max_ulp"):
            d[k] = _json_float(d[k])
        return d


def _json_float(v: float | None) -> float | str | None:
    if v is None:
        return None
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(v)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 1e7:
        return f"{int(v)}"
    return f"{v:.2g}"


def _truncate(s: str, width: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= width else s[: max(0, width - 3)] + "..."


def format_table(headers: Sequence[str], rows: Iterable[Sequence[str]],
                 max_widths: Sequence[int] | None = None) -> str:
    """Render a plain-text table with box-drawing-free ASCII separators."""
    rows = [list(r) for r in rows]
    if max_widths:
        rows = [[_truncate(c, w) for c, w in zip(r, max_widths)] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |", line]
    for r in rows:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    out.append(line)
    return "\n".join(out)


@dataclass
class Report:
    """All results of :func:`numguard.check` for one function."""

    function: str
    results: list[CheckResult] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    n_cases: int = 0
    reference: str | None = None
    duration_s: float = 0.0

    # ------------------------------------------------------------------ queries
    @property
    def ok(self) -> bool:
        """True if no check failed."""
        return not any(r.failed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed]

    @property
    def passes(self) -> list[CheckResult]:
        return [r for r in self.results if r.passed]

    def counts(self) -> dict[str, dict[str, int]]:
        """``{check: {"pass": n, "fail": n, "skip": n}}`` in first-seen order."""
        out: dict[str, dict[str, int]] = {}
        for r in self.results:
            c = out.setdefault(r.check, {PASS: 0, FAIL: 0, SKIP: 0})
            c[r.status] = c.get(r.status, 0) + 1
        return out

    def failed_checks(self) -> list[str]:
        return [c for c, v in self.counts().items() if v[FAIL]]

    def failed_cases(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.failures:
            seen.setdefault(r.case, None)
        return list(seen)

    def __bool__(self) -> bool:
        return self.ok

    # --------------------------------------------------------------- rendering
    def _grouped_failures(self) -> list[list[str]]:
        groups: dict[tuple[str, str], list[CheckResult]] = {}
        for r in self.failures:
            groups.setdefault((r.check, r.case), []).append(r)
        rows = []
        for (check, case), rs in groups.items():
            dtypes = ",".join(dict.fromkeys(r.dtype for r in rs))

            def worst(attr: str) -> float | None:
                vals = [getattr(r, attr) for r in rs if getattr(r, attr) is not None]
                if not vals:
                    return None
                return max(vals, key=lambda v: (math.isnan(v), v))

            first = rs[0]
            detail = first.message
            events = sorted({e for r in rs for e in r.fp_events})
            if events and "fp:" not in detail:
                detail += f" [fp: {', '.join(events)}]"
            rows.append([check, case, dtypes, _fmt(worst("max_rel_err")),
                         _fmt(worst("max_ulp")), detail])
        return rows

    def summary(self, max_rows: int = 25, width: int = 72) -> str:
        """Human-readable summary: verdict, per-check tally and failure table.

        Failures are grouped by (check, case) with the dtypes they occurred in.
        """
        n_fail = len(self.failures)
        verdict = "PASSED" if self.ok else "FAILED"
        head = (f"numguard: {self.function} {verdict} - "
                f"{len(self.results) - n_fail - sum(1 for r in self.results if r.status == SKIP)}"
                f" passed, {n_fail} failed"
                f" ({self.n_cases} cases; dtypes: {', '.join(self.dtypes) or '-'}"
                + (f"; reference: {self.reference}" if self.reference else "") + ")")
        lines = [head]
        if self.ok:
            return "\n".join(lines)
        rows = self._grouped_failures()
        shown = rows[:max_rows]
        lines.append(format_table(["check", "case", "dtypes", "rel err", "ulp", "detail"],
                                  shown, max_widths=[26, 22, 24, 8, 8, width]))
        if len(rows) > max_rows:
            lines.append(f"... and {len(rows) - max_rows} more failing (check, case) groups")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "ok": self.ok,
            "dtypes": list(self.dtypes),
            "n_cases": self.n_cases,
            "reference": self.reference,
            "duration_s": round(self.duration_s, 4),
            "counts": self.counts(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int | None = 2, failures_only: bool = False) -> str:
        """Serialize to JSON (non-finite floats become strings)."""
        d = self.to_dict()
        if failures_only:
            d["results"] = [r for r in d["results"] if r["status"] == FAIL]
        return json.dumps(d, indent=indent)

    def raise_if_failed(self) -> "Report":
        """Raise :class:`NumGuardError` (an ``AssertionError``) if any check failed."""
        if not self.ok:
            raise NumGuardError(self)
        return self
