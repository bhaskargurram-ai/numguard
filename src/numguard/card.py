"""Render the numguard card (Markdown) from a *validated* result document.

Everything in a result can originate from the pull request (function names,
failure messages, model-written diffs), so before it is placed in the card it
is stripped of control characters, truncated, HTML-escaped, Markdown-escaped,
defused for @mentions, issue references and autolinks, and pipe-escaped in
tables. Model-written diffs go into a fenced code block whose fence is longer
than any backtick run inside it. The card ends with a machine-readable data
block (see :mod:`numguard.carddata`).
"""

from __future__ import annotations

import re
from typing import Any

CARD_MARKER = "<!-- numguard:card -->"
MAX_CARD_CHARS = 60_000
MAX_TARGET_ROWS = 40
MAX_FAILURE_ROWS = 12
MAX_DIFF_IN_CARD = 12_000

ZWJ = "\u200d"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u202a-\u202e\u2066-\u2069]")
_MD_SPECIAL_RE = re.compile(r"([\\`*_\[\]#!~|])")
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9])")
_URL_RE = re.compile(r"(?i)\b(https?|ftp|file|javascript|data)(:)")
_WWW_RE = re.compile(r"(?i)\bwww\.")

STATUS_ICONS = {"pass": "✅", "fail": "❌", "error": "⚠️"}
FIX_LABELS = {
    "verified": "✅ verified",
    "rejected": "❌ rejected by the checks",
    "refused": "⛔ refused before execution",
    "no_submission": "⚠️ no rewrite proposed",
    "error": "⚠️ error",
    "skipped": "⏭️ skipped",
}


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _clean(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = _CONTROL_RE.sub(" ", s.replace("\r", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return _truncate(s, limit)


def _defuse(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = _MENTION_RE.sub("@" + ZWJ, s)
    s = _URL_RE.sub(lambda m: m.group(1) + ZWJ + m.group(2), s)
    s = _WWW_RE.sub(lambda m: m.group(0)[:3] + ZWJ + ".", s)
    return s


def escape_md(text: Any, limit: int = 300) -> str:
    """Escape untrusted text for Markdown prose or a table cell."""
    s = _clean(text, limit)
    s = _MD_SPECIAL_RE.sub(r"\\\1", s)
    s = _defuse(s)
    s = re.sub(r"\\#(?=\d)", "#" + ZWJ, s)
    return s


def code(text: Any, limit: int = 120) -> str:
    """Untrusted text as an inline code span that is safe inside a table cell."""
    s = _clean(text, limit).replace("`", "'")
    if not s:
        return "` `"
    return "`" + s.replace("|", "\\|") + "`"


def fenced(text: Any, lang: str = "", limit: int = MAX_DIFF_IN_CARD) -> str:
    """Untrusted multi-line text as a fenced code block it cannot break out of."""
    s = "" if text is None else str(text)
    s = _CONTROL_RE.sub("", s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    "))
    if len(s) > limit:
        s = s[:limit].rsplit("\n", 1)[0] + "\n… (truncated)"
    longest = max((len(m) for m in re.findall(r"`+|~+", s)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{s.rstrip()}\n{fence}"


def _sha(s: Any) -> str:
    s = str(s or "")
    return s[:7] if re.fullmatch(r"[0-9a-f]{7,64}", s) else "?"


def _headline(result: dict[str, Any]) -> str:
    targets = result.get("targets") or []
    n_fail = sum(t["status"] == "fail" for t in targets)
    n_err = sum(t["status"] == "error" for t in targets)
    py = result.get("pytest")
    status = result.get("status", "error")
    if status == "pass":
        return f"✅ {len(targets)} target{'s' if len(targets) != 1 else ''} pass numerical edge-case checks" \
            if targets else "✅ pytest passed"
    parts = []
    if n_fail:
        parts.append(f"{n_fail} of {len(targets)} target{'s' if len(targets) != 1 else ''} "
                     f"fail{'s' if n_fail == 1 else ''} numerical edge-case checks")
    if n_err:
        parts.append(f"{n_err} target{'s' if n_err != 1 else ''} could not be checked")
    if py and py["status"] != "pass":
        parts.append("pytest " + ("failed" if py["status"] == "fail" else "could not run"))
    icon = "❌" if status == "fail" else "⚠️"
    return icon + " " + ("; ".join(parts) or "numguard could not complete its checks")


def _summary_line(result: dict[str, Any]) -> str:
    targets = result.get("targets") or []
    parts = [f"{len(targets)} target{'s' if len(targets) != 1 else ''}: "
             f"{sum(t['status'] == 'pass' for t in targets)} pass, {sum(t['status'] == 'fail' for t in targets)} fail"
             + (f", {sum(t['status'] == 'error' for t in targets)} error" if any(t['status'] == 'error' for t in targets)
                else "")]
    py = result.get("pytest")
    if py:
        parts.append(f"pytest: {py['passed']} passed, {py['failed']} failed"
                     + (f", {py['errors']} errors" if py["errors"] else "")
                     + (f", {py['skipped']} skipped" if py["skipped"] else ""))
    fixes = result.get("fixes")
    if fixes:
        v = sum(f["status"] == "verified" for f in fixes)
        parts.append(f"agent: {v}/{len(fixes)} suggested fix{'es' if len(fixes) != 1 else ''} verified")
    return " · ".join(parts)


def _targets_table(result: dict[str, Any]) -> list[str]:
    targets = result.get("targets") or []
    if not targets:
        return []
    out = ["| | target | cases | failing checks | detail |", "|---|---|---:|---|---|"]
    for t in targets[:MAX_TARGET_ROWS]:
        checks = ", ".join(t["failed_checks"][:6]) + (" …" if len(t["failed_checks"]) > 6 else "")
        if t["status"] == "error":
            detail = escape_md(t.get("error") or "error", 160)
        elif t["status"] == "fail":
            detail = f"{t['n_failures']} failing result{'s' if t['n_failures'] != 1 else ''}"
        else:
            detail = "dtypes: " + escape_md(", ".join(t["dtypes"]), 60) if t["dtypes"] else ""
        if t.get("backend"):
            detail += f" (backend: {escape_md(t['backend'], 20)})"
        out.append(f"| {STATUS_ICONS[t['status']]} | {code(t['name'], 60)} | {t['n_cases']} | "
                   f"{escape_md(checks, 200) or '–'} | {detail} |")
    if len(targets) > MAX_TARGET_ROWS:
        out.append(f"\n_… and {len(targets) - MAX_TARGET_ROWS} more targets (see result.json)._")
    return out + [""]


def _failure_sections(result: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for t in result.get("targets") or []:
        if t["status"] != "fail" or not t["failures"]:
            continue
        out += [f"<details><summary>❌ <b>{escape_md(t['name'], 60)}</b> — failing checks "
                f"({len(t['failures'])} (check, case) group{'s' if len(t['failures']) != 1 else ''})</summary>", "",
                f"Function {code(t['function'], 120)}"
                + (f", reference {code(t['reference'], 80)}" if t.get("reference") else "") + ".", "",
                "| check | case | dtypes | rel err | ulp | detail |", "|---|---|---|---:|---:|---|"]
        for f in t["failures"][:MAX_FAILURE_ROWS]:
            out.append(f"| {code(f['check'], 40)} | {code(f['case'], 40)} | {escape_md(f['dtypes'], 40)} | "
                       f"{escape_md(f['rel_err'], 12)} | {escape_md(f['ulp'], 12)} | {escape_md(f['detail'], 160)} |")
        if len(t["failures"]) > MAX_FAILURE_ROWS:
            out.append(f"\n_… {len(t['failures']) - MAX_FAILURE_ROWS} more groups in result.json._")
        out += ["", "</details>", ""]
    return out


def _pytest_section(result: dict[str, Any]) -> list[str]:
    py = result.get("pytest")
    if not py or py["status"] == "pass":
        return []
    msg = f" ({escape_md(py['message'], 200)})" if py.get("message") else ""
    return [f"{STATUS_ICONS[py['status']]} **pytest**: {py['passed']} passed, {py['failed']} failed, "
            f"{py['errors']} errors{msg}.", ""]


def _fix_sections(result: dict[str, Any]) -> list[str]:
    fixes = result.get("fixes")
    if not fixes:
        return []
    out = ["### 🤖 Suggested fixes (experimental)", "",
           "An LLM proposed rewrites of failing functions; numguard re-ran every check on each rewrite in a "
           "subprocess. Only **verified** rewrites pass all checks the original failed without new failures. "
           "Review before applying.", ""]
    for f in fixes:
        label = FIX_LABELS.get(f["status"], f["status"])
        after = "–" if f.get("after_failures") is None else str(f["after_failures"])
        out.append(f"**{escape_md(f['target'], 60)}**: {label} · failing results {f['before_failures']} → {after}"
                   + (f" · {f['rounds']} attempt{'s' if f['rounds'] != 1 else ''}" if f.get("rounds") else "")
                   + (f" · model {code(f['model'], 60)}" if f.get("model") else ""))
        if f.get("rationale"):
            out.append(f"> _model's explanation (not verified)_: {escape_md(f['rationale'], 300)}")
        for r in f.get("reasons", [])[:5]:
            out.append(f"> {escape_md(r, 300)}")
        if f.get("diff"):
            title = "suggested fix (verified)" if f["status"] == "verified" else "rejected rewrite"
            out += ["", f"<details><summary>{title}: diff</summary>", "", fenced(f["diff"], "diff"), "", "</details>"]
        out.append("")
    return out


def render_card(result: dict[str, Any], trusted_warnings: list[str] | None = None) -> str:
    """Markdown card for a validated result (``trusted_warnings`` are placed verbatim)."""
    lines = [CARD_MARKER, f"## numguard: {_headline(result)}", ""]
    for w in trusted_warnings or []:
        lines += [w, ""]
    lines += [_summary_line(result), ""]
    lines += _targets_table(result)
    lines += _failure_sections(result)
    lines += _pytest_section(result)
    lines += _fix_sections(result)
    for n in (result.get("notes") or [])[:5]:
        lines.append(f"> {escape_md(n, 300)}")
    if result.get("notes"):
        lines.append("")
    lines += ["---",
              "<sub>numguard runs each function on its edge-case catalogue (zeros, empty arrays, NaN, ±inf, "
              "huge and subnormal magnitudes, cancellation, float16, ...) and compares against references and "
              "properties. Results come from executed checks; an LLM never decides a verdict. "
              f"Head <code>{_sha(result.get('head'))}</code> · numguard {escape_md(result.get('tool_version', ''), 20)}"
              "</sub>"]
    card = "\n".join(lines) + "\n"
    from .carddata import card_data_block  # noqa: PLC0415

    data = card_data_block(result, trusted_warnings=len(trusted_warnings or []))
    limit = MAX_CARD_CHARS - len(data) - 1
    if len(card) > limit:
        card = card[: limit - 200] + "\n\n_…card truncated; see result.json for the full report._\n"
    return card + data + "\n"


def harness_warning(paths: list[str]) -> str:
    """Warning shown when the PR modifies files that control how numguard runs."""
    shown = ", ".join(code(p, 80) for p in paths[:6]) + (f" and {len(paths) - 6} more" if len(paths) > 6 else "")
    return ("> [!WARNING]\n> This PR modifies files that control how this check runs (" + shown + "). "
            "The results below were produced by the PR's own workflow run and could have been altered by the PR.")
