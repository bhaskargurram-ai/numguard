"""Machine-readable data block at the end of the numguard card.

The card ends with one HTML comment::

    <!-- numguard:data {"v":1,"status":"fail","targets":[{"name":"lse","status":"fail",...}],...} -->

It is built only from fields of a *validated* result document; the JSON is
compact and ASCII-only, and every ``-``, ``<``, ``>``, ``&`` and ``@`` is a
``\\u`` escape, so nothing inside can end the comment or open markup. Strings
are cleaned and capped, lists are capped, and the block never exceeds
:data:`MAX_DATA_CHARS`. :func:`parse_card_data` only accepts the block when it
is the last thing in the card.
"""

from __future__ import annotations

import json
import re
from typing import Any

DATA_PREFIX = "<!-- numguard:data "
DATA_SUFFIX = " -->"
DATA_VERSION = 1
MAX_DATA_CHARS = 4000
MAX_TARGETS = 30

_ESCAPED = "-<>&@"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f‪-‮⁦-⁩]")
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_TRAILING_RE = re.compile(r"(?:\A|\n)<!-- numguard:data (\{[^\n]*\}) -->\s*\Z")


def _text(value: Any, limit: int) -> str:
    s = _CONTROL_RE.sub(" ", "" if value is None else str(value))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _sha(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHA_RE.match(value) else None


def build_card_data(result: dict[str, Any], trusted_warnings: int = 0) -> dict[str, Any]:
    data: dict[str, Any] = {
        "v": DATA_VERSION,
        "tool": _text(result.get("tool_version"), 20),
        "pr": result.get("pr_number") if _count(result.get("pr_number")) else None,
        "head": _sha(result.get("head")),
        "base": _sha(result.get("base")),
        "status": _text(result.get("status"), 10),
    }
    targets = []
    for t in (result.get("targets") or [])[:MAX_TARGETS]:
        targets.append({"name": _text(t.get("name"), 40), "status": _text(t.get("status"), 10),
                        "failures": _count(t.get("n_failures")),
                        "checks": [_text(c, 30) for c in (t.get("failed_checks") or [])[:6]]})
    data["targets"] = targets
    if len(result.get("targets") or []) > len(targets):
        data["targets_truncated"] = True
    data["counts"] = {s: sum(1 for t in (result.get("targets") or []) if t.get("status") == s)
                      for s in ("pass", "fail", "error")}
    py = result.get("pytest")
    data["pytest"] = None if not py else {"status": _text(py.get("status"), 10),
                                          **{k: _count(py.get(k)) for k in ("passed", "failed", "errors", "skipped")}}
    fixes = result.get("fixes")
    data["fixes"] = None if fixes is None else [
        {"target": _text(f.get("target"), 40), "status": _text(f.get("status"), 20),
         "before": _count(f.get("before_failures")),
         "after": _count(f.get("after_failures")) if f.get("after_failures") is not None else None}
        for f in fixes[:10]]
    data["warnings"] = _count(trusted_warnings)
    return data


def encode_card_data(data: dict[str, Any]) -> str:
    s = json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for ch in _ESCAPED:
        s = s.replace(ch, f"\\u{ord(ch):04x}")
    return s


def card_data_block(result: dict[str, Any], trusted_warnings: int = 0) -> str:
    """The ``<!-- numguard:data ... -->`` line, at most :data:`MAX_DATA_CHARS` long."""
    data = build_card_data(result, trusted_warnings)
    budget = MAX_DATA_CHARS - len(DATA_PREFIX) - len(DATA_SUFFIX)
    payload = encode_card_data(data)
    if len(payload) > budget:
        for t in data["targets"]:
            t["checks"] = t["checks"][:2]
            t["name"] = _text(t["name"], 16)
        payload = encode_card_data(data)
    while len(payload) > budget and data["targets"]:
        data["targets"].pop()
        data["targets_truncated"] = True
        payload = encode_card_data(data)
    if len(payload) > budget:  # pragma: no cover - fixed-size fields only
        data = {k: data[k] for k in ("v", "head", "status", "counts")}
        payload = encode_card_data(data)
    return DATA_PREFIX + payload + DATA_SUFFIX


def parse_card_data(markdown: Any) -> dict[str, Any] | None:
    """The data block of a card, or None (only a block at the very end counts)."""
    if not isinstance(markdown, str):
        return None
    m = _TRAILING_RE.search(markdown[-(MAX_DATA_CHARS + 64):])
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("v"), int):
        return None
    return data
