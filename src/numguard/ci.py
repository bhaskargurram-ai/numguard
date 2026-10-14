"""CI helpers: load a validated result and write the card to the job summary.

:func:`write_summary` needs no token, so the card is visible on the workflow
run page even for pull requests from forks, where the unprivileged job
cannot comment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from .card import render_card
from .result import MAX_RESULT_BYTES, validate_result

__all__ = ["load_validated", "write_summary", "MAX_SUMMARY_BYTES"]

#: GitHub limits one step summary to 1 MiB.
MAX_SUMMARY_BYTES = 1_000_000


def load_validated(path: str | Path) -> dict:
    """Read and validate result.json (raises ValueError / ResultValidationError / OSError)."""
    p = Path(path)
    if p.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("result.json is too large")
    return validate_result(json.loads(p.read_text(encoding="utf-8")))


def write_summary(json_path: str | Path, out_path: str | Path | None = None,
                  env: Mapping[str, str] | None = None) -> str | None:
    """Append the card to ``out_path`` or ``$GITHUB_STEP_SUMMARY``; print it when neither is set."""
    e = os.environ if env is None else env
    card = render_card(load_validated(json_path))
    dest = str(out_path) if out_path else e.get("GITHUB_STEP_SUMMARY") or None
    if dest is None:
        print(card, end="")
        return None
    data = card.encode("utf-8")[:MAX_SUMMARY_BYTES].decode("utf-8", errors="ignore")
    with open(dest, "a", encoding="utf-8") as fh:
        fh.write(data + "\n")
    return dest
