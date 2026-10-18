"""Read-only tools the agent's model can call.

Nothing here executes repository code: files are read as text (size-capped,
inside the repository root, symlinks resolved and checked), ``grep`` uses
Python regular expressions over text files, and the catalogue describes
numguard itself. Every output is passed to the model as delimited untrusted
data by :class:`numguard.agent.session.Session`.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Callable

from .llm import ToolSpec
from .session import Tool, ToolError

__all__ = ["RepoReader", "catalogue_text", "repo_tools", "static_tool"]

MAX_FILE_BYTES = 1_000_000
MAX_LINES = 400
MAX_MATCHES = 60
TEXT_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json"}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", "build", "dist",
             "site-packages", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs"}


class RepoReader:
    """Read-only access to text files under ``root``."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()

    def _path(self, rel: Any) -> Path:
        if not isinstance(rel, str) or not rel or len(rel) > 300:
            raise ToolError("path must be a non-empty relative path")
        if rel.startswith(("/", "\\")) or "\\" in rel or "\x00" in rel or any(
                p in ("..",) for p in rel.split("/")):
            raise ToolError("path must be relative to the repository root, without '..'")
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path resolves outside the repository") from exc
        if ".git" in p.relative_to(self.root).parts:
            raise ToolError("files under .git are not readable")
        return p

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        if p.suffix.lower() not in TEXT_SUFFIXES:
            raise ToolError(f"only text files ({', '.join(sorted(TEXT_SUFFIXES))}) can be read")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file is larger than 1 MB")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + MAX_LINES - 1, start + MAX_LINES - 1)
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n... ({len(lines) - end} more lines; ask with start_line={end + 1})" if end < len(lines) else ""
        return f"{path} (lines {start}-{end} of {len(lines)}):\n{body}{more}"

    def _walk(self, glob: str) -> list[Path]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                p = Path(dirpath) / f
                rel = p.relative_to(self.root).as_posix()
                if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f, glob):
                    if p.is_symlink() or p.suffix.lower() not in TEXT_SUFFIXES:
                        continue
                    out.append(p)
            if len(out) > 5000:
                break
        return out

    def list_files(self, glob: str = "*.py") -> str:
        files = [p.relative_to(self.root).as_posix() for p in self._walk(glob or "*")]
        shown = files[:300]
        return "\n".join(shown) + (f"\n... and {len(files) - 300} more" if len(files) > 300 else "") or "(no files)"

    def grep(self, pattern: str, glob: str = "*.py", ignore_case: bool = False) -> str:
        if not isinstance(pattern, str) or not pattern or len(pattern) > 200:
            raise ToolError("pattern must be a regular expression of 1-200 characters")
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        hits: list[str] = []
        for p in self._walk(glob or "*.py"):
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line[:1000]):
                    hits.append(f"{p.relative_to(self.root).as_posix()}:{i}: {line.strip()[:200]}")
                    if len(hits) >= MAX_MATCHES:
                        return "\n".join(hits) + f"\n... (stopped after {MAX_MATCHES} matches)"
        return "\n".join(hits) or "(no matches)"


def catalogue_text() -> str:
    """numguard's vocabulary: domains, dtypes, properties, case kinds, options and reference forms."""
    from ..cases import KINDS, _kind_tags  # noqa: PLC0415
    from ..dtypes import ALL_FLOAT_DTYPES  # noqa: PLC0415
    from ..properties import PROPERTY_FACTORIES  # noqa: PLC0415
    from ..specdoc import PRECISIONS, reference_namespace  # noqa: PLC0415

    ns = reference_namespace()
    lines = ["# numguard catalogue", "",
             "## Domains (per input)",
             "any | positive | nonneg | negative | nonpos | unit_interval | interval notation such as "
             "'[-1, inf)', '(0, 1]' | integer ranges 'int[0, 255]'", "",
             "## Input entries",
             "a domain string, or {name, domain, shape, dtype}; shape is a list like [\"n\"] (default), "
             "[\"n\", 3], [] for a scalar; dtype pins an argument (default: the dtype under test)", "",
             f"## dtypes\n{', '.join(ALL_FLOAT_DTYPES)} (default: float32, float64)", "",
             "## Properties (metamorphic relations; optional ':value')"]
    for name, factory in sorted(PROPERTY_FACTORIES.items()):
        doc = (factory.__doc__ or "").strip().splitlines()[0]
        lines.append(f"- {name}: {doc}")
    lines += ["", "## Case kinds (generated automatically) and their tags"]
    for kind in KINDS:
        lines.append(f"- {kind}: {', '.join(sorted(_kind_tags(kind) - {kind}))}")
    lines += ["", "## Options",
              "- nan: propagate (default; NaN in -> NaN out) | raise | ignore",
              "- finite: true (default; finite inputs must give finite outputs unless the reference is non-finite)",
              "- allow_nonfinite / may_raise / exclude / include: lists of case names, kinds or tags (globs ok)",
              "- ignore: entries 'check:case-pattern' that downgrade matching results to skip",
              "- bounds: [lo, hi] for every output value; monotonic: increasing | decreasing (elementwise functions)",
              "- batched: true if f(stack(xs)) == stack(f(x)) along axis 0 (row-wise functions)",
              "- elementwise: true if each output element depends only on the same input element",
              "- fp_signals: ignore | fail (fail on overflow/invalid/divide-by-zero for finite inputs)",
              "- rtol / atol: number or {dtype: number}; defaults 1e-2 (float16), 1e-5 (float32), 1e-12 (float64)",
              "- out_dtype: preserve (default) | none | dtype | list per output",
              "- kwargs: extra keyword arguments passed to the function (scalars only)",
              "",
              "## Reference",
              "A Python EXPRESSION over the argument names (no statements, no lambdas, no imports), evaluated with "
              "these names available: " + ", ".join(sorted(k for k in ns if k != "__builtins__"))
              + "; builtins: abs, min, max, sum, len, range, float, int, bool, tuple, list, zip, enumerate, round, "
              "pow, all, any, sorted, reversed, complex.",
              f"reference_precision: {' | '.join(PRECISIONS)} -- float64 evaluates on float64-upcast arrays; "
              "longdouble on np.longdouble arrays; mpmath evaluates the expression ELEMENTWISE with each name "
              "bound to an mpmath.mpf scalar" + ("" if "mpmath" in ns else " (mpmath is NOT installed here)") + ".",
              "Example: \"special.logsumexp(x)\", \"np.hypot(x, y)\", \"mpmath.log1p(x)\" with mpmath precision."]
    return "\n".join(lines)


def static_tool(name: str, description: str, text_fn: Callable[[], str], untrusted: bool = True) -> Tool:
    return Tool(ToolSpec(name, description, {"type": "object", "properties": {}, "additionalProperties": False}),
                lambda args: text_fn(), untrusted=untrusted)


def repo_tools(reader: RepoReader) -> list[Tool]:
    """``read_file``, ``list_files``, ``grep`` and ``list_catalogue``."""
    return [
        Tool(ToolSpec("read_file", "Read a text file of the repository (line-numbered, at most 400 lines per call).",
                      {"type": "object", "properties": {
                          "path": {"type": "string", "description": "Path relative to the repository root."},
                          "start_line": {"type": "integer", "minimum": 1},
                          "end_line": {"type": "integer", "minimum": 1}},
                       "required": ["path"], "additionalProperties": False}),
             lambda a: reader.read_file(a["path"], int(a.get("start_line", 1)),
                                        int(a["end_line"]) if a.get("end_line") else None)),
        Tool(ToolSpec("list_files", "List repository files matching a glob (default *.py).",
                      {"type": "object", "properties": {"glob": {"type": "string"}}, "additionalProperties": False}),
             lambda a: reader.list_files(a.get("glob", "*.py"))),
        Tool(ToolSpec("grep", "Search text files with a Python regular expression; returns path:line: text "
                              "(use it to find call sites and tests of the function).",
                      {"type": "object", "properties": {
                          "pattern": {"type": "string"}, "glob": {"type": "string"},
                          "ignore_case": {"type": "boolean"}},
                       "required": ["pattern"], "additionalProperties": False}),
             lambda a: reader.grep(a["pattern"], a.get("glob", "*.py"), bool(a.get("ignore_case", False)))),
        static_tool("list_catalogue", "numguard's domains, dtypes, properties, case kinds, options and the "
                                      "reference-expression rules.", catalogue_text, untrusted=False),
    ]
