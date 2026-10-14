"""Resolving ``module:function`` targets: by import, or as source text without importing.

:func:`resolve_target` imports the module (the function is going to be
called, so its module is executed anyway). :func:`locate_source` finds the
same function's *source text* through the file system and the AST only,
without executing anything; the agent's propose step uses it so that it can
read a pull request's code as data.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = ["TargetError", "split_target", "resolve_target", "SourceInfo", "locate_source", "function_source"]

MAX_SOURCE_BYTES = 2_000_000


class TargetError(ValueError):
    """A target string cannot be resolved."""


def split_target(target: str) -> tuple[str, str]:
    """Split ``module:attr``, ``path/file.py:attr`` or ``module.attr`` into (module, attr)."""
    if not isinstance(target, str) or not target.strip():
        raise TargetError("empty target")
    target = target.strip()
    drive_only = len(target) > 2 and target[1] == ":" and target[2] in "/\\" and target.count(":") == 1
    if ":" in target and not drive_only:
        mod_name, _, attr = target.rpartition(":")
    elif "." in target:
        mod_name, _, attr = target.rpartition(".")
    else:
        raise TargetError(f"expected 'module:function', got {target!r}")
    if not mod_name or not attr:
        raise TargetError(f"expected 'module:function', got {target!r}")
    return mod_name, attr


def _is_path(mod_name: str) -> bool:
    return mod_name.endswith(".py") or "/" in mod_name or "\\" in mod_name


def resolve_target(target: str, root: str | Path | None = None) -> Callable[..., Any]:
    """Import and return the callable named by ``target``.

    ``root`` (default: the current directory) is put on ``sys.path`` so local
    modules resolve as with ``python -m``; relative file paths are relative to it.
    """
    mod_name, attr = split_target(target)
    base = Path(root) if root is not None else Path.cwd()
    if _is_path(mod_name):
        path = Path(mod_name)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise TargetError(f"file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise TargetError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(path.stem, module)
        spec.loader.exec_module(module)
    else:
        for p in (base, base / "src"):
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            module = importlib.import_module(mod_name)
        except ImportError as exc:
            raise TargetError(f"cannot import module {mod_name!r}: {exc}") from exc
    obj: Any = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise TargetError(f"{mod_name!r} has no attribute {attr!r}") from exc
    if not callable(obj):
        raise TargetError(f"{target!r} is not callable")
    return obj


@dataclass
class SourceInfo:
    """Where a function's definition lives, found without importing it."""

    path: Path
    qualname: str
    source: str
    lineno: int        # first line of the ``def`` (decorators excluded)
    end_lineno: int
    decorator_lineno: int  # first decorator line (== lineno without decorators)
    module_source: str


def _module_file(mod_name: str, roots: Sequence[Path]) -> Path:
    parts = mod_name.split(".")
    if not all(p.isidentifier() for p in parts):
        raise TargetError(f"invalid module name {mod_name!r}")
    for root in roots:
        base = root.joinpath(*parts)
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.is_file():
                return cand
    raise TargetError(f"module {mod_name!r} not found under {', '.join(str(r) for r in roots)}")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def function_source(module_source: str, qualname: str) -> tuple[str, int, int, int]:
    """``(source, lineno, end_lineno, decorator_lineno)`` of a top-level function or method."""
    tree = ast.parse(module_source)
    body: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    parts = qualname.split(".")
    for i, part in enumerate(parts):
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise TargetError(f"no definition named {qualname!r}")
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                raise TargetError(f"{'.'.join(parts[:i + 1])} is not a class")
            body = node.body
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TargetError(f"{qualname!r} is not a function")
    lines = module_source.splitlines(keepends=True)
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    end = node.end_lineno or node.lineno
    return "".join(lines[node.lineno - 1:end]), node.lineno, end, start


def locate_source(target: str, root: str | Path) -> SourceInfo:
    """Find the source of ``target`` under ``root`` (and ``root/src``) without importing anything.

    Paths that resolve outside ``root`` (for example through symlinks) are refused.
    """
    mod_name, attr = split_target(target)
    root = Path(root)
    if _is_path(mod_name):
        path = Path(mod_name)
        if path.is_absolute():
            raise TargetError("absolute paths are not allowed here")
        path = root / path
        if not path.is_file():
            raise TargetError(f"file not found: {mod_name}")
    else:
        path = _module_file(mod_name, [root, root / "src"])
    if not _within(path, root):
        raise TargetError(f"{mod_name} resolves outside the repository")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise TargetError(f"{path.name} is too large")
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        src, lineno, end, dec = function_source(text, attr)
    except SyntaxError as exc:
        raise TargetError(f"cannot parse {path.name}: {exc.msg}") from exc
    return SourceInfo(path, attr, src, lineno, end, dec, text)
