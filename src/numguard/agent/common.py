"""Shared helpers for the agent commands: target source lookup and usage reporting."""

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..resolve import SourceInfo, TargetError, function_source, locate_source, resolve_target, split_target
from .llm import Usage, prices_from_env

__all__ = ["TargetInfo", "target_info", "usage_dict", "arg_names_from_source"]


@dataclass
class TargetInfo:
    target: str
    qualname: str
    source: str
    path: Path | None
    rel_path: str
    lineno: int
    arg_names: list[str]
    signature: str
    source_info: SourceInfo | None  # set when found without importing (needed for --apply)


def arg_names_from_source(source: str) -> tuple[list[str], str]:
    """Required positional parameter names and a signature string, from a function's source."""
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    n_req = len(positional) - len(a.defaults)
    names = [p.arg for p in positional[:n_req] if p.arg not in ("self", "cls")]
    sig = "(" + ast.unparse(a) + ")"
    return names or ["x"], sig


def target_info(target: str, root: str | os.PathLike[str], *, allow_import: bool = True) -> TargetInfo:
    """Find the target's source: statically first (no code runs), else by importing it."""
    root = Path(root).resolve()
    try:
        info = locate_source(target, root)
        names, sig = arg_names_from_source(info.source)
        rel = info.path.resolve().relative_to(root).as_posix()
        return TargetInfo(target, info.qualname, info.source, info.path, rel, info.lineno, names, sig, info)
    except (TargetError, OSError, StopIteration, SyntaxError) as exc:
        if not allow_import:
            raise TargetError(f"cannot find the source of {target}: {exc}") from exc
    fn = resolve_target(target, root)
    try:
        src = inspect.getsource(fn)
        path = Path(inspect.getsourcefile(fn) or "")
        line = inspect.getsourcelines(fn)[1]
    except (OSError, TypeError) as exc:
        raise TargetError(f"cannot read the source of {target}: {exc}") from exc
    names, sig = arg_names_from_source(src)
    try:
        rel = path.resolve().relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    _, attr = split_target(target)
    info = None
    try:  # a file inside the repository: allow --apply
        text = path.read_text(encoding="utf-8")
        s, lineno, end, dec = function_source(text, attr)
        info = SourceInfo(path, attr, s, lineno, end, dec, text) if path.resolve().is_relative_to(root) else None
    except (OSError, TargetError, SyntaxError, ValueError):
        info = None
    return TargetInfo(target, attr, src, path, rel, line, names, sig, info)


def usage_dict(usage: Usage, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    price_in, price_out = prices_from_env(env)
    d: dict[str, Any] = usage.to_dict()
    d["cost_usd"] = usage.cost(price_in, price_out)
    return d


def usage_line(usage: Usage, env: Mapping[str, str] | None = None) -> str:
    d = usage_dict(usage, env)
    cost = f", ~${d['cost_usd']:.4f}" if d["cost_usd"] is not None else ""
    return (f"{d['requests']} request(s), {d['input_tokens'] + d['cache_read_tokens'] + d['cache_write_tokens']} "
            f"input + {d['output_tokens']} output tokens{cost}")
