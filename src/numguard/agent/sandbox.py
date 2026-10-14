"""Guard rails for executing model-written code.

Everything an LLM writes that numguard executes (a reference expression from
``numguard suggest``, a rewritten function from ``numguard fix``, a benchmark
solution from ``numguard bench``) goes through two layers:

1. :func:`static_check` -- a static **denylist** applied before anything
   runs: no imports outside a small numerical allowlist (``os``,
   ``subprocess``, ``socket``, ``shutil``, ``ctypes``, ``importlib`` and
   friends are named explicitly), no ``open``/``eval``/``exec``/``compile``/
   ``__import__``/``getattr``..., no dunder attributes, no file-I/O NumPy
   functions, no ``global``/``nonlocal``. A rejected submission is refused and
   never executed.
2. :func:`run_worker` -- execution happens only in a **subprocess**
   (``python -m numguard.agent.worker``) with a timeout (the whole process
   group is killed), a scrubbed environment (every variable whose name looks
   like a secret, including ``NUMGUARD_LLM_API_KEY``, is removed) and its own
   temporary working files. Only the executed, re-checked outcome counts.

This is defence in depth, **not a security sandbox**: the subprocess runs as
the same user with network and file-system access. In CI the code runs in a
job that holds no secrets (see docs/SECURITY_MODEL.md).
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ALLOWED_IMPORTS",
    "DENIED_MODULES",
    "DENIED_CALLS",
    "SECRET_MARKERS",
    "MAX_CODE_CHARS",
    "static_check",
    "scrubbed_env",
    "WorkerResult",
    "run_worker",
]

#: Top-level modules model-written code may import.
ALLOWED_IMPORTS = frozenset({
    "numpy", "math", "cmath", "scipy", "mpmath", "functools", "itertools", "operator", "numbers",
    "typing", "warnings", "dataclasses", "collections", "__future__", "decimal", "fractions", "statistics",
})
#: Modules named explicitly in rejections (they are also outside the allowlist).
DENIED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "importlib", "pathlib", "io", "builtins",
    "pickle", "marshal", "multiprocessing", "threading", "signal", "urllib", "http", "requests",
    "tempfile", "glob", "inspect", "runpy", "code", "pty", "asyncio", "gc", "resource", "ftplib",
    "smtplib", "telnetlib", "webbrowser", "sqlite3", "shelve", "dbm", "zipfile", "tarfile",
})
DENIED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "breakpoint", "input", "memoryview", "exit", "quit", "help",
})
DENIED_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "fromregex", "memmap", "DataSource", "ctypeslib", "ctypes", "f2py", "distutils", "system", "popen",
    "seterrcall", "setbufsize", "show_config", "dump", "dumps",
})
#: Environment variables whose names contain one of these are removed for subprocesses.
SECRET_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "COOKIE", "SESSION")
MAX_CODE_CHARS = 30_000


def _root(name: str) -> str:
    return name.split(".")[0]


def static_check(code: Any, function_name: str | None = None, *, require_function: bool = True,
                 allowed_imports: frozenset[str] = ALLOWED_IMPORTS) -> list[str]:
    """Static denylist check of model-written code. Returns problems (empty = acceptable).

    With ``function_name``, the code must define that function at top level.
    Top-level statements are limited to imports, function definitions,
    assignments and a docstring.
    """
    if not isinstance(code, str) or not code.strip():
        return ["code must be a non-empty string"]
    if len(code) > MAX_CODE_CHARS:
        return [f"code is longer than {MAX_CODE_CHARS} characters"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]
    problems: list[str] = []
    defined = []
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.FunctionDef):
            defined.append(stmt.name)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            pass
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            pass  # docstring / string literal
        else:
            problems.append(f"line {stmt.lineno}: top-level {type(stmt).__name__} statements are not allowed "
                            "(only imports, function definitions and assignments)")
    if function_name is not None and require_function and function_name not in defined:
        problems.append(f"the code must define a top-level function named {function_name!r}")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    problems.append(f"line {line}: relative imports are not allowed")
                    continue
                names = [node.module or ""]
            else:
                names = [a.name for a in node.names]
            for name in names:
                root = _root(name)
                if root in DENIED_MODULES:
                    problems.append(f"line {line}: import of {root!r} is not allowed")
                elif root not in allowed_imports:
                    problems.append(f"line {line}: import of {root!r} is not allowed "
                                    f"(allowed: {', '.join(sorted(allowed_imports - {'__future__'}))})")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            problems.append(f"line {line}: calling {node.func.id}() is not allowed")
        elif isinstance(node, ast.Name) and node.id in DENIED_CALLS and isinstance(node.ctx, ast.Load):
            problems.append(f"line {line}: using {node.id!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            problems.append(f"line {line}: dunder name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"line {line}: dunder attribute {node.attr!r} is not allowed")
            elif node.attr in DENIED_ATTRIBUTES:
                problems.append(f"line {line}: attribute {node.attr!r} (file/OS access) is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {line}: {type(node).__name__.lower()} statements are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncWith, ast.AsyncFor)):
            problems.append(f"line {line}: async code is not allowed")
        elif isinstance(node, ast.ClassDef):
            problems.append(f"line {line}: class definitions are not allowed")
    return list(dict.fromkeys(problems))


def scrubbed_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``env`` without secret-looking variables, plus deterministic settings."""
    src = os.environ if env is None else env
    out = {k: v for k, v in src.items()
           if not any(m in k.upper() for m in SECRET_MARKERS) and not k.upper().startswith("NUMGUARD_LLM_")}
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    out["PYTHONHASHSEED"] = "0"
    out.setdefault("OMP_NUM_THREADS", "2")
    out.setdefault("OPENBLAS_NUM_THREADS", "2")
    out.setdefault("MKL_NUM_THREADS", "2")
    return out


@dataclass
class WorkerResult:
    ok: bool
    data: dict[str, Any]
    error: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    stderr_tail: str = ""


def _numguard_path() -> str:
    import numguard  # noqa: PLC0415

    return str(Path(numguard.__file__).resolve().parent.parent)


def child_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """:func:`scrubbed_env` plus this numguard on ``PYTHONPATH`` (so the child imports the same copy)."""
    child_env = scrubbed_env(env)
    pp = [_numguard_path()] + [p for p in child_env.get("PYTHONPATH", "").split(os.pathsep) if p]
    child_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pp))
    return child_env


def run_worker(request: Mapping[str, Any], *, timeout: float = 120, cwd: str | os.PathLike[str] | None = None,
               env: Mapping[str, str] | None = None, python: str | None = None) -> WorkerResult:
    """Run ``python -m numguard.agent.worker`` on ``request`` in a subprocess.

    The request and response travel through temporary JSON files (so prints of
    the code under test cannot corrupt them). On timeout the whole process
    group is killed.
    """
    t0 = time.perf_counter()
    child_env = child_environment(env)
    with tempfile.TemporaryDirectory(prefix="numguard-worker-") as tmp:
        req = Path(tmp) / "request.json"
        resp = Path(tmp) / "response.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        cmd = [python or sys.executable, "-m", "numguard.agent.worker", str(req), str(resp)]
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=child_env, stdout=subprocess.PIPE,  # noqa: S603
                                    stderr=subprocess.PIPE, text=True, **kwargs)
        except OSError as exc:
            return WorkerResult(False, {}, f"cannot start worker: {exc}", duration_s=time.perf_counter() - t0)
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.communicate()
            return WorkerResult(False, {}, f"timed out after {timeout:g}s", timed_out=True,
                                duration_s=time.perf_counter() - t0)
        finally:
            if proc.poll() is None:  # pragma: no cover - defensive
                _kill(proc)
        dur = time.perf_counter() - t0
        tail = (stderr or "")[-3000:]
        if not resp.is_file():
            return WorkerResult(False, {}, f"worker exited with status {proc.returncode} and no result",
                                duration_s=dur, stderr_tail=tail)
        try:
            data = json.loads(resp.read_text(encoding="utf-8"))
        except ValueError as exc:
            return WorkerResult(False, {}, f"unreadable worker result: {exc}", duration_s=dur, stderr_tail=tail)
    if not isinstance(data, dict):
        return WorkerResult(False, {}, "unexpected worker result", duration_s=dur, stderr_tail=tail)
    return WorkerResult(bool(data.get("ok")), data, str(data.get("error") or ""), duration_s=dur, stderr_tail=tail)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def failure_keys(failures: Sequence[Sequence[str]]) -> set[tuple[str, str, str]]:
    """``{(check, case, dtype)}`` from a worker's ``failures`` list."""
    return {(str(f[0]), str(f[1]), str(f[2])) for f in failures if len(f) >= 3}
