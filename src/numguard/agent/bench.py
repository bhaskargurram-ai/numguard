"""``numguard bench``: run an LLM on the bug corpus and grade it with numguard.

For every corpus task the model gets the task prompt (``task.md``) and
submits ``solution.py`` (through a ``submit_solution`` tool; a fenced
```python block in a plain-text answer is accepted as a fallback). Each
solution is checked statically first (a denylisted solution is *refused* and
never runs), then graded by ``numguard eval`` in a subprocess with a timeout
and a scrubbed environment, one task at a time. The result is a leaderboard
row: pass@1 (one sample per task), happy-path pass rate and per-task reasons.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..corpus import list_tasks, load_task
from .common import usage_dict
from .llm import LLMToolError, ToolLLM, Turn, Usage
from .prompts import BENCH_SYSTEM, SUBMIT_SOLUTION
from .sandbox import child_environment, static_check

__all__ = ["BenchTask", "BenchResult", "run_bench", "extract_code_block"]

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code_block(text: str) -> str | None:
    blocks = _CODE_RE.findall(text or "")
    return max(blocks, key=len) if blocks else None


@dataclass
class BenchTask:
    task: str
    status: str  # pass | fail | error | missing | refused | no_submission
    happy_path: bool | None = None
    failed_checks: list[str] = field(default_factory=list)
    failed_cases: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "status": self.status, "happy_path": self.happy_path,
                "failed_checks": self.failed_checks, "failed_cases": self.failed_cases[:10], "reason": self.reason}


@dataclass
class BenchResult:
    model: str
    provider: str
    tasks: list[BenchTask]
    solutions_dir: str
    usage: Usage = field(default_factory=Usage)

    @property
    def n_pass(self) -> int:
        return sum(t.status == "pass" for t in self.tasks)

    @property
    def n_happy(self) -> int:
        return sum(bool(t.happy_path) for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        n = len(self.tasks)
        return {"model": self.model, "provider": self.provider, "n_tasks": n, "n_pass": self.n_pass,
                "pass_at_1": round(self.n_pass / n, 4) if n else 0.0, "n_happy_path_pass": self.n_happy,
                "solutions_dir": self.solutions_dir, "tasks": [t.to_dict() for t in self.tasks],
                "usage": usage_dict(self.usage)}

    def leaderboard_row(self) -> str:
        n = len(self.tasks)
        fails = "; ".join(f"{t.task}: {t.reason or ', '.join(t.failed_checks) or t.status}"
                          for t in self.tasks if t.status != "pass")
        return (f"| {self.model} | {self.n_pass}/{n} ({100 * self.n_pass / max(n, 1):.0f}%) | "
                f"{self.n_happy}/{n} | {fails or '-'} |")

    def table(self) -> str:
        from ..report import format_table  # noqa: PLC0415

        rows = [[t.task, {True: "pass", False: "FAIL", None: "-"}[t.happy_path], t.status,
                 ", ".join(t.failed_checks) or t.reason] for t in self.tasks]
        return format_table(["task", "happy-path", "numguard", "failed checks / reason"], rows,
                            max_widths=[20, 10, 14, 70])


def _ask(llm: ToolLLM, prompt: str, usage: Usage) -> tuple[str | None, str]:
    turns = [Turn("user", text=prompt)]
    try:
        reply = llm.chat(BENCH_SYSTEM, turns, [SUBMIT_SOLUTION], force_tool=SUBMIT_SOLUTION.name)
    except LLMToolError as exc:
        return None, f"LLM error: {exc}"
    usage.add(reply.usage)
    for call in reply.tool_calls:
        if call.name == SUBMIT_SOLUTION.name and isinstance(call.arguments.get("code"), str):
            return call.arguments["code"], ""
    code = extract_code_block(reply.text)
    if code:
        return code, ""
    return None, "the model returned no solution"


def _grade(task: str, cand_dir: Path, corpus: str | None, timeout: float) -> BenchTask:
    with tempfile.TemporaryDirectory(prefix="numguard-bench-") as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "numguard", "eval", "--candidate-dir", str(cand_dir), "--task", task,
               "--json-out", str(out), "--json"]
        if corpus:
            cmd += ["--corpus-dir", corpus]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=child_environment(),  # noqa: S603
                           check=False, start_new_session=True)
        except subprocess.TimeoutExpired:
            return BenchTask(task, "error", reason=f"grading timed out after {timeout:g}s")
        if not out.is_file():
            return BenchTask(task, "error", reason="the grader produced no result (the solution crashed it)")
        data = json.loads(out.read_text(encoding="utf-8"))
    t = data["tasks"][0]
    return BenchTask(task, t["status"], t.get("happy_path"), list(t.get("failed_checks") or []),
                     list(t.get("failed_cases") or []), reason=t.get("error") or "")


def run_bench(llm: ToolLLM, *, tasks: Sequence[str] | None = None, solutions_dir: str | Path,
              corpus: str | None = None, timeout: float = 300,
              log: Callable[[str], None] = lambda s: None) -> BenchResult:
    names = list(tasks) if tasks else list_tasks(corpus)
    sol_root = Path(solutions_dir)
    sol_root.mkdir(parents=True, exist_ok=True)
    result = BenchResult(llm.model, llm.provider, [], str(sol_root))
    for name in names:
        task = load_task(name, corpus)
        log(f"bench: {name}: asking the model ...")
        code, why = _ask(llm, task.prompt, result.usage)
        tdir = sol_root / name
        tdir.mkdir(parents=True, exist_ok=True)
        for stale in ("solution.py", "solution.py.refused"):
            (tdir / stale).unlink(missing_ok=True)
        if code is None:
            result.tasks.append(BenchTask(name, "no_submission", reason=why))
            continue
        problems = static_check(code, task.function)
        if problems:
            (tdir / "solution.py.refused").write_text(code, encoding="utf-8")
            result.tasks.append(BenchTask(name, "refused", reason="refused before execution: " + problems[0]))
            continue
        (tdir / "solution.py").write_text(code, encoding="utf-8")
        log(f"bench: {name}: grading in a subprocess ...")
        result.tasks.append(_grade(name, sol_root, corpus, timeout))
    return result
