"""``numguard fix``: an LLM proposes a rewrite of a failing function; numguard verifies it.

Herbie-style repair, except that nothing is trusted until it has been
executed and re-checked:

1. numguard runs the function's spec in a subprocess worker ("before").
2. The model (read-only tools: the module's files, the failure table,
   numguard's catalogue) submits a rewrite: imports + the complete function,
   same name and signature.
3. The rewrite is checked statically (:func:`numguard.agent.sandbox.static_check`);
   a denylisted rewrite is **refused before execution** and the model is told why.
4. The rewrite is executed only in the subprocess worker (timeout, scrubbed
   environment). It is **verified** only if it has the same signature, passes
   every check the original failed, introduces no new failure, agrees with the
   original on benign inputs where the original passed every check, and keeps
   the output shapes and dtypes there.
5. Otherwise the failure table of the rewrite goes back to the model for
   another round (``--rounds``).

The result is a unified diff and a before/after table; files are only
changed with ``apply_rewrite`` (``numguard fix --apply``).
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..resolve import SourceInfo
from .common import TargetInfo, target_info, usage_dict
from .llm import ToolLLM, Usage
from .prompts import FIX_SYSTEM, SUBMIT_REWRITE, failures_table, fix_feedback, fix_message, spec_text, untrusted_rule
from .sandbox import WorkerResult, failure_keys, run_worker, static_check
from .session import Session, new_nonce
from .tools import RepoReader, repo_tools, static_tool

__all__ = ["Attempt", "FixOutcome", "evaluate_candidate", "run_fix", "apply_rewrite", "make_diff",
           "before_after_rows", "rewrite_validator"]


@dataclass
class Attempt:
    round: int
    status: str  # verified | rejected | refused | error
    code: str
    rationale: str = ""
    reasons: list[str] = field(default_factory=list)
    n_failures: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FixOutcome:
    target: str
    status: str  # verified | rejected | refused | no_submission | error | nothing_to_fix
    before: dict[str, Any] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    code: str = ""
    diff: str = ""
    new_text: str = ""
    rel_path: str = ""
    error: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def verified(self) -> Attempt | None:
        ok = [a for a in self.attempts if a.status == "verified"]
        return min(ok, key=lambda a: (a.n_failures if a.n_failures is not None else 0, a.round)) if ok else None

    def fix_entry(self) -> dict[str, Any]:
        """The ``fixes[]`` entry of result.json."""
        best = self.verified or self.final
        after = best.n_failures if best is not None else None
        fixed = sorted({k[0] for k in failure_keys(self.before.get("failures", []))}
                       - ({k[0] for k in failure_keys(best.data.get("failures", []))} if best and best.data else set())
                       ) if best and best.data else []
        status = self.status if self.status != "nothing_to_fix" else "skipped"
        return {
            "target": self.target[:80], "status": status, "rounds": len(self.attempts),
            "before_failures": int(self.before.get("n_failures", 0)), "after_failures": after,
            "fixed_checks": fixed[:60], "diff": self.diff[:20_000],
            "reasons": [r[:400] for r in (best.reasons if best else ([self.error] if self.error else []))][:10],
            "model": self.model[:100] or None, "rationale": (best.rationale[:2000] if best else None) or None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "status": self.status, "error": self.error, "model": self.model,
            "before": {"n_failures": self.before.get("n_failures"), "summary": self.before.get("summary")},
            "attempts": [{"round": a.round, "status": a.status, "reasons": a.reasons, "n_failures": a.n_failures,
                          "rationale": a.rationale, "code": a.code} for a in self.attempts],
            "diff": self.diff, "before_after": before_after_rows(self.before, (self.verified or self.final).data
                                                                 if self.attempts else {}),
            "usage": usage_dict(self.usage),
        }


def rewrite_validator(name: str) -> Callable[[dict[str, Any]], tuple[Any, list[str]]]:
    def validate(args: dict[str, Any]) -> tuple[Any, list[str]]:
        code = args.get("code")
        problems = static_check(code, name)
        if problems:
            return None, ["static check (the code was NOT executed): " + p for p in problems]
        return {"code": code, "rationale": str(args.get("rationale", ""))[:2000]}, []

    return validate


def evaluate_candidate(before: Mapping[str, Any], wr: WorkerResult) -> tuple[bool, list[str]]:
    """Apply the verification criteria to an executed rewrite."""
    if not wr.ok:
        stage = wr.data.get("stage", "worker") if wr.data else "worker"
        return False, [f"the rewrite could not be checked ({stage}): {wr.error}"]
    data = wr.data
    reasons: list[str] = []
    sig = data.get("signature") or {}
    if not sig.get("match", False):
        reasons.append(f"signature changed: {sig.get('candidate')} (original {sig.get('original')})")
    ob = failure_keys(before.get("failures", []))
    oa = failure_keys(data.get("failures", []))
    still = sorted(ob & oa)
    new = sorted(oa - ob)
    if still:
        reasons.append(f"{len(still)} originally failing result(s) still fail, e.g. "
                       + ", ".join(f"{c}:{k}[{d}]" for c, k, d in still[:5]))
    if new:
        reasons.append(f"{len(new)} new failure(s), e.g. " + ", ".join(f"{c}:{k}[{d}]" for c, k, d in new[:5]))
    bad_cases = {k for _, k, _ in ob}
    mism = [m for m in (data.get("agreement") or {}).get("mismatches", []) if m.get("case") not in bad_cases]
    if mism:
        reasons.append("differs from the original on ordinary inputs where the original passed every check: "
                       + "; ".join(f"{m['case']}[{m['dtype']}]: {m['message']}" for m in mism[:3]))
    shapes = [m for m in (data.get("shapes") or {}).get("mismatches", []) if m.get("case") not in bad_cases]
    if shapes:
        reasons.append("changes output shape/dtype where the original passed every check: "
                       + "; ".join(f"{m['case']}[{m['dtype']}]: {m['message']}" for m in shapes[:3]))
    return not reasons, reasons


def before_after_rows(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per check: failing results before and after."""
    def count(fails: Sequence[Sequence[str]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in fails:
            out[str(f[0])] = out.get(str(f[0]), 0) + 1
        return out

    b, a = count(before.get("failures", [])), count(after.get("failures", []) if after else [])
    return [{"check": c, "before": b.get(c, 0), "after": a.get(c, 0) if after else None}
            for c in sorted(set(b) | set(a))]


def _split_candidate(code: str) -> tuple[list[str], str]:
    """(top-level import statements, the rest) of a candidate's source."""
    tree = ast.parse(code)
    lines = code.splitlines()
    imports: list[str] = []
    drop: set[int] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            end = stmt.end_lineno or stmt.lineno
            imports.append("\n".join(lines[stmt.lineno - 1:end]))
            drop.update(range(stmt.lineno, end + 1))
    rest = "\n".join(line for i, line in enumerate(lines, 1) if i not in drop).strip("\n")
    return imports, rest


def apply_rewrite(info: SourceInfo, code: str) -> str:
    """The module text with the target function replaced by the candidate.

    The original decorators are kept on the new definition; imports the
    candidate needs and the module lacks are added after the module's last
    top-level import.
    """
    imports, rest = _split_candidate(code)
    text = info.module_source
    lines = text.splitlines()
    decorators = lines[info.decorator_lineno - 1:info.lineno - 1]
    rest_lines = rest.splitlines()
    if decorators:
        tree = ast.parse(rest)
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == info.qualname.split(".")[-1]), None)
        if node is not None and not node.decorator_list:
            at = node.lineno - 1
            rest_lines = rest_lines[:at] + decorators + rest_lines[at:]
    new_lines = lines[:info.decorator_lineno - 1] + rest_lines + lines[info.end_lineno:]
    tree = ast.parse(text)
    existing = {ast.unparse(s) for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))}
    missing = [imp for imp in imports if ast.unparse(ast.parse(imp).body[0]) not in existing]
    if missing:
        top_imports = [s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))
                       and (s.end_lineno or s.lineno) < info.decorator_lineno]
        if top_imports:
            at = top_imports[-1].end_lineno or top_imports[-1].lineno
        else:
            first = tree.body[0] if tree.body else None
            doc = (first is not None and isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                   and isinstance(first.value.value, str))
            at = (first.end_lineno or 0) if doc else 0
        new_lines = new_lines[:at] + missing + new_lines[at:]
    out = "\n".join(new_lines)
    return out + ("\n" if text.endswith("\n") else "")


def make_diff(old: str, new: str, rel_path: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3))


def _function_diff(info: TargetInfo, code: str) -> tuple[str, str]:
    """(diff, new module text or "") for a candidate."""
    if info.source_info is not None:
        try:
            new_text = apply_rewrite(info.source_info, code)
            return make_diff(info.source_info.module_source, new_text, info.rel_path), new_text
        except (SyntaxError, ValueError):
            pass
    return make_diff(info.source, code.rstrip("\n") + "\n", info.rel_path), ""


def run_fix(target: str, *, llm: ToolLLM, root: str | Path = ".", spec_req: Mapping[str, Any],
            rounds: int = 3, max_tool_calls: int = 25, timeout: float = 120,
            log: Callable[[str], None] = lambda s: None) -> FixOutcome:
    """The local propose -> verify -> refine loop (see the module docstring)."""
    root = Path(root).resolve()
    info = target_info(target, root)
    out = FixOutcome(target=target, status="error", rel_path=info.rel_path, model=llm.model)
    base_req = {"root": str(root), "target": target, "name": info.qualname, "spec": dict(spec_req)}
    log(f"fix: checking {target} (before) ...")
    wb = run_worker(base_req, timeout=timeout, cwd=root)
    if not wb.ok:
        out.error = f"could not check the original ({wb.data.get('stage', 'worker')}): {wb.error}"
        return out
    out.before = wb.data
    if not wb.data.get("n_failures"):
        out.status = "nothing_to_fix"
        return out
    nonce = new_nonce()
    current = {"groups": wb.data.get("groups", [])}
    tools = [static_tool("show_failures", "The current failure table (of the original, or of your last rewrite).",
                         lambda: failures_table(current["groups"]))] + repo_tools(RepoReader(root))
    session = Session(llm, FIX_SYSTEM.format(untrusted=untrusted_rule(nonce)), tools, SUBMIT_REWRITE,
                      rewrite_validator(info.qualname), max_tool_calls=max_tool_calls, nonce=nonce, log=log)
    res = session.run(fix_message(target, f"{info.rel_path}:{info.lineno}", info.source, spec_text(spec_req),
                                  failures_table(wb.data.get("groups", [])), nonce))
    for rnd in range(1, max(1, rounds) + 1):
        if res.status != "submitted":
            # rejected static checks inside the session never executed anything
            out.status = "refused" if res.rejections and not out.attempts else (
                "rejected" if out.attempts else res.status)
            out.error = res.error
            if res.rejections and not out.attempts:
                out.attempts.append(Attempt(rnd, "refused", "", reasons=res.rejections[-1]))
            break
        code, rationale = res.value["code"], res.value["rationale"]
        log(f"fix: round {rnd}: running the rewrite in a subprocess ...")
        wr = run_worker({**base_req, "candidate": {"code": code, "name": info.qualname}, "compare_original": True},
                        timeout=timeout, cwd=root)
        ok, reasons = evaluate_candidate(wb.data, wr)
        att = Attempt(rnd, "verified" if ok else ("error" if not wr.ok else "rejected"), code, rationale, reasons,
                      wr.data.get("n_failures") if wr.ok else None, wr.data if wr.ok else {})
        out.attempts.append(att)
        out.code = code
        if ok:
            out.status = "verified"
            break
        out.status = "rejected"
        if rnd < rounds:
            current["groups"] = wr.data.get("groups", []) if wr.ok else []
            res = session.followup(fix_feedback(rnd, reasons, failures_table(current["groups"]) if wr.ok else "",
                                                nonce))
    out.usage = session.usage
    out.transcript = session.transcript
    best = out.verified or (out.final if out.final and out.final.code else None)
    if best is not None:
        out.code = best.code
        out.diff, out.new_text = _function_diff(info, best.code)
    return out
