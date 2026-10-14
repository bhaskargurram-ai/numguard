"""``numguard suggest``: an LLM proposes a spec; numguard validates it and runs it.

1. The model reads the function (read-only tools: its source, files and call
   sites in the repository, numguard's catalogue) and calls ``submit_spec``.
2. The submission is validated strictly (:func:`numguard.specdoc.validate_spec_doc`
   with the model-allowed keys; the reference expression is checked statically).
3. numguard runs the spec against the function in a subprocess worker (the
   reference expression is model-written code, so it is only evaluated there).
4. If the reference disagrees with the function on *benign* inputs (ordinary
   finite values, not edge cases), the reference is **suspect**: the model gets
   one more round to correct it; if it stays suspect, the reference failures are
   not reported as bugs and the spec is re-run without the reference.

The output is a ready-to-paste ``@numguard.guard(...)`` decorator, a
``[[target]]`` table for numguard.toml, a Python spec/test file, and the
check report. The model never decides whether the function is correct: only
the executed checks do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..specdoc import MODEL_KEYS, SpecDocError, render_decorator, render_python_spec, render_toml_target, \
    validate_spec_doc
from .common import TargetInfo, target_info, usage_dict
from .llm import ToolLLM, Usage
from .prompts import SUBMIT_SPEC, SUGGEST_SYSTEM, failures_table, suggest_feedback, suggest_message, untrusted_rule
from .sandbox import run_worker
from .session import Session, new_nonce
from .tools import RepoReader, repo_tools, static_tool

__all__ = ["SuggestOutcome", "run_suggest"]


@dataclass
class SuggestOutcome:
    target: str
    status: str  # ok | no_submission | error
    doc: dict[str, Any] | None = None
    arg_names: list[str] = field(default_factory=list)
    reference_status: str = "none"  # none | consistent | suspect | error
    suspect_cases: list[str] = field(default_factory=list)
    check_ok: bool | None = None
    n_failures: int = 0
    summary: str = ""
    groups: list[dict[str, Any]] = field(default_factory=list)
    decorator: str = ""
    toml: str = ""
    python_spec: str = ""
    rounds: int = 0
    notes: list[str] = field(default_factory=list)
    error: str = ""
    usage: Usage = field(default_factory=Usage)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "status": self.status, "spec": self.doc, "arg_names": self.arg_names,
            "reference_status": self.reference_status, "suspect_cases": self.suspect_cases,
            "check_ok": self.check_ok, "n_failures": self.n_failures, "summary": self.summary,
            "failures": self.groups, "decorator": self.decorator, "toml": self.toml,
            "rounds": self.rounds, "notes": self.notes, "error": self.error, "usage": usage_dict(self.usage),
        }


def _validator(info: TargetInfo) -> Callable[[dict[str, Any]], tuple[Any, list[str]]]:
    def validate(args: dict[str, Any]) -> tuple[Any, list[str]]:
        try:
            doc = validate_spec_doc(args, allowed=MODEL_KEYS, arg_names=info.arg_names)
        except SpecDocError as exc:
            return None, exc.problems
        if "inputs" not in doc:
            return None, ["inputs is required (one entry per required positional argument)"]
        return doc, []

    return validate


def run_suggest(target: str, *, llm: ToolLLM, root: str | Path = ".", rounds: int = 2, max_tool_calls: int = 25,
                timeout: float = 120, overrides: Mapping[str, Any] | None = None,
                log: Callable[[str], None] = lambda s: None) -> SuggestOutcome:
    root = Path(root).resolve()
    info = target_info(target, root)
    out = SuggestOutcome(target=target, status="error", arg_names=list(info.arg_names))
    nonce = new_nonce()
    reader = RepoReader(root)
    source_text = (f"{info.rel_path}:{info.lineno}\n{info.source}")
    tools = [static_tool("read_function", "The target function's source code and location.",
                         lambda: source_text)] + repo_tools(reader)
    session = Session(llm, SUGGEST_SYSTEM.format(untrusted=untrusted_rule(nonce)), tools, SUBMIT_SPEC,
                      _validator(info), max_tool_calls=max_tool_calls, nonce=nonce, log=log)
    message = suggest_message(target, info.signature, info.arg_names, f"{info.rel_path}:{info.lineno}", nonce)
    worker_base = {"root": str(root), "target": target, "name": info.qualname}
    ov = dict(overrides or {})
    res = session.run(message)
    data: dict[str, Any] = {}
    for rnd in range(1, max(1, rounds) + 1):
        out.rounds = rnd
        if res.status != "submitted":
            out.status = res.status
            out.error = res.error
            break
        doc = res.value
        log(f"suggest: round {rnd}: spec submitted; running it in a subprocess ...")
        wr = run_worker({**worker_base, "spec": {"source": "doc", "doc": doc, "overrides": ov}}, timeout=timeout,
                        cwd=root)
        if not wr.ok:
            problem = f"The spec could not be run ({wr.data.get('stage', 'worker')}: {wr.error})."
            if rnd < rounds:
                res = session.followup(suggest_feedback(problem, wr.error, nonce))
                continue
            out.status, out.doc, out.error = "error", doc, problem
            out.reference_status = "error" if "reference" in doc else "none"
            break
        data = wr.data
        out.doc = doc
        out.status = "ok"
        suspect = data.get("regular_reference_failures") or []
        if "reference" not in doc:
            out.reference_status = "none"
        elif suspect:
            out.reference_status = "suspect"
            out.suspect_cases = list(suspect)
            if rnd < rounds:
                rows = [g for g in data.get("groups", []) if g.get("check") == "reference"]
                log(f"suggest: reference disagrees on ordinary inputs ({', '.join(suspect[:5])}); asking again")
                res = session.followup(suggest_feedback(
                    "The reference disagrees with the function on ORDINARY inputs (not edge cases): "
                    + ", ".join(suspect[:10]) + ".", failures_table(rows), nonce))
                continue
        else:
            out.reference_status = "consistent"
        break
    out.usage = session.usage
    out.transcript = session.transcript
    if out.status != "ok" or out.doc is None:
        return out
    doc = out.doc
    if out.reference_status == "suspect":
        out.notes.append("The reference disagrees with the function on ordinary inputs ("
                         + ", ".join(out.suspect_cases[:8]) + "). It is marked SUSPECT: its disagreements are not "
                         "reported as bugs. Check by hand whether the reference or the function is wrong.")
        no_ref = {k: v for k, v in doc.items() if k not in ("reference", "reference_precision")}
        wr = run_worker({**worker_base, "spec": {"source": "doc", "doc": no_ref, "overrides": ov}},
                        timeout=timeout, cwd=root)
        if wr.ok:
            data = wr.data
        else:
            out.notes.append(f"re-running without the reference failed: {wr.error}")
            data = {}
    out.check_ok = data.get("n_failures", 0) == 0 if data else None
    out.n_failures = int(data.get("n_failures", 0)) if data else 0
    out.summary = data.get("summary", "")
    out.groups = list(data.get("groups", []))
    header = (f"Suggested by `numguard suggest` (model {llm.model}); reference status: {out.reference_status}."
              + (f" Rationale: {doc['rationale']}" if doc.get("rationale") else ""))
    render_doc = {k: v for k, v in doc.items() if k != "rationale"}
    out.decorator = render_decorator(render_doc, info.arg_names)
    out.toml = render_toml_target(render_doc, target, name=info.qualname.split(".")[-1])
    out.python_spec = render_python_spec(render_doc, target, info.arg_names, header=_wrap_comment(header))
    return out


def _wrap_comment(text: str, width: int = 88) -> str:
    import textwrap  # noqa: PLC0415

    return "\n".join(textwrap.wrap(text.replace('"""', "'''"), width))
