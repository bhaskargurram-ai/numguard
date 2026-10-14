"""The agent split for CI: ``numguard agent propose`` and ``numguard agent verify``.

* **propose** holds the LLM key and runs **no repository code**: it reads
  ``numguard.toml`` (TOML, data), the check job's ``result.json`` (validated,
  untrusted), and the failing functions' source through the file system and
  the AST only. The model's rewrites are checked statically and written,
  with a manifest, to a candidates directory. Nothing is executed.
* **verify** has **no key**: it re-checks every candidate statically, runs
  the original and the candidate in the subprocess worker against the
  target's spec, applies the verification criteria of :mod:`numguard.agent.fix`,
  and merges the outcome into ``result.json`` (``fixes``) and the card.

In GitHub Actions these run in separate jobs (separate VMs), so code that ran
in ``verify`` can never reach the key used in ``propose``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import __version__
from ..config import Config
from ..result import validate_result
from .common import target_info, usage_dict
from .fix import Attempt, FixOutcome, _function_diff, evaluate_candidate, rewrite_validator
from .llm import ToolLLM
from .prompts import FIX_SYSTEM, SUBMIT_REWRITE, failures_table, fix_message, spec_text, untrusted_rule
from .sandbox import run_worker, static_check
from .session import Session, new_nonce
from .tools import RepoReader, repo_tools, static_tool

__all__ = ["MANIFEST_SCHEMA", "ManifestError", "propose", "verify", "load_manifest"]

MANIFEST_SCHEMA = "numguard-candidates"
MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 1_000_000
MAX_CANDIDATE_BYTES = 100_000
_FILE_RE = re.compile(r"^cand_[0-9]{1,3}\.py$")
_CAND_STATUSES = ("submitted", "refused", "no_submission", "error")


class ManifestError(ValueError):
    pass


def failing_targets(config: Config, result: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    by_name = {t["name"]: t for t in config.targets}
    out = []
    for t in result.get("targets") or []:
        if t["status"] == "fail" and t["name"] in by_name:
            out.append(by_name[t["name"]])
    return out[:limit]


def propose(config: Config, result: Mapping[str, Any], repo: str | Path, out_dir: str | Path, llm: ToolLLM, *,
            max_targets: int | None = None, max_tool_calls: int | None = None,
            log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Ask the model for rewrites of the failing targets; write candidates + manifest. Executes nothing."""
    repo = Path(repo).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = config.agent
    limit = max_targets or settings.max_targets
    results_by_name = {t["name"]: t for t in result.get("targets") or []}
    manifest: dict[str, Any] = {"schema": MANIFEST_SCHEMA, "version": MANIFEST_VERSION, "tool_version": __version__,
                                "model": llm.model, "provider": llm.provider, "head": result.get("head"),
                                "candidates": [], "notes": []}
    from .llm import Usage  # noqa: PLC0415

    total = Usage()
    targets = failing_targets(config, result, limit)
    if not targets:
        manifest["notes"].append("no failing targets to fix")
    for i, doc in enumerate(targets):
        entry: dict[str, Any] = {"target": doc["name"], "function": doc["function"], "file": None,
                                 "status": "error", "rationale": "", "reasons": []}
        manifest["candidates"].append(entry)
        try:
            info = target_info(doc["function"], repo, allow_import=False)
        except Exception as exc:  # noqa: BLE001
            entry["reasons"] = [f"source not found without importing: {exc}"[:400]]
            continue
        nonce = new_nonce()
        groups = results_by_name[doc["name"]].get("failures", [])
        tools = [static_tool("show_failures", "The failure table of the function (from the check job).",
                             lambda g=groups: failures_table(g))] + repo_tools(RepoReader(repo))
        session = Session(llm, FIX_SYSTEM.format(untrusted=untrusted_rule(nonce)), tools, SUBMIT_REWRITE,
                          rewrite_validator(info.qualname), max_tool_calls=max_tool_calls or settings.max_tool_calls,
                          nonce=nonce, log=log)
        spec_req = {"source": "doc", "doc": {k: v for k, v in doc.items() if k not in ("name", "function")}}
        log(f"agent propose: {doc['name']} ...")
        res = session.run(fix_message(doc["function"], f"{info.rel_path}:{info.lineno}", info.source,
                                      spec_text(spec_req), failures_table(groups), nonce))
        total.add(session.usage)
        if res.status == "submitted":
            name = f"cand_{i}.py"
            (out / name).write_text(res.value["code"], encoding="utf-8")
            entry.update(file=name, status="submitted", rationale=res.value["rationale"][:2000])
        elif res.rejections:
            entry.update(status="refused", reasons=[r[:400] for r in res.rejections[-1]][:10])
        else:
            entry.update(status="no_submission" if res.status == "no_submission" else "error",
                         reasons=[res.error[:400]] if res.error else [])
    manifest["usage"] = usage_dict(total)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _s(v: Any, path: str, limit: int, optional: bool = False) -> Any:
    if v is None and optional:
        return None
    if not isinstance(v, str) or len(v) > limit:
        raise ManifestError(f"{path} must be a string of at most {limit} characters")
    return v


def load_manifest(cand_dir: str | Path) -> dict[str, Any]:
    """Load and strictly validate a candidates manifest (written by propose, an LLM-facing job)."""
    p = Path(cand_dir) / "manifest.json"
    if not p.is_file():
        raise ManifestError(f"{p} not found")
    if p.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest.json is too large")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(d, dict) or d.get("schema") != MANIFEST_SCHEMA or d.get("version") != MANIFEST_VERSION:
        raise ManifestError("not a numguard candidates manifest")
    cands = d.get("candidates")
    if not isinstance(cands, list) or len(cands) > 50:
        raise ManifestError("candidates must be a list of at most 50 entries")
    out = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            raise ManifestError(f"candidates[{i}] must be an object")
        status = c.get("status")
        if status not in _CAND_STATUSES:
            raise ManifestError(f"candidates[{i}].status is invalid")
        f = c.get("file")
        if status == "submitted":
            if not isinstance(f, str) or not _FILE_RE.match(f):
                raise ManifestError(f"candidates[{i}].file must be cand_<n>.py")
            fp = Path(cand_dir) / f
            if fp.is_symlink() or not fp.is_file() or fp.stat().st_size > MAX_CANDIDATE_BYTES:
                raise ManifestError(f"candidates[{i}].file is missing, a symlink or too large")
        reasons = c.get("reasons") or []
        if not isinstance(reasons, list) or len(reasons) > 10:
            raise ManifestError(f"candidates[{i}].reasons must be a short list")
        out.append({"target": _s(c.get("target"), f"candidates[{i}].target", 80),
                    "status": status, "file": f if status == "submitted" else None,
                    "rationale": _s(c.get("rationale") or "", f"candidates[{i}].rationale", 2000),
                    "reasons": [_s(r, f"candidates[{i}].reasons[]", 400) for r in reasons]})
    model = _s(d.get("model") or "", "model", 100)
    return {"model": model, "candidates": out}


def verify(cand_dir: str | Path, config: Config, result: Mapping[str, Any], repo: str | Path, *,
           timeout: float | None = None, log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Execute and verify the candidates; return the updated (validated) result document."""
    repo = Path(repo).resolve()
    manifest = load_manifest(cand_dir)
    by_name = {t["name"]: t for t in config.targets}
    tmo = timeout or config.agent.timeout
    fixes = []
    for c in manifest["candidates"]:
        fo = FixOutcome(target=c["target"], status=c["status"], model=manifest["model"])
        doc = by_name.get(c["target"])
        if doc is None:
            fo.status, fo.error = "error", "the target is not in the configuration"
            fixes.append(fo.fix_entry())
            continue
        if c["status"] != "submitted":
            fo.attempts.append(Attempt(1, c["status"] if c["status"] == "refused" else "error", "",
                                       reasons=c["reasons"]))
            fo.status = c["status"]
            fixes.append(fo.fix_entry())
            continue
        code = (Path(cand_dir) / c["file"]).read_text(encoding="utf-8")
        try:
            info = target_info(doc["function"], repo, allow_import=False)
        except Exception as exc:  # noqa: BLE001
            fo.status, fo.error = "error", f"source not found: {exc}"[:400]
            fixes.append(fo.fix_entry())
            continue
        problems = static_check(code, info.qualname)
        if problems:  # never trust the propose job's static check: repeat it
            fo.status = "refused"
            fo.attempts.append(Attempt(1, "refused", code, c["rationale"], ["static check: " + p for p in problems]))
            fixes.append(fo.fix_entry())
            continue
        spec_doc = {k: v for k, v in doc.items() if k not in ("name", "function")}
        base_req = {"root": str(repo), "target": doc["function"], "name": doc["name"],
                    "spec": {"source": "doc", "doc": spec_doc}}
        log(f"agent verify: {doc['name']}: checking the original ...")
        wb = run_worker(base_req, timeout=tmo, cwd=repo)
        if not wb.ok:
            fo.status, fo.error = "error", f"could not check the original: {wb.error}"
            fixes.append(fo.fix_entry())
            continue
        fo.before = wb.data
        log(f"agent verify: {doc['name']}: running the candidate ...")
        wr = run_worker({**base_req, "candidate": {"code": code, "name": info.qualname}, "compare_original": True},
                        timeout=tmo, cwd=repo)
        ok, reasons = evaluate_candidate(wb.data, wr)
        fo.attempts.append(Attempt(1, "verified" if ok else ("error" if not wr.ok else "rejected"), code,
                                   c["rationale"], reasons, wr.data.get("n_failures") if wr.ok else None,
                                   wr.data if wr.ok else {}))
        fo.status = "verified" if ok else ("error" if not wr.ok else "rejected")
        fo.diff, _ = _function_diff(info, code)
        fixes.append(fo.fix_entry())
    doc = dict(result)
    doc["fixes"] = fixes
    return validate_result(doc)
