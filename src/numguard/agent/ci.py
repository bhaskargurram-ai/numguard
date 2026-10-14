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
from .prompts import FIX_SYSTEM, SUBMIT_REWRITE, VARIANT_HINTS, failures_table, fix_message, spec_text, untrusted_rule
from .sandbox import run_worker, static_check
from .session import Session, new_nonce
from .tools import RepoReader, repo_tools, static_tool

__all__ = ["MANIFEST_SCHEMA", "ManifestError", "propose", "verify", "load_manifest"]

MANIFEST_SCHEMA = "numguard-candidates"
MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 1_000_000
MAX_CANDIDATE_BYTES = 100_000
MAX_CANDIDATES = 50
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
            max_targets: int | None = None, max_tool_calls: int | None = None, variants: int | None = None,
            log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Ask the model for rewrites of the failing targets; write candidates + manifest. Executes nothing.

    Nothing can be executed here, so instead of refinement rounds the model makes ``variants`` independent
    attempts per target (each in a fresh session); the verify job runs all of them and keeps the best.
    """
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
    n_variants = max(1, min(variants or settings.variants, MAX_CANDIDATES // max(1, len(targets))))
    k = 0
    for doc in targets:
        try:
            info = target_info(doc["function"], repo, allow_import=False)
        except Exception as exc:  # noqa: BLE001
            manifest["candidates"].append({"target": doc["name"], "function": doc["function"], "file": None,
                                           "status": "error", "rationale": "",
                                           "reasons": [f"source not found without importing: {exc}"[:400]]})
            continue
        groups = results_by_name[doc["name"]].get("failures", [])
        spec_req = {"source": "doc", "doc": {k: v for k, v in doc.items() if k not in ("name", "function")}}
        for v in range(n_variants):
            entry: dict[str, Any] = {"target": doc["name"], "function": doc["function"], "file": None,
                                     "status": "error", "rationale": "", "reasons": []}
            manifest["candidates"].append(entry)
            nonce = new_nonce()
            tools = [static_tool("show_failures", "The failure table of the function (from the check job).",
                                 lambda g=groups: failures_table(g))] + repo_tools(RepoReader(repo))
            session = Session(llm, FIX_SYSTEM.format(untrusted=untrusted_rule(nonce)), tools, SUBMIT_REWRITE,
                              rewrite_validator(info.qualname),
                              max_tool_calls=max_tool_calls or settings.max_tool_calls, nonce=nonce, log=log)
            log(f"agent propose: {doc['name']} (variant {v + 1}/{n_variants}) ...")
            try:
                res = session.run(fix_message(doc["function"], f"{info.rel_path}:{info.lineno}", info.source,
                                              spec_text(spec_req), failures_table(groups), nonce,
                                              hint=VARIANT_HINTS[v % len(VARIANT_HINTS)]))
            except Exception as exc:  # noqa: BLE001 - one failed variant must not lose the others
                entry["reasons"] = [f"the model call failed: {exc}"[:400]]
                continue
            finally:
                total.add(session.usage)
            if res.status == "submitted":
                name = f"cand_{k}.py"
                k += 1
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
    if not isinstance(cands, list) or len(cands) > MAX_CANDIDATES:
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
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in manifest["candidates"]:
        groups.setdefault(c["target"], []).append(c)
    for target, cands in groups.items():
        fixes.append(_verify_target(target, cands, by_name.get(target), manifest["model"], repo, cand_dir, tmo,
                                    log).fix_entry())
    doc = dict(result)
    doc["fixes"] = fixes
    return validate_result(doc)


def _verify_target(target: str, cands: list[dict[str, Any]], doc: Mapping[str, Any] | None, model: str, repo: Path,
                   cand_dir: str | Path, tmo: float, log: Callable[[str], None]) -> FixOutcome:
    """Run every variant proposed for one target; the outcome's best attempt is its last."""
    fo = FixOutcome(target=target, status="error", model=model)
    if doc is None:
        fo.error = "the target is not in the configuration"
        return fo
    submitted = [c for c in cands if c["status"] == "submitted"]
    if not submitted:
        last = cands[-1]
        fo.status = last["status"]
        fo.attempts = [Attempt(i + 1, c["status"] if c["status"] == "refused" else "error", "", reasons=c["reasons"])
                       for i, c in enumerate(cands)]
        return fo
    try:
        info = target_info(doc["function"], repo, allow_import=False)
    except Exception as exc:  # noqa: BLE001
        fo.error = f"source not found: {exc}"[:400]
        return fo
    spec_doc = {k: v for k, v in doc.items() if k not in ("name", "function")}
    base_req = {"root": str(repo), "target": doc["function"], "name": doc["name"],
                "spec": {"source": "doc", "doc": spec_doc}}
    log(f"agent verify: {target}: checking the original ...")
    wb = run_worker(base_req, timeout=tmo, cwd=repo)
    if not wb.ok:
        fo.error = f"could not check the original: {wb.error}"
        return fo
    fo.before = wb.data
    attempts: list[Attempt] = []
    for i, c in enumerate(cands):
        if c["status"] != "submitted":
            attempts.append(Attempt(i + 1, c["status"] if c["status"] == "refused" else "error", "",
                                    reasons=c["reasons"]))
            continue
        code = (Path(cand_dir) / c["file"]).read_text(encoding="utf-8")
        problems = static_check(code, info.qualname)
        if problems:  # never trust the propose job's static check: repeat it
            attempts.append(Attempt(i + 1, "refused", code, c["rationale"], ["static check: " + p for p in problems]))
            continue
        log(f"agent verify: {target}: running variant {i + 1}/{len(cands)} ...")
        wr = run_worker({**base_req, "candidate": {"code": code, "name": info.qualname}, "compare_original": True},
                        timeout=tmo, cwd=repo)
        ok, reasons = evaluate_candidate(wb.data, wr)
        attempts.append(Attempt(i + 1, "verified" if ok else ("error" if not wr.ok else "rejected"), code,
                                c["rationale"], reasons, wr.data.get("n_failures") if wr.ok else None,
                                wr.data if wr.ok else {}))
    best = min(attempts, key=_attempt_rank)
    attempts.remove(best)
    fo.attempts = attempts + [best]  # FixOutcome reports its last (or first verified) attempt
    fo.status = best.status
    if best.code:
        fo.diff, _ = _function_diff(info, best.code)
    return fo


_RANK = {"verified": 0, "rejected": 1, "error": 2, "refused": 3}


def _attempt_rank(a: Attempt) -> tuple[int, float, int]:
    """Verified first; then fewest failing results; then the earliest variant."""
    nf = a.n_failures if a.n_failures is not None else float("inf")
    return (_RANK.get(a.status, 4), nf, a.round)
