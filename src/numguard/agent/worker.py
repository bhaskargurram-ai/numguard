"""Subprocess worker: run numguard checks on a function or on a model-written rewrite.

Invoked as ``python -m numguard.agent.worker REQUEST.json RESPONSE.json`` by
:func:`numguard.agent.sandbox.run_worker`, never imported by the parent
process's agent code paths. Request::

    {
      "root": "/path/to/repo",            # working directory, put on sys.path
      "target": "pkg.mod:func",           # the original function
      "spec": {"source": "doc" | "guard" | "pyfile", "doc": {...}, "pyfile": "spec.py",
               "overrides": {...}},       # e.g. {"large_n": 10000}
      "candidate": null | {"code": "...", "name": "func"},
      "compare_original": false,          # run the original on benign cases and compare
      "max_failures": 2000
    }

Response: ``{"ok": true, "summary": ..., "groups": [...], "failures": [[check, case, dtype, message]...],
"n_results": ..., "signature": {...}, "agreement": {...}, "shapes": {...}}`` or
``{"ok": false, "stage": ..., "error": ...}``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import linecache
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _short(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}".rstrip(": ")[:500]


def _sig(fn: Callable[..., Any]) -> str:
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return "(?)"


def _load_candidate(original: Callable[..., Any], code: str, name: str) -> Callable[..., Any]:
    if "." in getattr(original, "__qualname__", name).replace("<locals>", ""):
        raise ValueError("only module-level functions can be rewritten")
    filename = "<numguard-candidate>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    ns = dict(getattr(original, "__globals__", {}))
    ns.pop(name, None)  # the rewrite must define the function itself
    exec(compile(code, filename, "exec"), ns)  # noqa: S102 - statically checked, runs in this subprocess only
    fn = ns.get(name)
    if not callable(fn):
        raise ValueError(f"the rewrite does not define {name!r}")
    return fn


def _build_spec(req: dict[str, Any], original: Callable[..., Any], root: Path) -> Any:
    from numguard.api import get_spec, make_spec  # noqa: PLC0415
    from numguard.runconfig import build_check_kwargs  # noqa: PLC0415
    from numguard.specdoc import TARGET_KEYS, validate_spec_doc  # noqa: PLC0415

    spec_req = req.get("spec") or {}
    source = spec_req.get("source", "doc")
    overrides = dict(spec_req.get("overrides") or {})
    if source == "guard":
        spec = get_spec(original)
        if spec is None:
            raise ValueError("the function has no @numguard.guard spec")
        return spec.replace(**overrides) if overrides else spec
    if source == "pyfile":
        path = Path(spec_req["pyfile"])
        if not path.is_absolute():
            path = root / path
        mspec = importlib.util.spec_from_file_location("_numguard_specfile", path)
        if mspec is None or mspec.loader is None:
            raise ValueError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        kwargs = dict(getattr(mod, "SPEC"))
        kwargs.update(overrides)
        return make_spec(original, **kwargs)
    doc = validate_spec_doc(spec_req.get("doc") or {}, allowed=TARGET_KEYS)
    kwargs = build_check_kwargs(doc, original, root)
    kwargs.update(overrides)
    kwargs.pop("name", None)
    return make_spec(original, **kwargs)


def _compare_with_original(original: Callable[..., Any], candidate: Callable[..., Any], spec: Any) -> dict[str, Any]:
    from numguard.backends import BackendCall, wrap  # noqa: PLC0415
    from numguard.checks import _compare_outputs, call  # noqa: PLC0415
    from numguard.dtypes import dtype_name  # noqa: PLC0415

    cases = spec.cases()
    mism: list[dict[str, str]] = []
    shapes: list[dict[str, str]] = []
    checked = 0
    fo = wrap(original, spec.backend, spec.dtypes)
    fc = wrap(candidate, spec.backend, spec.dtypes)
    for case in cases:
        for f in (fo, fc):
            if isinstance(f, BackendCall):
                f.set_case([dtype_name(s.dtype) if s.dtype is not None else case.dtype for s in spec.inputs])
        ro = call(fo, case.args, spec.kwargs, check_mutation=False)
        rc = call(fc, case.args, spec.kwargs, check_mutation=False)
        if not (ro.ok and rc.ok):
            continue
        got = [f"{o.shape}/{n}" for o, n in zip(rc.outputs or [], rc.out_dtypes)]
        want = [f"{o.shape}/{n}" for o, n in zip(ro.outputs or [], ro.out_dtypes)]
        if got != want:
            shapes.append({"case": case.name, "dtype": case.dtype,
                           "message": f"rewrite gives {', '.join(got)}; original gives {', '.join(want)}"})
        if not (case.regular and case.finite_inputs):
            continue
        checked += 1
        with np.errstate(all="ignore"):
            c = _compare_outputs(rc.outputs or [], ro.outputs or [], case, spec, 4.0, "original")
        if not c.ok:
            mism.append({"case": case.name, "dtype": case.dtype, "message": c.message[:300]})
    return {"agreement": {"checked": checked, "mismatches": mism[:200]},
            "shapes": {"mismatches": shapes[:200]}}


def handle(req: dict[str, Any]) -> dict[str, Any]:
    from numguard.api import check_spec  # noqa: PLC0415
    from numguard.resolve import resolve_target  # noqa: PLC0415

    root = Path(req.get("root") or ".").resolve()
    for p in [*(req.get("sys_path") or []), str(root / "src"), str(root)]:
        if p not in sys.path and Path(p).is_dir():
            sys.path.insert(0, p)
    stage = "import"
    try:
        original = resolve_target(req["target"], root)
        stage = "spec"
        spec = _build_spec(req, original, root)
        fn = original
        out: dict[str, Any] = {"signature": {"original": _sig(original)}}
        cand = req.get("candidate")
        if cand:
            stage = "candidate"
            fn = _load_candidate(original, cand["code"], cand["name"])
            out["signature"]["candidate"] = _sig(fn)
            out["signature"]["match"] = out["signature"]["candidate"] == out["signature"]["original"]
        stage = "check"
        report = check_spec(fn, spec, name=req.get("name") or req["target"])
        max_f = int(req.get("max_failures", 2000))
        benign = {c.name for c in spec.cases() if c.regular and c.finite_inputs}
        out.update({
            "ok": True,
            "summary": report.summary(max_rows=30),
            "groups": report.grouped_failures()[:60],
            "failures": [[r.check, r.case, r.dtype, r.message[:300]] for r in report.failures][:max_f],
            "n_results": len(report.results),
            "n_failures": len(report.failures),
            "n_cases": report.n_cases,
            "dtypes": report.dtypes,
            "reference": report.reference,
            "has_reference": spec.reference is not None,
            "regular_reference_failures": sorted({r.case for r in report.failures
                                                  if r.check == "reference" and r.case in benign}),
        })
        if cand and req.get("compare_original"):
            stage = "compare"
            out.update(_compare_with_original(original, fn, spec))
        return out
    except Exception as exc:  # noqa: BLE001 - report every failure to the parent
        return {"ok": False, "stage": stage, "error": _short(exc),
                "traceback": traceback.format_exc()[-2000:]}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m numguard.agent.worker REQUEST.json RESPONSE.json", file=sys.stderr)
        return 2
    req = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    resp = handle(req)
    Path(argv[2]).write_text(json.dumps(resp, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    sys.exit(main(sys.argv))
