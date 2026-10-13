"""Command-line interface: ``numguard check`` and ``numguard eval``."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import click

from . import __version__
from .api import check, get_spec
from .cases import FLOAT_DTYPES, KINDS
from .corpus import BASELINES, list_tasks, load_task, run_eval
from .corpus import corpus_dir as find_corpus_dir
from .properties import PROPERTY_FACTORIES

__all__ = ["main", "resolve_callable"]


def resolve_callable(target: str) -> Callable[..., Any]:
    """Resolve ``module:attr``, ``module.attr`` or ``path/to/file.py:attr``.

    The current working directory is put on ``sys.path`` so local modules can
    be referenced, as with ``python -m``.
    """
    if ":" in target and not (len(target) > 1 and target[1] == ":" and target.count(":") == 1):
        mod_name, _, attr = target.rpartition(":")
    elif "." in target:
        mod_name, _, attr = target.rpartition(".")
    else:
        raise click.BadParameter(f"expected 'module:function', got {target!r}")
    if mod_name.endswith(".py") or "/" in mod_name or "\\" in mod_name:
        path = Path(mod_name)
        if not path.is_file():
            raise click.BadParameter(f"file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise click.BadParameter(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(path.stem, module)
        spec.loader.exec_module(module)
    else:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(mod_name)
        except ImportError as exc:
            raise click.BadParameter(f"cannot import module {mod_name!r}: {exc}") from exc
    obj: Any = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise click.BadParameter(f"{mod_name!r} has no attribute {attr!r}") from exc
    if not callable(obj):
        raise click.BadParameter(f"{target!r} is not callable")
    return obj


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="numguard")
def main() -> None:
    """numguard: numerical edge-case testing for array code."""


@main.command("check")
@click.argument("target")
@click.option("--ref", "reference", metavar="MODULE:FUNC",
              help="Reference implementation to compare against.")
@click.option("--dtype", "dtypes", multiple=True, type=click.Choice(FLOAT_DTYPES),
              help="Floating dtype to test (repeatable). Default: float32, float64.")
@click.option("--prop", "properties", multiple=True, metavar="NAME[:VALUE]",
              help="Metamorphic property (repeatable): "
                   + ", ".join(sorted(PROPERTY_FACTORIES)) + ".")
@click.option("--domain", default=None,
              help="Input domain for every argument: any, positive, nonneg, unit_interval, "
                   "'[a, b)', 'int[0, 255]'. Default: any.")
@click.option("--nan", "nan", type=click.Choice(["propagate", "raise", "ignore"]), default=None,
              help="Expected NaN behaviour (default: propagate).")
@click.option("--batched", is_flag=True, default=None,
              help="Check that batching along axis 0 does not change results.")
@click.option("--monotonic", type=click.Choice(["increasing", "decreasing"]), default=None)
@click.option("--fp-signals", type=click.Choice(["ignore", "fail"]), default=None,
              help="Fail on overflow/invalid/divide-by-zero signals for finite inputs.")
@click.option("--rtol", type=float, default=None)
@click.option("--atol", type=float, default=None)
@click.option("--large-n", type=int, default=None, help="Size of the large_n case.")
@click.option("--seed", type=int, default=None)
@click.option("--exclude", multiple=True, help="Case name/kind/tag (glob) to skip; repeatable.")
@click.option("--allow-nonfinite", multiple=True,
              help="Case name/kind/tag allowed to produce inf/NaN; repeatable.")
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
@click.option("--max-rows", type=int, default=25, show_default=True)
def check_cmd(target: str, reference: str | None, dtypes: tuple[str, ...],
              properties: tuple[str, ...], domain: str | None, nan: str | None,
              batched: bool | None, monotonic: str | None, fp_signals: str | None,
              rtol: float | None, atol: float | None, large_n: int | None, seed: int | None,
              exclude: tuple[str, ...], allow_nonfinite: tuple[str, ...], as_json: bool,
              max_rows: int) -> None:
    """Check TARGET (``module:function``) against numguard's edge cases.

    If TARGET is decorated with ``@numguard.guard``, its attached spec is used
    and command-line options override it. Exits with status 1 on failure.
    """
    fn = resolve_callable(target)
    ref = resolve_callable(reference) if reference else None
    options: dict[str, Any] = {
        "nan": nan, "batched": batched, "monotonic": monotonic, "fp_signals": fp_signals,
        "large_n": large_n, "seed": seed,
    }
    if exclude:
        options["exclude"] = tuple(exclude)
    if allow_nonfinite:
        options["allow_nonfinite"] = tuple(allow_nonfinite)
    options = {k: v for k, v in options.items() if v is not None}
    try:
        spec = get_spec(fn)
        if spec is not None:
            report = check(fn, spec=spec, reference=ref, properties=list(properties),
                           dtypes=list(dtypes) or None, rtol=rtol, atol=atol, domain=domain,
                           name=target, **options)
        else:
            report = check(fn, reference=ref, properties=list(properties),
                           dtypes=list(dtypes) or None, rtol=rtol, atol=atol, domain=domain,
                           name=target, **options)
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(report.summary(max_rows=max_rows))
    sys.exit(0 if report.ok else 1)


@main.command("eval")
@click.option("--candidate-dir", type=click.Path(file_okay=False, path_type=Path),
              help="Directory with <task>/solution.py files to grade.")
@click.option("--baseline", type=click.Choice(BASELINES),
              help="Grade the bundled buggy or reference implementations.")
@click.option("--task", "tasks", multiple=True, help="Only these tasks (repeatable).")
@click.option("--corpus-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Use a different corpus directory.")
@click.option("--solution-name", default="solution.py", show_default=True,
              help="File name of the candidate implementation inside each task folder.")
@click.option("--json", "as_json", is_flag=True, help="Print only JSON.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path),
              help="Also write the JSON results (with full reports) to this file.")
@click.option("--expect", type=click.Choice(["pass", "fail"]), default=None,
              help="Exit 0 only if every task passes (pass) or every task fails (fail). "
                   "Default: pass for candidates and --baseline reference, fail for "
                   "--baseline buggy.")
def eval_cmd(candidate_dir: Path | None, baseline: str | None, tasks: tuple[str, ...],
             corpus_dir: Path | None, solution_name: str, as_json: bool,
             json_out: Path | None, expect: str | None) -> None:
    """Grade implementations of the bundled bug-corpus tasks.

    Agent-agnostic: give each task's ``task.md`` to any coding agent, save its
    answer as ``DIR/<task>/solution.py`` and run
    ``numguard eval --candidate-dir DIR``.
    """
    if (candidate_dir is None) == (baseline is None):
        raise click.UsageError("give exactly one of --candidate-dir and --baseline")
    known = list_tasks(corpus_dir)
    for t in tasks:
        if t not in known:
            raise click.UsageError(f"unknown task {t!r}; known tasks: {', '.join(known)}")
    result = run_eval(candidate_dir=candidate_dir, baseline=baseline, tasks=list(tasks) or None,
                      corpus=corpus_dir, solution_name=solution_name)
    if json_out is not None:
        json_out.write_text(result.to_json(include_reports=True), encoding="utf-8")
    if as_json:
        click.echo(result.to_json())
    else:
        click.echo(result.scoreboard())
        click.echo("")
        click.echo(result.to_json(indent=None))
    if expect is None:
        expect = "fail" if baseline == "buggy" else "pass"
    if expect == "pass":
        ok = result.n_pass == result.n_tasks
    else:
        ok = result.n_pass == 0 and all(r.status == "fail" for r in result.results)
    sys.exit(0 if ok else 1)


@main.command("tasks")
@click.option("--corpus-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--show", metavar="TASK", default=None, help="Print one task's prompt.")
def tasks_cmd(corpus_dir: Path | None, show: str | None) -> None:
    """List the corpus tasks (or print one task's prompt)."""
    if show:
        click.echo(load_task(show, corpus_dir).prompt)
        return
    root = find_corpus_dir(corpus_dir)
    click.echo(f"corpus: {root}")
    for name in list_tasks(corpus_dir):
        task = load_task(name, corpus_dir)
        click.echo(f"  {name:<20} {task.function:<20} {task.title()}")


@main.command("cases")
def cases_cmd() -> None:
    """List the edge-case kinds numguard generates."""
    from .cases import _kind_tags  # noqa: PLC0415 - internal helper

    for kind in KINDS:
        tags = sorted(_kind_tags(kind) - {kind})
        click.echo(f"  {kind:<16} {', '.join(tags)}")


if __name__ == "__main__":  # pragma: no cover
    main()
