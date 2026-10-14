"""Command-line interface.

Deterministic commands: ``check`` (one function, or every target of a
``numguard.toml``), ``eval``, ``tasks``, ``cases``, ``summary``, ``comment``
and ``init``. Agent commands (opt-in, bring your own LLM key): ``suggest``,
``fix``, ``bench`` and ``agent propose`` / ``agent verify`` for CI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import click

from . import __version__
from .api import check, get_spec
from .backends import BACKENDS, BackendError
from .cases import KINDS
from .corpus import BASELINES, list_tasks, load_task, run_eval
from .corpus import corpus_dir as find_corpus_dir
from .dtypes import ALL_FLOAT_DTYPES
from .properties import PROPERTY_FACTORIES
from .resolve import TargetError, resolve_target

__all__ = ["main", "resolve_callable"]


def resolve_callable(target: str) -> Callable[..., Any]:
    """Resolve ``module:attr``, ``module.attr`` or ``path/to/file.py:attr``.

    The current working directory is put on ``sys.path`` so local modules can
    be referenced, as with ``python -m``.
    """
    try:
        return resolve_target(target)
    except TargetError as exc:
        raise click.BadParameter(str(exc)) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="numguard")
def main() -> None:
    """numguard: numerical edge-case testing for array code."""


@main.command("check")
@click.argument("target", required=False)
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Check every [[target]] of this numguard.toml (default when TARGET is omitted: "
                   "./numguard.toml).")
@click.option("--target", "target_names", multiple=True, metavar="NAME",
              help="With --config: only these targets (repeatable; pytest is then skipped).")
@click.option("--ref", "reference", metavar="MODULE:FUNC",
              help="Reference implementation to compare against.")
@click.option("--dtype", "dtypes", multiple=True, type=click.Choice(ALL_FLOAT_DTYPES),
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
@click.option("--backend", type=click.Choice(BACKENDS), default=None,
              help="Array library the function expects (inputs are converted). Default: numpy.")
@click.option("--json", "as_json", is_flag=True, help="Print the report (or result document) as JSON.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write the result document (result.json, used by the GitHub workflows) here.")
@click.option("--card-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write the Markdown card here.")
@click.option("--no-pytest", is_flag=True, help="With --config: skip the [pytest] run.")
@click.option("--pr-number", default=None, help="Metadata for result.json (CI).")
@click.option("--head-sha", default=None, help="Metadata for result.json (CI).")
@click.option("--base-sha", default=None, help="Metadata for result.json (CI).")
@click.option("--fail-on", type=click.Choice(["fail", "error", "never"]), default="fail", show_default=True,
              help="Exit 1 when a target fails or errors (fail), only on errors (error), or never.")
@click.option("--max-rows", type=int, default=25, show_default=True)
def check_cmd(target: str | None, config_path: Path | None, target_names: tuple[str, ...],
              reference: str | None, dtypes: tuple[str, ...],
              properties: tuple[str, ...], domain: str | None, nan: str | None,
              batched: bool | None, monotonic: str | None, fp_signals: str | None,
              rtol: float | None, atol: float | None, large_n: int | None, seed: int | None,
              exclude: tuple[str, ...], allow_nonfinite: tuple[str, ...], backend: str | None,
              as_json: bool, json_out: Path | None, card_out: Path | None, no_pytest: bool,
              pr_number: str | None, head_sha: str | None, base_sha: str | None, fail_on: str,
              max_rows: int) -> None:
    """Check TARGET (``module:function``), or every target of a numguard.toml.

    If TARGET is decorated with ``@numguard.guard``, its attached spec is used
    and command-line options override it. Without TARGET, the targets of
    ``--config`` (default ``./numguard.toml``) are checked, plus the
    ``[pytest]`` run if configured. Exits with status 1 on failure.
    """
    from .config import find_config  # noqa: PLC0415

    if target is None:
        if config_path is None:
            config_path = find_config()
        if config_path is None:
            raise click.UsageError("give a TARGET (module:function) or --config numguard.toml "
                                   "(no numguard.toml in the current directory)")
        overrides = {"large_n": large_n, "seed": seed, "backend": backend}
        if dtypes:
            overrides["dtypes"] = list(dtypes)
        _check_config(config_path, target_names, no_pytest, as_json, json_out, card_out, pr_number,
                      head_sha, base_sha, fail_on, max_rows, overrides)
        return
    if config_path is not None or target_names:
        raise click.UsageError("give either TARGET or --config/--target, not both")
    fn = resolve_callable(target)
    ref = resolve_callable(reference) if reference else None
    options: dict[str, Any] = {
        "nan": nan, "batched": batched, "monotonic": monotonic, "fp_signals": fp_signals,
        "large_n": large_n, "seed": seed, "backend": backend,
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
    except (TypeError, ValueError, BackendError) as exc:
        raise click.UsageError(str(exc)) from exc
    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(report.summary(max_rows=max_rows))
    if json_out is not None or card_out is not None:
        from .runconfig import ConfigRun, TargetOutcome  # noqa: PLC0415

        run = ConfigRun([TargetOutcome(target, target, "pass" if report.ok else "fail", report=report,
                                       duration_s=report.duration_s)])
        _write_outputs(run, json_out, card_out, None, pr_number, head_sha, base_sha)
    sys.exit(0 if report.ok or fail_on in ("never", "error") else 1)


def _meta(pr_number: str | None, head_sha: str | None, base_sha: str | None) -> dict[str, Any]:
    import re  # noqa: PLC0415

    out: dict[str, Any] = {"pr_number": None, "head": None, "base": None}
    if pr_number and re.fullmatch(r"[0-9]{1,10}", pr_number.strip()):
        out["pr_number"] = int(pr_number) or None
    for key, val in (("head", head_sha), ("base", base_sha)):
        if val and re.fullmatch(r"[0-9a-f]{7,64}", val.strip().lower()):
            out[key] = val.strip().lower()
    return out


def _write_outputs(run: Any, json_out: Path | None, card_out: Path | None, config_path: str | None,
                   pr_number: str | None, head_sha: str | None, base_sha: str | None) -> dict[str, Any]:
    from .card import render_card  # noqa: PLC0415
    from .result import build_result, validate_result  # noqa: PLC0415

    doc = build_result(run, config_path=config_path, **_meta(pr_number, head_sha, base_sha))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if card_out is not None:
        card_out.parent.mkdir(parents=True, exist_ok=True)
        card_out.write_text(render_card(validate_result(doc)), encoding="utf-8")
    return doc


def _check_config(config_path: Path, names: tuple[str, ...], no_pytest: bool, as_json: bool,
                  json_out: Path | None, card_out: Path | None, pr_number: str | None, head_sha: str | None,
                  base_sha: str | None, fail_on: str, max_rows: int, overrides: dict[str, Any]) -> None:
    from .config import ConfigError, load_config  # noqa: PLC0415
    from .runconfig import run_config  # noqa: PLC0415

    try:
        config = load_config(config_path)
        for n in names:
            config.target(n)
    except ConfigError as exc:
        raise click.UsageError(f"{config_path}: {exc}") from exc
    log = (lambda m: click.echo(m, err=True)) if not as_json else (lambda m: None)
    run = run_config(config, list(names) or None, run_pytest_too=not no_pytest, log=log, **overrides)
    try:
        rel = os.path.relpath(config.path or config_path)
    except ValueError:  # pragma: no cover - different drive on Windows
        rel = str(config_path)
    doc = _write_outputs(run, json_out, card_out, rel, pr_number, head_sha, base_sha)
    if as_json:
        click.echo(json.dumps(doc, indent=2))
    else:
        for t in run.targets:
            if t.report is not None:
                click.echo(t.report.summary(max_rows=max_rows))
            else:
                click.echo(f"numguard: {t.name} ERROR - {t.error}")
        if run.pytest is not None:
            p = run.pytest
            click.echo(f"pytest: {p.status.upper()} - {p.passed} passed, {p.failed} failed, {p.errors} errors, "
                       f"{p.skipped} skipped" + (f" ({p.message})" if p.message else ""))
            if p.status != "pass" and p.output_tail:
                click.echo(p.output_tail[-2000:], err=True)
        n_pass = sum(t.status == "pass" for t in run.targets)
        click.echo(f"numguard: {n_pass}/{len(run.targets)} targets passed; overall: {run.status.upper()}")
    bad = {"fail": ("fail", "error"), "error": ("error",), "never": ()}[fail_on]
    sys.exit(1 if run.status in bad else 0)


@main.command("summary")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="result.json from `numguard check --json-out`.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Append the card here (default: $GITHUB_STEP_SUMMARY, else stdout).")
def summary_cmd(json_path: str, out_path: str | None) -> None:
    """Render the card from result.json into the job summary (no token needed).

    The result is validated and the card re-rendered from it, as in `numguard comment`.
    """
    from .ci import write_summary  # noqa: PLC0415
    from .result import ResultValidationError  # noqa: PLC0415

    try:
        dest = write_summary(json_path, out_path)
    except (ValueError, ResultValidationError, OSError) as exc:
        raise click.ClickException(f"could not write summary: {exc}") from exc
    if dest:
        click.echo(f"wrote numguard card to {dest}", err=True)


@main.command("comment")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="result.json (from the downloaded artifact; treated as untrusted).")
@click.option("--pr", "pr_number", default=None, help="PR number (default: from result.json).")
@click.option("--repo", "repo_slug", default=None, help="owner/repo (default: $GITHUB_REPOSITORY).")
@click.option("--bot-login", default=None, help="Only update existing cards written by this login "
                                                "(default: any Bot account).")
@click.option("--expect-head-sha", default=None,
              help="Refuse unless result.json and the PR's current head both match this commit "
                   "(pass github.event.workflow_run.head_sha).")
@click.option("--check-pr-files/--no-check-pr-files", default=True, show_default=True,
              help="Warn on the card when the PR changes workflows, numguard.toml or the test harness.")
@click.option("--labels/--no-labels", default=False, show_default=True,
              help="Also set a `numguard: pass|fail|error` label (created if missing).")
@click.option("--dry-run", is_flag=True, help="Print the card instead of posting it.")
def comment_cmd(json_path: str, pr_number: str | None, repo_slug: str | None, bot_login: str | None,
                expect_head_sha: str | None, check_pr_files: bool, labels: bool, dry_run: bool) -> None:
    """Post or update the numguard card comment on a pull request.

    result.json is validated and the card re-rendered from it (card.md is
    never posted). Meant for the privileged workflow_run job, which never
    runs pull-request code.
    """
    from .card import harness_warning, render_card  # noqa: PLC0415
    from .ci import load_validated  # noqa: PLC0415
    from .github_api import GitHubError, env_client, harness_changes, validate_number  # noqa: PLC0415
    from .result import ResultValidationError  # noqa: PLC0415

    try:
        result = load_validated(json_path)
    except (ValueError, ResultValidationError, OSError) as exc:
        raise click.ClickException(f"invalid result.json: {exc}") from exc
    try:
        if pr_number is not None:
            number = validate_number(pr_number)
        elif result.get("pr_number"):
            number = validate_number(result["pr_number"])
        else:
            raise click.ClickException("PR number not given (--pr) and not in result.json")
    except GitHubError as exc:
        raise click.ClickException(str(exc)) from exc
    card = render_card(result)
    if dry_run:
        click.echo(card)
        return
    env = dict(os.environ)
    if repo_slug:
        env["GITHUB_REPOSITORY"] = repo_slug
    if not env.get("GITHUB_TOKEN"):
        raise click.ClickException("GITHUB_TOKEN is not set")
    try:
        client = env_client(env)
        if expect_head_sha:
            expected = expect_head_sha.strip().lower()
            if result.get("head") != expected:
                raise click.ClickException("result.json head commit does not match --expect-head-sha")
            pr = client.get_pull(number)
            if str((pr.get("head") or {}).get("sha", "")).lower() != expected:
                raise click.ClickException(f"PR #{number} head does not match --expect-head-sha "
                                           "(PR updated, or wrong PR number)")
        if check_pr_files:
            changed = harness_changes(client.list_pull_files(number))
            if changed:
                card = render_card(result, trusted_warnings=[harness_warning(changed)])
        action, cid = client.upsert_card_comment(number, card, bot_login=bot_login)
    except GitHubError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{action} numguard card comment {cid} on #{number}")
    if labels:
        try:
            added, removed = client.sync_verdict_label(number, result["status"])
        except GitHubError as exc:
            click.echo(f"warning: could not update labels: {exc}", err=True)
        else:
            click.echo(f"labels: added {added or 'none'}, removed {removed or 'none'}")


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
