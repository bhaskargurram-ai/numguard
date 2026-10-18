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
@click.option("--root", "root_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="With --config: directory targets are resolved against and pytest runs in "
                   "(default: the configuration file's directory).")
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
@click.option("--elementwise", is_flag=True, default=None,
              help="Each output element depends only on the same input element: check that "
                   "results do not change with the other elements.")
@click.option("--grad", is_flag=True, default=None,
              help="Check gradients from automatic differentiation (needs --backend jax or torch).")
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
              root_dir: Path | None, reference: str | None, dtypes: tuple[str, ...],
              properties: tuple[str, ...], domain: str | None, nan: str | None,
              batched: bool | None, elementwise: bool | None, grad: bool | None,
              monotonic: str | None, fp_signals: str | None,
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
                      head_sha, base_sha, fail_on, max_rows, overrides, root_dir)
        return
    if config_path is not None or target_names or root_dir is not None:
        raise click.UsageError("give either TARGET or --config/--target/--root, not both")
    fn = resolve_callable(target)
    ref = resolve_callable(reference) if reference else None
    options: dict[str, Any] = {
        "nan": nan, "batched": batched, "elementwise": elementwise, "grad": grad,
        "monotonic": monotonic, "fp_signals": fp_signals,
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
                  base_sha: str | None, fail_on: str, max_rows: int, overrides: dict[str, Any],
                  root_dir: Path | None = None) -> None:
    from .config import ConfigError, load_config  # noqa: PLC0415
    from .runconfig import run_config  # noqa: PLC0415

    try:
        config = load_config(config_path)
        if root_dir is not None:
            config.root_override = root_dir.resolve()
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
        for n in run.notes:
            click.echo(f"note: {n}")
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


@main.command("scan")
@click.argument("module")
@click.option("--ref", "reference_module", metavar="MODULE", default=None,
              help="Module with trusted functions of the same names, used as references "
                   "(e.g. scipy.special when scanning jax.scipy.special).")
@click.option("--backend", type=click.Choice(BACKENDS), default="numpy", show_default=True,
              help="Array library the module's functions expect.")
@click.option("--dtype", "dtypes", multiple=True, type=click.Choice(ALL_FLOAT_DTYPES),
              help="Floating dtype to test (repeatable). Default: float32, float64.")
@click.option("--include", multiple=True, metavar="REGEX", help="Only functions whose name matches (repeatable).")
@click.option("--exclude", multiple=True, metavar="REGEX", help="Skip functions whose name matches (repeatable).")
@click.option("--budget", type=float, default=30.0, show_default=True,
              help="Seconds per function after which further dtypes are skipped.")
@click.option("--json", "as_json", is_flag=True, help="Print the scan result as JSON.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Also write the JSON result here.")
@click.option("--fail-on-flag", is_flag=True, help="Exit 1 if any function has a failing check.")
@click.option("--max-rows", type=int, default=50, show_default=True)
def scan_cmd(module: str, reference_module: str | None, backend: str, dtypes: tuple[str, ...],
             include: tuple[str, ...], exclude: tuple[str, ...], budget: float, as_json: bool,
             json_out: Path | None, fail_on_flag: bool, max_rows: int) -> None:
    """Check every public numerical function of MODULE.

    Each function's arity, input domain and whether it is elementwise are
    worked out by probing it. Results are candidates to triage, not bug
    reports: check each finding against the documentation and the latest
    release first.
    """
    from .scan import scan_module  # noqa: PLC0415

    def progress(name: str) -> None:
        if not as_json:
            click.echo(f"  checking {name} ...", err=True)

    try:
        res = scan_module(module, reference_module=reference_module, backend=backend,
                          dtypes=dtypes or ("float32", "float64"), include=include, exclude=exclude,
                          budget_s=budget, progress=progress)
    except ImportError as exc:
        raise click.ClickException(f"cannot import: {exc}") from exc
    doc = res.to_dict()
    if json_out is not None:
        json_out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if as_json:
        click.echo(json.dumps(doc, indent=2))
    else:
        click.echo(res.summary(max_rows))
    if fail_on_flag and res.flagged:
        sys.exit(1)


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


@main.command("init")
@click.option("--repo", default=".", show_default=True, type=click.Path(exists=True, file_okay=False),
              help="Repository to set up.")
@click.option("--agent", is_flag=True, help="Also add the (experimental, opt-in) suggested-fix workflow.")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
@click.option("--ref", default=None, help="numguard release (tag or commit SHA) the workflows use. "
                                          "Default: this version's tag.")
@click.option("--install-command", default=None,
              help="Shell command that installs your project in CI (default: pip install -e \".[dev]\" || "
                   "pip install -e .).")
@click.option("--target", "targets", multiple=True, metavar="MODULE:FUNC",
              help="Add a [[target]] stub for this function to numguard.toml (repeatable).")
def init_cmd(repo: str, agent: bool, force: bool, ref: str | None, install_command: str | None,
             targets: tuple[str, ...]) -> None:
    """Write the GitHub workflows and a commented numguard.toml into a repository."""
    from .init_cmd import InitError, init_repo  # noqa: PLC0415

    try:
        res = init_repo(repo, agent=agent, force=force, ref=ref, install_command=install_command, targets=targets)
    except InitError as exc:
        raise click.ClickException(str(exc)) from exc
    for f in res.written:
        click.echo(f"wrote {f}")
    click.echo("\nNext steps:")
    for i, step in enumerate(res.next_steps, 1):
        click.echo(f"  {i}. {step}")


# --------------------------------------------------------------------------
# Agent commands (opt-in; bring your own LLM key)
# --------------------------------------------------------------------------


def _llm_options(f: Callable[..., Any]) -> Callable[..., Any]:
    f = click.option("--reasoning-effort", default=None,
                     help="OpenAI-compatible only: reasoning_effort (default: $NUMGUARD_LLM_REASONING_EFFORT).")(f)
    f = click.option("--base-url", default=None, help="API base URL (default: $NUMGUARD_LLM_BASE_URL).")(f)
    f = click.option("--model", default=None, help="Model id (default: $NUMGUARD_LLM_MODEL; there is no "
                                                   "built-in default).")(f)
    f = click.option("--provider", type=click.Choice(["anthropic", "openai"]), default=None,
                     help="LLM API (default: $NUMGUARD_LLM_PROVIDER).")(f)
    return f


def _make_llm(provider: str | None, model: str | None, base_url: str | None, effort: str | None,
              max_output_tokens: int = 8192) -> Any:
    from .agent.llm import LLMConfigError, client_from_env  # noqa: PLC0415

    env = dict(os.environ)
    for key, val in (("PROVIDER", provider), ("MODEL", model), ("BASE_URL", base_url),
                     ("REASONING_EFFORT", effort)):
        if val:
            env[f"NUMGUARD_LLM_{key}"] = val
    try:
        return client_from_env(env, max_output_tokens=max_output_tokens)
    except LLMConfigError as exc:
        raise click.UsageError(str(exc)) from exc


def _echo_usage(usage: Any) -> None:
    from .agent.common import usage_line  # noqa: PLC0415

    try:
        click.echo(f"LLM usage: {usage_line(usage)}", err=True)
    except Exception as exc:  # noqa: BLE001 - a bad price variable must not hide the result
        click.echo(f"LLM usage: {usage.to_dict()} ({exc})", err=True)


@main.command("suggest")
@click.argument("target")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default=Path("."), show_default=True,
              help="Repository root the model may read (read-only).")
@click.option("--write", "write_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write the spec: *.py -> a spec/test module; *.toml -> append a [[target]] table.")
@click.option("--force", is_flag=True, help="With --write: overwrite an existing .py file.")
@click.option("--rounds", type=click.IntRange(1, 5), default=2, show_default=True,
              help="Rounds (a suspect reference or a spec that cannot run is sent back to the model).")
@click.option("--max-tool-calls", type=click.IntRange(1, 200), default=25, show_default=True)
@click.option("--timeout", type=float, default=120, show_default=True,
              help="Seconds for each subprocess run of the spec.")
@click.option("--large-n", type=int, default=None, help="Size of the large_n case (default: 100000).")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Also write the outcome (spec, reference status, report, usage) as JSON.")
@_llm_options
def suggest_cmd(target: str, root: Path, write_path: Path | None, force: bool, rounds: int, max_tool_calls: int,
                timeout: float, large_n: int | None, json_out: Path | None, provider: str | None, model: str | None,
                base_url: str | None, reasoning_effort: str | None) -> None:
    """(agent) Let an LLM write a numguard spec for TARGET; numguard validates and runs it.

    The model only gets read-only tools; its spec is validated strictly and
    run in a subprocess. A reference that disagrees with the function on
    ordinary inputs is reported as SUSPECT rather than as a bug. Exit status:
    0 spec found and checks pass, 1 checks fail, 3 no usable spec.
    """
    from .agent.suggest import run_suggest  # noqa: PLC0415

    llm = _make_llm(provider, model, base_url, reasoning_effort)
    try:
        out = run_suggest(target, llm=llm, root=root, rounds=rounds, max_tool_calls=max_tool_calls,
                          timeout=timeout, overrides={"large_n": large_n} if large_n else None,
                          log=lambda m: click.echo(m, err=True))
    except TargetError as exc:
        raise click.BadParameter(str(exc)) from exc
    if json_out is not None:
        json_out.write_text(json.dumps(out.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    _echo_usage(out.usage)
    if out.status != "ok":
        click.echo(f"numguard suggest: no usable spec ({out.status}): {out.error}")
        sys.exit(3)
    click.echo(f"numguard suggest: {target} -- reference: {out.reference_status}"
               + (f" (disagrees on ordinary inputs: {', '.join(out.suspect_cases[:6])})" if out.suspect_cases else ""))
    for n in out.notes:
        click.echo(f"note: {n}")
    click.echo("\n# --- decorator (paste above the function) ---")
    click.echo(out.decorator)
    click.echo("# --- numguard.toml ---")
    click.echo(out.toml)
    click.echo("# --- check report ---")
    click.echo(out.summary)
    if write_path is not None:
        _write_suggestion(write_path, out, force)
    sys.exit(0 if out.check_ok else 1)


def _write_suggestion(path: Path, out: Any, force: bool) -> None:
    if path.suffix == ".toml":
        from .config import ConfigError, load_config  # noqa: PLC0415

        name = out.toml.split("name = ", 1)[1].split("\n", 1)[0].strip('"') if "name = " in out.toml else ""
        if path.exists():
            try:
                cfg = load_config(path)
            except ConfigError as exc:
                raise click.ClickException(f"{path}: {exc}") from exc
            if any(t["name"] == name for t in cfg.targets):
                raise click.ClickException(f"{path} already has a target named {name!r}; not appending")
            text = path.read_text(encoding="utf-8")
            path.write_text(text.rstrip("\n") + "\n\n" + out.toml, encoding="utf-8")
        else:
            path.write_text(out.toml, encoding="utf-8")
        click.echo(f"appended [[target]] {name!r} to {path}", err=True)
        return
    if path.exists() and not force:
        raise click.ClickException(f"{path} exists (use --force to overwrite)")
    path.write_text(out.python_spec, encoding="utf-8")
    click.echo(f"wrote {path}", err=True)


def _fix_spec(target: str, spec: Path | None, config_path: Path | None) -> tuple[str, dict[str, Any], Any]:
    """(function, worker spec request, [agent] settings or None)."""
    from .config import ConfigError, find_config, load_config  # noqa: PLC0415

    if spec is not None and spec.suffix == ".py":
        return target, {"source": "pyfile", "pyfile": str(spec.resolve())}, None
    cfg_path = spec if spec is not None else (config_path or find_config())
    if cfg_path is not None:
        try:
            cfg = load_config(cfg_path)
        except ConfigError as exc:
            raise click.UsageError(f"{cfg_path}: {exc}") from exc
        for t in cfg.targets:
            if target in (t["name"], t["function"]):
                doc = {k: v for k, v in t.items() if k not in ("name", "function")}
                return t["function"], {"source": "doc", "doc": doc}, cfg.agent
        if spec is not None or config_path is not None:
            raise click.UsageError(f"{cfg_path} has no target named {target!r}")
    return target, {"source": "guard"}, None


@main.command("fix")
@click.argument("target")
@click.option("--spec", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Spec: a Python spec file (SPEC dict, as written by `suggest --write x.py`) or a numguard.toml.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="numguard.toml holding TARGET (by name or function). Default: ./numguard.toml if it has it, "
                   "else the function's @numguard.guard spec.")
@click.option("--rounds", type=click.IntRange(1, 10), default=None,
              help="Refinement rounds: a rejected rewrite's failure table goes back to the model "
                   "(default: [agent] rounds of the config, else 3).")
@click.option("--apply", "apply_", is_flag=True, help="Write a VERIFIED rewrite into the source file.")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default=Path("."), show_default=True)
@click.option("--timeout", type=float, default=120, show_default=True,
              help="Seconds for each subprocess check (original and every rewrite).")
@click.option("--max-tool-calls", type=click.IntRange(1, 200), default=None,
              help="Read-only tool calls per conversation (default: [agent] max_tool_calls, else 25).")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@_llm_options
def fix_cmd(target: str, spec: Path | None, config_path: Path | None, rounds: int | None, apply_: bool,
            root: Path, timeout: float, max_tool_calls: int | None, json_out: Path | None, provider: str | None, model: str | None,
            base_url: str | None, reasoning_effort: str | None) -> None:
    """(agent) Let an LLM rewrite a failing function; numguard verifies the rewrite by executing it.

    A rewrite is verified only if, run in a subprocess, it passes every check
    the original failed, adds no new failure, keeps the signature, and agrees
    with the original on ordinary inputs where the original passed. Prints a
    unified diff and a before/after table; files change only with --apply.
    Exit status: 0 verified (or nothing to fix), 1 otherwise.
    """
    from .agent.fix import before_after_rows, run_fix  # noqa: PLC0415
    from .report import format_table  # noqa: PLC0415

    function, spec_req, settings = _fix_spec(target, spec, config_path)
    rounds = rounds or (settings.rounds if settings else 3)
    max_tool_calls = max_tool_calls or (settings.max_tool_calls if settings else 25)
    llm = _make_llm(provider, model, base_url, reasoning_effort)
    try:
        out = run_fix(function, llm=llm, root=root, spec_req=spec_req, rounds=rounds, timeout=timeout,
                      max_tool_calls=max_tool_calls, log=lambda m: click.echo(m, err=True))
    except TargetError as exc:
        raise click.BadParameter(str(exc)) from exc
    if json_out is not None:
        json_out.write_text(json.dumps(out.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    if out.status == "nothing_to_fix":
        click.echo(f"numguard fix: {function} passes every check; nothing to fix.")
        sys.exit(0)
    _echo_usage(out.usage)
    if out.status == "error" and not out.attempts:
        click.echo(f"numguard fix: error: {out.error}")
        sys.exit(1)
    best = out.verified or out.final
    rows = before_after_rows(out.before, best.data if best and best.data else {})
    click.echo(f"numguard fix: {function}: {out.status.upper()} after {len(out.attempts)} round(s)")
    if best is not None:
        for r in best.reasons:
            click.echo(f"  - {r}")
    click.echo(format_table(["check", "failing before", "failing after"],
                            [[r["check"], str(r["before"]), "-" if r["after"] is None else str(r["after"])]
                             for r in rows]))
    if out.diff:
        click.echo(out.diff)
    if apply_:
        if out.status != "verified":
            click.echo("not applied: the rewrite is not verified", err=True)
        elif not out.new_text:
            click.echo("not applied: the function's source file could not be located for editing", err=True)
        else:
            (root / out.rel_path).write_text(out.new_text, encoding="utf-8")
            click.echo(f"applied the verified rewrite to {out.rel_path}", err=True)
    sys.exit(0 if out.status == "verified" else 1)


@main.command("bench")
@click.option("--task", "tasks", multiple=True, help="Only these corpus tasks (repeatable; default: all).")
@click.option("--tasks", "tasks_csv", default=None, help="Comma-separated task names (alternative to --task).")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Where <task>/solution.py files are written (default: numguard-bench/<model>).")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--corpus-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--timeout", type=float, default=300, show_default=True, help="Seconds to grade each task.")
@_llm_options
def bench_cmd(tasks: tuple[str, ...], tasks_csv: str | None, out_dir: Path | None, json_out: Path | None,
              corpus_dir: Path | None, timeout: float, provider: str | None, model: str | None,
              base_url: str | None, reasoning_effort: str | None) -> None:
    """(agent) Benchmark an LLM on the bug corpus: one solution per task, graded by numguard.

    Prints a leaderboard row (model, pass@1, happy-path pass, per-task
    reasons) and the per-task table. Solutions are statically checked and
    graded in a subprocess.
    """
    import re  # noqa: PLC0415

    from .agent.bench import run_bench  # noqa: PLC0415

    names = list(tasks) + [t.strip() for t in (tasks_csv or "").split(",") if t.strip()]
    known = list_tasks(corpus_dir)
    for t in names:
        if t not in known:
            raise click.UsageError(f"unknown task {t!r}; known tasks: {', '.join(known)}")
    llm = _make_llm(provider, model, base_url, reasoning_effort)
    out_dir = out_dir or Path("numguard-bench") / re.sub(r"[^A-Za-z0-9_.-]+", "-", llm.model)
    res = run_bench(llm, tasks=names or None, solutions_dir=out_dir,
                    corpus=str(corpus_dir) if corpus_dir else None, timeout=timeout,
                    log=lambda m: click.echo(m, err=True))
    _echo_usage(res.usage)
    click.echo(res.table())
    click.echo("")
    click.echo("| model | pass@1 | happy-path | failures |")
    click.echo("|---|---|---|---|")
    click.echo(res.leaderboard_row())
    data = res.to_dict()
    if json_out is not None:
        json_out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps({k: data[k] for k in ("model", "n_tasks", "n_pass", "pass_at_1", "n_happy_path_pass")}))


@main.group("agent")
def agent_group() -> None:
    """(agent) The CI split: `propose` (holds the key, runs no repository code) and `verify` (no key)."""


@agent_group.command("propose")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--result", "result_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="result.json of the check job (validated; treated as untrusted data).")
@click.option("--repo", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."),
              show_default=True, help="Checkout of the pull request, read as DATA (never imported or run).")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Candidates directory (manifest.json + cand_<n>.py).")
@click.option("--max-targets", type=click.IntRange(1, 50), default=None,
              help="Failing targets to try (default: [agent] max_targets, 3).")
@click.option("--variants", type=click.IntRange(1, 10), default=None,
              help="Independent rewrites per target; verify keeps the best (default: [agent] variants, 3).")
@_llm_options
def agent_propose_cmd(config_path: Path, result_path: Path, repo: Path, out_dir: Path, max_targets: int | None,
                      variants: int | None, provider: str | None, model: str | None, base_url: str | None,
                      reasoning_effort: str | None) -> None:
    """Ask the model for rewrites of failing targets. Executes nothing from the repository."""
    from .agent.ci import propose  # noqa: PLC0415
    from .ci import load_validated  # noqa: PLC0415
    from .config import ConfigError, load_config  # noqa: PLC0415
    from .result import ResultValidationError  # noqa: PLC0415

    try:
        config = load_config(config_path)
        result = load_validated(result_path)
    except (ConfigError, ResultValidationError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not config.agent.enabled:
        click.echo("numguard agent: [agent] enabled = true is not set in the configuration; nothing to do")
        return
    llm = _make_llm(provider, model, base_url, reasoning_effort)
    manifest = propose(config, result, repo, out_dir, llm, max_targets=max_targets, variants=variants,
                       log=lambda m: click.echo(m, err=True))
    n = sum(c["status"] == "submitted" for c in manifest["candidates"])
    click.echo(f"numguard agent propose: {n} candidate rewrite(s) in {out_dir}")


@agent_group.command("verify")
@click.option("--candidates", "cand_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--result", "result_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--repo", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."),
              show_default=True, help="Checkout of the pull request (its code runs here: use a job without secrets).")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Updated result.json (with `fixes`).")
@click.option("--card-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--timeout", type=float, default=None, help="Seconds per subprocess check (default: [agent] timeout).")
def agent_verify_cmd(cand_dir: Path, config_path: Path, result_path: Path, repo: Path, json_out: Path,
                     card_out: Path | None, timeout: float | None) -> None:
    """Run every candidate rewrite against its target's checks; add `fixes` to result.json. Uses no LLM."""
    from .agent.ci import ManifestError, verify  # noqa: PLC0415
    from .card import render_card  # noqa: PLC0415
    from .ci import load_validated  # noqa: PLC0415
    from .config import ConfigError, load_config  # noqa: PLC0415
    from .result import ResultValidationError  # noqa: PLC0415

    try:
        config = load_config(config_path)
        result = load_validated(result_path)
        doc = verify(cand_dir, config, result, repo, timeout=timeout, log=lambda m: click.echo(m, err=True))
    except (ConfigError, ResultValidationError, ManifestError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    json_out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if card_out is not None:
        card_out.write_text(render_card(doc), encoding="utf-8")
    for f in doc["fixes"] or []:
        click.echo(f"numguard agent verify: {f['target']}: {f['status']} "
                   f"({f['before_failures']} -> {f['after_failures'] if f['after_failures'] is not None else '-'})")


if __name__ == "__main__":  # pragma: no cover
    main()
