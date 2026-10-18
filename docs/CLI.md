# Command-line reference

`numguard` (also `python -m numguard`). Every command has `--help`.
Exit status: `0` success, `1` checks failed (or the command's verdict is
negative), `2` usage error; `suggest` uses `3` for "no usable spec".

## Deterministic commands

### `numguard check [TARGET]`

Check one function, or every target of a `numguard.toml`.

```bash
numguard check mypkg.ops:logsumexp --ref scipy.special:logsumexp --prop translation_equivariant
numguard check path/to/ops.py:softmax --dtype float16 --batched
numguard check                                   # every [[target]] in ./numguard.toml (+ [pytest])
numguard check --config numguard.toml --target logsumexp --json-out result.json --card-out card.md
```

Single-function options (override an attached `@numguard.guard` spec):
`--ref MODULE:FUNC`, `--dtype float16|float32|float64|bfloat16` (repeatable),
`--prop NAME[:VALUE]` (repeatable), `--domain`, `--nan propagate|raise|ignore`,
`--batched`, `--elementwise`, `--grad` (with `--backend jax|torch`),
`--monotonic increasing|decreasing`, `--fp-signals ignore|fail`,
`--rtol`, `--atol`, `--large-n`, `--seed`, `--exclude` (repeatable),
`--allow-nonfinite` (repeatable), `--backend numpy|torch|jax|cupy`,
`--json` (print the report as JSON), `--max-rows`.

Config mode (no `TARGET`): `--config FILE` (default `./numguard.toml`),
`--target NAME` (repeatable; skips pytest), `--root DIR` (resolve targets
against `DIR`), `--no-pytest`, `--dtype`/`--large-n`/`--seed`/`--backend`
(override every target), `--json` (print the result document).

Both modes: `--json-out FILE` (write `result.json`), `--card-out FILE`
(write the Markdown card), `--pr-number`, `--head-sha`, `--base-sha`
(metadata recorded in `result.json`), `--fail-on fail|error|never`
(default `fail`: exit 1 when a target fails or errors).

### `numguard eval`

Grade implementations of the 12 bug-corpus tasks:
`--candidate-dir DIR` (`DIR/<task>/solution.py`) or
`--baseline buggy|reference`; `--task` (repeatable), `--corpus-dir`,
`--solution-name`, `--json`, `--json-out FILE`, `--expect pass|fail`.

### `numguard tasks`, `numguard cases`

List the corpus tasks (`--show TASK` prints a task prompt) and the edge-case
kinds with their tags.

### `numguard summary --json result.json [--out FILE]`

Validate `result.json`, render the card and append it to `--out`, to
`$GITHUB_STEP_SUMMARY`, or print it. Needs no token.

### `numguard comment --json result.json`

Post or update the sticky card comment on a pull request. Options:
`--pr N` (default: from `result.json`), `--repo owner/name` (default
`$GITHUB_REPOSITORY`), `--expect-head-sha SHA` (refuse unless `result.json`
and the PR's current head match), `--check-pr-files/--no-check-pr-files`
(default on: warn when the PR changes workflows, `numguard.toml`,
`pyproject.toml`, `conftest.py`, ...), `--labels` (set
`numguard: pass|fail|error`), `--bot-login`, `--dry-run`. Needs
`GITHUB_TOKEN` with `pull-requests: write`.

### `numguard init`

Write `.github/workflows/numguard.yml`, `numguard-comment.yml`, (with
`--agent`) `numguard-agent.yml`, and a commented `numguard.toml`. Options:
`--repo DIR`, `--agent`, `--force`, `--ref TAG_OR_SHA` (default: this
release's tag), `--install-command CMD`, `--target MODULE:FUNC` (repeatable;
adds `[[target]]` stubs). Refuses to overwrite anything without `--force`.

## Agent commands (experimental; bring your own key)

All agent commands read the model from the environment, overridable with
`--provider anthropic|openai`, `--model`, `--base-url`,
`--reasoning-effort`:

| variable | |
|---|---|
| `NUMGUARD_LLM_PROVIDER` | `anthropic` or `openai` (any OpenAI-compatible chat-completions server) |
| `NUMGUARD_LLM_MODEL` | model id; **required**, there is no default model |
| `NUMGUARD_LLM_API_KEY` | API key (not needed for a `http://localhost` base URL) |
| `NUMGUARD_LLM_BASE_URL` | optional API base URL |
| `NUMGUARD_LLM_REASONING_EFFORT` | optional, OpenAI-compatible only: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. OpenAI's GPT-6 Luna needs `none` to call tools through Chat Completions; GPT-6 models reject `temperature` unless reasoning is off (numguard sends no temperature) |
| `NUMGUARD_LLM_PRICE_IN`, `NUMGUARD_LLM_PRICE_OUT` | optional USD per million input/output tokens, for the cost estimate |

See [AGENT.md](AGENT.md) for what each command does and does not guarantee.

### `numguard suggest TARGET`

Draft a spec for a function. Options: `--root DIR` (what the model may
read; default `.`), `--write FILE` (`*.py`: a runnable spec/test module;
`*.toml`: append a `[[target]]`, refusing a duplicate name), `--force`,
`--rounds N` (default 2), `--max-tool-calls N` (default 25), `--timeout S`
(default 120, per subprocess run), `--large-n`, `--json-out FILE`.
Exit 0: spec found, checks pass; 1: spec found, checks fail; 3: no usable spec.

### `numguard fix TARGET`

Propose and verify a rewrite of a failing function. The spec comes from
`--spec FILE` (a Python spec file with a `SPEC` dict, as written by
`suggest --write x.py`, or a `numguard.toml`), `--config FILE`, the target
of `./numguard.toml` whose `name` or `function` is `TARGET`, or the function's
`@numguard.guard` spec. Options: `--rounds N` (default: `[agent] rounds`, else
3), `--apply` (write a *verified* rewrite into the source file), `--root DIR`,
`--timeout S` (default 120), `--max-tool-calls N`, `--json-out FILE`.
Prints the outcome, the reasons, a before/after table of failing results per
check, and a unified diff. Exit 0: verified (or nothing to fix); 1 otherwise.

### `numguard bench`

Run a model on the corpus: `--task NAME` (repeatable) / `--tasks a,b`,
`--out-dir DIR` (default `numguard-bench/<model>`), `--json-out FILE`,
`--corpus-dir`, `--timeout S` (grading time per task, default 300). Prints a
per-task table, a Markdown leaderboard row
(`| model | pass@1 | happy-path | failures |`) and a JSON summary.

### `numguard agent propose` / `numguard agent verify`

The CI split (see [GITHUB.md](GITHUB.md)).

```bash
numguard agent propose --config numguard.toml --result result.json --repo pr/ --out-dir candidates/ [--max-targets N]
numguard agent verify --candidates candidates/ --config numguard.toml --result result.json --repo . \
    --json-out result.json [--card-out card.md] [--timeout S]
```

`propose` holds the key and executes nothing from `--repo`; it does nothing
unless `[agent] enabled = true`. `verify` uses no LLM and runs the code.

## pytest plugin options

`--numguard-modules mod1,mod2` (collect `@numguard.guard` functions; also the
`numguard_modules` ini option), `--numguard-dtypes float16,float32`,
`--numguard-large-n N`, `--numguard-backend numpy|torch|jax|cupy`, and the
`@pytest.mark.numguard(fn, **check_options)` marker.

## Environment variables (deterministic side)

`NUMGUARD_CORPUS` (corpus directory override), `NUMGUARD_TORCH_DEVICE`
(device for the torch backend), `NUMGUARD_DEBUG` (print tracebacks of target
errors in config mode).
