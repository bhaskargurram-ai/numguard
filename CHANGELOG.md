# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `numguard scan --timeout SECONDS` (default 300): a hard per-function limit, so one slow function (JAX recompiles for every new input shape) cannot stall a scan. A function over the limit is recorded as an error and the scan moves on.

## [1.1.0] - 2026-10-18

### Added

- **Gradient checks.** `grad=True` checks automatic differentiation in JAX and PyTorch; `grad=fn` checks a gradient function you wrote. The gradient of `sum(f)` is compared, at every finite edge case, with float64 central differences (Richardson-extrapolated, with an error bound per point) of `reference` or of the function in float64, or with `grad_reference`. Two more checks run with `grad=True`: `grad:primal` (the primal value of a forward-mode `jvp` must equal `f(x)`, the bug fixed in jax-ml/jax#41017) and, for elementwise functions, `grad:forward` (forward and reverse mode must agree). Points where finite differences can't be trusted are skipped, not reported.
- **`isolation` check** (`elementwise=True`): each element's result must be the same when computed alone, in the full array, and next to every other case's inputs, NaN and infinities included. It catches work shared across a batch, such as a single convergence loop for every element (the bug fixed in jax-ml/jax#41440).
- `--elementwise` and `--grad` CLI flags and `elementwise`, `grad`, `grad_rtol` keys in `numguard.toml`.
- CI runs the backend and gradient tests against real CPU builds of JAX and PyTorch.
- **`numguard scan MODULE`**: checks every public numerical function of a module. It probes each function for its arity, a domain with finite output and whether it is elementwise, and compares with same-named functions of `--ref MODULE` (for example `jax.scipy.special` against `scipy.special`). Results are candidates to triage, with JSON output for CI.
- `pytest-numguard` on PyPI: a metapackage that installs numguard-ci, so the
  pytest plugin appears in pytest's plugin list (which indexes `pytest-*`
  projects only). Published with the `publish pytest-numguard` workflow.

### Fixed

- Windows: target functions given as a path with a drive letter (`C:\work\ops.py:f`) are accepted, and output of child processes (agent sandbox, `bench`, the `[pytest]` run) is decoded as UTF-8 instead of the system code page.

### Changed

- CI also runs the unit tests on macOS and Windows.
- Status is now labelled beta in the README and in the PyPI classifiers (`Development Status :: 4 - Beta`), until there are confirmed outside users. Version numbering stays 1.x.
- Added `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, `CITATION.cff`, issue and pull request templates, and README sections linking the other two tools of the verification toolkit and explaining how to cite.

## [1.0.1] - 2026-10-14

### Changed

- The GitHub Action is listed on the Marketplace as "numguard numerical edge cases" (`numguard` alone is reserved by an existing GitHub account). Workflows that use `bhaskargurram-ai/numguard@v1.0.1` are unaffected; only the display name changed.

## [1.0.0] - 2026-10-14

Published on PyPI as **`numguard-ci`** (`pip install numguard-ci`), because the name
`numguard` is taken there. The command, the import (`import numguard`) and the
GitHub repository keep the name `numguard`.

### Added

- **`numguard.toml` and `numguard check --config`.** `[[target]]` tables
  describe how to check a function without decorators (`function`, `inputs`,
  `dtypes`, `properties`, `reference` as a restricted Python expression with
  `reference_precision` `float64`/`longdouble`/`mpmath`, or
  `reference_function`, tolerances, `nan`, `bounds`, `monotonic`, `batched`,
  `fp_signals`, `out_dtype`, `kwargs`, `backend`, ...), with `[defaults]`, an
  optional `[pytest]` run and `[agent]` settings. Unknown keys are errors and
  API keys are refused. `check` without a target uses `./numguard.toml`;
  `--target`, `--root`, `--json-out`, `--card-out`, `--fail-on`,
  `--no-pytest` and `--pr-number/--head-sha/--base-sha` support CI.
- **result.json, the card and its data block.** `check --json-out` writes a
  versioned result document; `numguard summary` writes the Markdown card to
  the job summary and `numguard comment` upserts one sticky PR comment (bot
  author check, `--expect-head-sha`, a warning when the PR changes workflows,
  `numguard.toml` or the test harness, optional `numguard: pass|fail|error`
  labels). The card is re-rendered from a strictly validated result, escapes
  everything that can come from the PR, and ends with a
  `<!-- numguard:data {...} -->` block (`numguard.carddata.parse_card_data`).
- **GitHub integration.** A composite `action.yml` (`check`, `summary`,
  `comment`, `agent-propose`, `agent-verify`) and reusable workflows:
  `check.yml` (unprivileged `pull_request` job; numguard installed from its
  own repository before PR code; config read from the base commit),
  `comment.yml` (privileged `workflow_run` job that never runs PR code) and
  `agent.yml` (gate / propose / verify). `numguard init [--agent]` writes the
  caller workflows and a commented `numguard.toml`; `examples/` is exactly
  what `init` writes. Workflow security invariants are tested.
- **Agent (experimental, bring your own key).** `numguard suggest` lets an
  LLM write a spec (read-only tools: the function, repository files, grep,
  numguard's catalogue); the spec is validated strictly and run in a
  subprocess, and a reference that disagrees with the function on benign
  inputs is marked *suspect* instead of reported as bugs. Output: a
  `@numguard.guard(...)` decorator, a `[[target]]` table and a runnable spec
  file (`--write spec.py|numguard.toml`). `numguard fix` lets an LLM rewrite a
  failing function and verifies the rewrite by executing it: same signature,
  every originally failing check passes, no new failures, agreement with the
  original on benign inputs, unchanged shapes and dtypes there; up to
  `--rounds` refinement rounds; a unified diff and a before/after table;
  `--apply` writes only verified rewrites. `numguard bench` runs a model on
  the 12 corpus tasks and prints a leaderboard row (pass@1, happy-path pass,
  per-task reasons) and JSON. `numguard agent propose` / `agent verify` split
  the work for CI; since the propose job cannot execute code, it asks for
  `[agent] variants` (default 3) independent rewrites per failing target and
  verify keeps the best. Model-written code passes a static denylist and runs only
  in a subprocess worker with a timeout and a scrubbed environment. The
  tool-use client (Anthropic Messages and OpenAI-compatible chat completions,
  retries, token and cost accounting) is configured with `NUMGUARD_LLM_*`
  environment variables; there is no default model.
- **Backends.** `backend="torch"|"jax"|"cupy"` (`--backend`,
  `--numguard-backend`, `backend = ...` in the config): cases are generated
  and compared in NumPy, inputs converted to the backend's arrays and
  outputs back, with backend dtypes reported to the dtype check.
- **bfloat16** as a dtype under test (emulated: bfloat16 limits, values
  stored as representable float32, bfloat16 tolerances and ULPs; the NumPy
  backend needs `ml_dtypes`).
- **`numguard.strategies`**: Hypothesis strategies (`cases`, `edge_cases`,
  `random_inputs`, `inputs_for`) that draw from the edge-case catalogue and
  from random in-domain arrays. Hypothesis is optional.
- `numguard.resolve` (import a target, or locate its source without
  importing), `Report.grouped_failures()`, `Report.backend`.
- Release engineering: CI packaging job (build, `twine check`, install the
  wheel into a fresh virtualenv, smoke-test outside the checkout),
  `release.yml` (one-click release that creates the tag; PyPI publishing with a
  `PYPI_API_TOKEN` secret or trusted publishing; GitHub release from this file),
  `pages.yml`; docs/AGENT.md, docs/SECURITY_MODEL.md, docs/TESTING.md,
  docs/CONFIG.md, docs/CLI.md and docs/GITHUB.md.

### Changed

- `numguard check TARGET` is unchanged; `TARGET` is now optional (config mode).
- `--dtype` accepts `bfloat16`.
- New dependency on Python 3.10 only: `tomli` (the standard `tomllib`
  backport) for reading `numguard.toml`.
- The `dev` extra also installs Hypothesis and PyYAML; new extras
  `hypothesis` and `bfloat16`.

### Fixed

- `module:function` targets with a one-letter module name (for example
  `m:f`) were mistaken for Windows drive letters.

## [0.1.0] - 2026-10-13

First release.

### Added

- `numguard.cases`: a deterministic, domain-aware edge-case generator for float16,
  float32, float64 and integer inputs. It covers zeros, signed zero, constants,
  empty/single/pair/large-n sizes, NaN and ±inf, huge/large/small/tiny/subnormal
  magnitudes, wide-range sweeps, cancellation-prone large-offset data,
  alternating signs, duplicates, sorted and reverse-sorted orderings, and
  interval boundaries. Multi-argument functions also get per-argument variants.
- `numguard.checks`: the error, mutation, dtype, finite, nan, reference
  (rtol/atol with max ULP error), fp_signals, bounds, postcondition, batch and
  monotonic checks, plus metamorphic properties (permutation, translation and
  scale invariance/equivariance, symmetry, odd/even). Floating-point exceptions
  are recorded as evidence.
- `numguard.reference`: `ulp_distance`, `longdouble()`, `mp_scalar()` (mpmath)
  and `highprec()`, which picks the most precise reference available.
- `numguard.check()`, the `@numguard.guard` decorator and `Spec`.
- `Report` with `.ok`, `.failures`, `.summary()`, `.to_json()` and
  `.raise_if_failed()` (raises `NumGuardError`).
- A pytest plugin (`pytest11` entry point) with the `@pytest.mark.numguard`
  marker, `--numguard-modules` auto-collection, and the `--numguard-dtypes` and
  `--numguard-large-n` options.
- A CLI with `numguard check`, `numguard eval`, `numguard tasks`,
  `numguard cases` and `python -m numguard`.
- A bug corpus of 12 tasks (task prompt, buggy and reference implementations,
  numguard spec), and an agent-agnostic eval harness
  (`numguard eval --candidate-dir DIR`, `--baseline buggy|reference`).
