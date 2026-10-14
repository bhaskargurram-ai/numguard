# Testing numguard

numguard is tested at five levels. The first four run offline (no network,
no GitHub, no LLM) and run in CI on every push and pull request; the fifth
needs a real repository and, for the agent, a real API key, and is started by
hand.

| Level | Where | Runs in CI | Needs |
|---|---|---|---|
| 1. Unit and integration tests | `tests/` | yes, Python 3.10–3.13 | numpy, click (scipy, mpmath, Hypothesis, PyYAML via `.[dev]`) |
| 2. Agent end to end with a scripted model | `tests/test_agent_*.py` | yes | nothing extra (no network) |
| 3. Workflow invariants | `tests/test_workflows.py`, `tests/test_init.py` | yes | PyYAML; actionlint optional |
| 4. Self-checks and packaging | `numguard eval`, `numguard check --config numguard.toml`, the `package` CI job | yes | `build`, `twine` (from PyPI) |
| 5. On GitHub, with a real model | a test repository | no (manual) | a repository, a token, an LLM key |

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m unittest discover -s tests      # the same suite without pytest
```

Every test is a `unittest.TestCase`, so both runners execute the same suite.
Tests that need an optional package skip without it: the pytest-plugin tests
need pytest (they run it in a subprocess), the workflow tests need PyYAML,
the real-Hypothesis test needs Hypothesis (without it, the strategies are
exercised with a small fake Hypothesis in `tests/fake_hypothesis.py`), and
SciPy-based references need SciPy.

With pytest installed as a standalone tool rather than in the environment
(as in some sandboxes), put both on the path:

```bash
PYTHONPATH=src:/path/to/pytest/site-packages python -m pytest -q
```

## 1. Unit and integration tests

- edge-case generation, domains, dtypes (including emulated bfloat16:
  rounding, limits, ULPs), checks, references, properties, reports
  (`test_cases.py`, `test_checks.py`, `test_reference.py`,
  `test_api_report.py`, `test_backends.py`);
- array backends through fake `torch` and `jax` modules
  (`tests/fake_backends.py`): conversion, bfloat16 dtype reporting, mutation
  detection, JAX's silent float32 downcast (`test_backends.py`);
- Hypothesis strategies (`test_strategies.py`);
- `numguard.toml` parsing and validation, spec documents, reference
  expressions (accepted and rejected forms, precisions), rendering of
  decorators / TOML tables / spec files and running them, `numguard check
  --config` with failing, passing and broken targets, `--root`, `--target`,
  `--fail-on`, the `[pytest]` run, and the repository's own `numguard.toml`
  (`test_config.py`);
- `result.json` validation, the card with hostile input (HTML, mentions,
  links, table breaks, fence breaks, bidi characters), size caps, and the data
  block's escaping, truncation and trailing-only parsing (`test_card.py`);
- `numguard comment` / `summary` against an in-memory fake GitHub
  (`tests/fake_github.py`): create then update one sticky comment, a human
  comment with the marker is not hijacked, head-SHA mismatch refusals, the
  harness warning, label sync, invalid input (`test_github_comment.py`);
- the CLI, the eval harness and the pytest plugin (`test_cli.py`,
  `test_corpus.py`, `test_pytest_plugin.py`).

## 2. Agent end to end with a scripted model

`tests/fake_llm.py` provides a transport that answers each request with the
next scripted step (tool calls, text, HTTP errors, or a function of the
request), in Anthropic or OpenAI wire format. The tests drive the real
client, session, tools, worker subprocess and verification logic:

- the client: both wire formats, retries and `retry-after`, error hints
  without the key, configuration from `NUMGUARD_LLM_*`, reasoning effort,
  usage and cost (`test_agent_llm.py`);
- the sandbox: the static denylist (each forbidden form), the scrubbed
  environment, the worker on the original and on a rewrite, **timeouts kill a
  rewrite that never returns**, the worker never sees `NUMGUARD_LLM_API_KEY`
  or `GITHUB_TOKEN`, errors by stage (`test_agent_sandbox.py`);
- `numguard fix`: a naive `logsumexp` gets a **plausible but wrong rewrite
  (rejected, with new failures) and then the max-shifted rewrite, which is
  verified**; a **denylisted rewrite is refused before execution** (its side
  effect never happens); nothing to fix means no model call; the CLI applies a
  verified rewrite and `numguard check` then passes; `apply_rewrite` keeps
  decorators and adds imports (`test_agent_fix.py`);
- the CI split: `agent propose` **never imports the repository** (the
  fixture module leaves a marker file when imported) and `agent verify`
  verifies or rejects the candidate, repeats the static check on a tampered
  candidate, rejects a manipulated manifest, and writes the card with the
  "suggested fix (verified)" diff (`test_agent_fix.py`);
- `numguard suggest`: a spec with a consistent reference reports the naive
  function's bugs; the model-written reference is **never compiled in the
  key-holding process**; an invalid spec is rejected with reasons; a reference
  that disagrees on benign inputs becomes *suspect* and is not reported as
  bugs; `--write` for `.py` and `.toml` (`test_agent_suggest.py`);
- `numguard bench`: pass, fail, refused (denylisted) and code-block-fallback
  tasks, the leaderboard row and JSON (`test_agent_bench.py`).

## 3. Workflow invariants

`tests/test_workflows.py` parses every workflow in `.github/workflows/` and
`examples/workflows/` and `action.yml`, and checks the security model of
[SECURITY_MODEL.md](SECURITY_MODEL.md) as code: no `pull_request_target`, no
`${{ }}` inside `run:` scripts, `persist-credentials: false` on every
checkout, minimal permissions (only comment jobs write to PRs), the check job
is unprivileged and installs numguard before PR code, the comment workflow
never checks out or installs the PR, the agent's key-holding job never runs
PR code and is the only job (with the gate's boolean) that references the
secret, the verify job has no secrets and no caches, callers pass only
declared inputs, and the release and pages workflows have the expected
triggers and permissions. `tests/test_init.py` checks that
`examples/workflows/` and `examples/numguard.toml` are exactly what
`numguard init --agent` writes.

[actionlint](https://github.com/rhysd/actionlint) catches expression and
schema mistakes the invariants do not look for; run it before changing a
workflow (it is not run in CI):

```bash
actionlint .github/workflows/*.yml examples/workflows/*.yml
```

## 4. Self-checks and packaging

CI also runs `numguard eval --baseline reference` (all 12 reference
implementations must pass), `numguard eval --baseline buggy` (every buggy one
must be detected) and `numguard check --config numguard.toml` (the
repository's own targets). The `package` job builds the sdist and wheel,
runs `twine check --strict`, checks that the sdist and the wheel contain the
CLI, the agent worker and the corpus, installs the wheel into a fresh
virtualenv and, from outside the checkout, runs `--help`, `--version`,
`agent --help`, both eval baselines, `numguard init --agent`, and a
config-driven check of a naive function whose `result.json` and card data
block it validates.

## 5. On GitHub, with a real model (manual)

What no offline test covers: the reusable workflows on GitHub's runners, the
`pull_request` → `workflow_run` hand-off, artifacts, permissions, comments
and labels, fork pull requests, and the agent with a real model. Run
`numguard init --agent` in a test repository with a naive function and a
`numguard.toml`, add the secret and variables, open a pull request, and check
the job summary, the comment, the label and (for the agent) the suggested-fix
section. These runs had not been performed when 1.0.0 was prepared; results
with real models are unknown.
