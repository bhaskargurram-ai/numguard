# GitHub integration

numguard runs on pull requests with the same security model as its sister
project proofcheck: the pull request's code only runs in an **unprivileged**
job, and the job that comments on the pull request **never runs PR code**.
See [SECURITY_MODEL.md](SECURITY_MODEL.md) for the reasoning.

## Set up

```bash
pip install "numguard-ci @ git+https://github.com/bhaskargurram-ai/numguard@v1.0.0"
numguard init            # or: numguard init --agent
```

`init` writes three small caller workflows that delegate to reusable
workflows in this repository, plus a commented `numguard.toml` (it refuses to
overwrite existing files without `--force`). The files in
[`examples/workflows/`](../examples/workflows) and
[`examples/numguard.toml`](../examples/numguard.toml) are exactly what
`numguard init --agent` writes.

| caller | trigger | calls | does |
|---|---|---|---|
| `.github/workflows/numguard.yml` (`name: numguard`) | `pull_request` | `check.yml` | installs numguard and your project, runs `numguard check --config`, writes the card to the job summary, uploads the `numguard-result` artifact |
| `.github/workflows/numguard-comment.yml` | `workflow_run` of `numguard` | `comment.yml` | validates the artifact, re-renders the card, upserts one comment, sets a `numguard: pass\|fail\|error` label |
| `.github/workflows/numguard-agent.yml` (`--agent`) | `workflow_run` of `numguard` | `agent.yml`, then `comment.yml` | suggested fixes for failing targets (experimental) |

`workflow_run` workflows only run from the default branch: merge the files
before expecting comments.

## Reusable workflows

### `check.yml` (unprivileged)

| input | default | |
|---|---|---|
| `python-version` | `"3.12"` | |
| `install-command` | `pip install -e ".[dev]" \|\| pip install -e .` | installs the project under test (runs PR code) |
| `config` | `numguard.toml` | path of the config; targets resolve relative to its directory |
| `config-from` | `base` | read the config from the PR's base commit when it exists there; `head` to use the PR's |
| `fail-on` | `fail` | `fail`, `error` or `never` |
| `numguard-extras` | `scipy` | extras installed with numguard (e.g. `scipy,mpmath`) |
| `numguard-ref` | `v1.0.0` | numguard version to install; keep it equal to the `@ref` of the workflow |
| `numguard-repository` | `bhaskargurram-ai/numguard` | |
| `timeout-minutes` | 30 | |

Output: `status` (`pass` / `fail` / `error`). Artifact `numguard-result`:
`result.json`, `card.md`, `pr_number`, `head_sha`.

Job: `pull_request` only, `permissions: contents: read`, no secrets. Order:
check out the PR head (`persist-credentials: false`) → check out and install
numguard from its own repository → select the config (base commit first) →
install the project (PR code) → `python -m numguard check --config ... --root
... --json-out ... --card-out ... --pr-number ... --head-sha ... --base-sha ...
--fail-on ...` → job summary → upload.

### `comment.yml` (privileged)

Inputs: `numguard-ref`, `numguard-repository`, `python-version`, `labels`
(default `true`), `pr-number` (a trusted number from a job that ran no PR
code; default: the artifact's, validated), `artifact-run-id` (default: the
triggering run). Permissions: `contents: read`, `actions: read`,
`pull-requests: write`. It checks out only numguard itself, installs it into
an isolated virtualenv, downloads the artifact, validates the PR number and
the triggering run's head SHA, and runs `numguard comment --json ... --pr ...
--expect-head-sha ... [--labels]`.

### `agent.yml` (experimental)

| input | default | |
|---|---|---|
| `llm-provider`, `llm-model`, `llm-base-url`, `llm-reasoning-effort` | `""` | passed as `NUMGUARD_LLM_*`; a model is required |
| `run-for` | `trusted` | `trusted`: only PRs by owners/members/collaborators, or labelled `label`; `everyone`: any PR (spends credits on fork PRs) |
| `label` | `numguard:agent` | opt-in label for other authors |
| `config` | `numguard.toml` | read from the base commit when it exists there |
| `install-command`, `python-version`, `numguard-extras`, `numguard-ref`, `numguard-repository` | | as in `check.yml` (install runs only in `verify`) |
| `timeout-minutes` | 30 | timeout of the verify job |

Secret: `llm-api-key` (optional; without it the gate skips). Outputs:
`pr-number` (from the gate, trusted), `verified` (`'true'` when the verify
job uploaded a result). Jobs:

1. **gate** (no checkout, `actions: read`, `pull-requests: read`): skips
   unless a key and a model are configured, the check run produced a result
   with at least one failing target, the PR is open and still at the checked
   commit, and the author is trusted (or the PR carries the label, or
   `run-for: everyone`).
2. **propose** (holds the key; `contents: read`): checks the PR out into
   `pr/` as data, installs numguard into an isolated virtualenv and runs
   `numguard agent propose`. Nothing from the PR is installed, imported or
   executed. Requires `[agent] enabled = true` in the config.
3. **verify** (no secrets, `contents: read`, no caches): installs the project
   and runs `numguard agent verify`, which executes each candidate against the
   target's checks and adds a `fixes` section (with a diff) to `result.json`.

The caller's second job runs `comment.yml` with the gate's `pr-number` and
`artifact-run-id: ${{ github.run_id }}`; the card then shows a
"🤖 Suggested fixes" section with a collapsible "suggested fix (verified)"
diff.

Repository settings for the agent: secret `NUMGUARD_LLM_API_KEY`; variables
`NUMGUARD_LLM_PROVIDER`, `NUMGUARD_LLM_MODEL` and optionally
`NUMGUARD_LLM_BASE_URL`, `NUMGUARD_LLM_REASONING_EFFORT`.

## The composite action

`uses: bhaskargurram-ai/numguard@v1.0.0` with `command:`

| command | where | needs |
|---|---|---|
| `check` (default) | unprivileged job, after installing your project | `config`, `root`, `fail-on`, `numguard-extras`, `output-dir`; output `status` |
| `summary` | any job | `result` |
| `comment` | privileged `workflow_run` job that never runs PR code | `result`, `pr-number`, `expected-head-sha`, `labels`, `github-token` |
| `agent-propose` | a job that never installs or runs PR code | `config`, `result`, `repo-dir`, `candidates-dir`, `llm-*` |
| `agent-verify` | a job without secrets | `config`, `result`, `repo-dir`, `candidates-dir`, `output-dir` |

`numguard-version` installs a pip requirement instead of the action's own
source. Inputs reach scripts only through `env:`. The reusable workflows above
are the recommended way to combine these safely.

## The card

```
<!-- numguard:card -->
## numguard: ❌ 1 of 5 targets fail numerical edge-case checks
5 targets: 4 pass, 1 fail · pytest: 12 passed, 0 failed
| | target | cases | failing checks | detail |
...
<details><summary>❌ logsumexp — failing checks ...</summary> (check / case / dtypes / rel err / ulp / detail) </details>
### 🤖 Suggested fixes (experimental)      (agent workflow only)
...
<!-- numguard:data {"v":1,"tool":"1.0.0","pr":12,"head":"…","base":"…","status":"fail",
  "targets":[{"name":"logsumexp","status":"fail","failures":9,"checks":["finite","reference"]}],
  "counts":{"pass":4,"fail":1,"error":0},"pytest":{...},"fixes":null,"warnings":0} -->
```

The data block (one line; `-`, `<`, `>`, `&` and `@` appear as `\u`
escapes) is built from validated fields only, capped at 4000 characters, and
only accepted at the very end of the card:
`numguard.carddata.parse_card_data(comment_body)`.
