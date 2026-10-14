# Contributing to numguard

Thanks for your interest. Bug reports, new edge-case kinds, new checks and new
corpus tasks are all welcome.

## Development setup

```bash
git clone <your fork> numguard && cd numguard
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite with either runner. Both discover the same `unittest` test
cases. The pytest-plugin tests are skipped when pytest isn't installed.

```bash
python -m unittest discover -s tests -v
pytest
```

See docs/TESTING.md for the test levels. Both corpus baselines must exit
with status 0:

```bash
numguard eval --baseline reference   # every reference implementation passes
numguard eval --baseline buggy       # every buggy implementation is detected
```

## Guidelines

- Keep runtime dependencies to NumPy and click (plus `tomli` on Python 3.10).
  Anything else (SciPy, mpmath, pytest, Hypothesis, PyTorch/JAX/CuPy,
  ml_dtypes) must stay optional and degrade gracefully when it's missing. The
  LLM client uses only the standard library.
- Agent tests use the scripted fake model in `tests/fake_llm.py`; no test may
  touch the network. Model-written code may only ever run through
  `numguard.agent.sandbox.run_worker`.
- Workflow changes must keep `tests/test_workflows.py` passing (see
  docs/SECURITY_MODEL.md); if you change `numguard init`, regenerate
  `examples/` with `numguard init --agent` in an empty directory.
- Add type hints and docstrings to public functions.
- Keep the generator deterministic. Derive every random draw from the `seed`
  argument.
- Tests use `unittest.TestCase`, so they run without pytest.
- Every change in behaviour needs a test that fails without it.

## Adding a corpus task

Create `corpus/<task_name>/` with these files:

1. `task.md`: the prompt an agent receives. Give the exact function name and
   signature, and state the contract explicitly: dtype behaviour, NaN/inf
   behaviour, what happens on empty input. Grade only what the prompt promises.
2. `buggy.py`: a plausible naive implementation that passes the happy-path
   test. It should be the kind of code a competent developer or an AI assistant
   would write on a first attempt, not something deliberately sabotaged.
3. `reference.py`: a correct implementation.
4. `spec.py`, which defines:
   - `FUNCTION = "<function name>"`
   - `options(reference) -> dict`: keyword arguments for `numguard.check`.
     Prefer an independent oracle (for example `numguard.longdouble(np.log1p)`)
     over the bundled reference when one exists.
   - `happy_path(fn)`: a simple example-based test that both versions pass.
5. An `__init__.py` (a one-line docstring is enough), and an entry in the `packages` list in
   `pyproject.toml` (`numguard._corpus.<task_name>`).
6. Add the task to `EXPECTED_TASKS` and `EXPECTED_FAILURES` in
   `tests/test_corpus.py`.

Be honest about provenance. Describe tasks as *inspired by* a bug class unless
you can point to a specific public, merged fix.

## Commit messages

Use the imperative mood ("Add hypot task", not "Added hypot task") and keep
commits focused.
