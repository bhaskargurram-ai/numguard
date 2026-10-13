# numguard

**Numerical edge-case testing for array code.** numguard throws the inputs that
break numerical code at your function (all zeros, empty arrays, NaN, ±inf, values
near the overflow threshold, subnormals, signed zero, large-offset data, float16,
batches) and checks the answers against a reference, against invariants, and
against the dtype's limits.

It's aimed at a kind of bug that AI coding agents (and people) write all the time:
code that runs, passes the obvious test, and returns silently wrong numbers on
edge cases.

```python
import numpy as np
import scipy.special
import numguard

def logsumexp(x):                       # what an assistant often writes first
    return np.log(np.sum(np.exp(x)))

assert np.isclose(logsumexp(np.array([1.0, 2.0, 3.0])), 3.40760596)  # happy path: passes

report = numguard.check(logsumexp, reference=scipy.special.logsumexp,
                        properties=["translation_equivariant"])
print(report.summary())
```

Actual output (`examples/readme_logsumexp.py`):

```text
numguard: logsumexp FAILED - 230 passed, 16 failed (46 cases; dtypes: float32, float64; reference: logsumexp)
+-----------+--------------+-----------------+---------+---------+------------------------------------------------------------+
| check     | case         | dtypes          | rel err | ulp     | detail                                                     |
+-----------+--------------+-----------------+---------+---------+------------------------------------------------------------+
| finite    | huge         | float32,float64 | -       | -       | finite input gave 1/1 non-finite (e.g. inf) [fp: overflow] |
| reference | huge         | float32,float64 | inf     | 5.2e+15 | got inf, reference 1.40244e+38 [fp: overflow]              |
| finite    | large        | float32,float64 | -       | -       | finite input gave 1/1 non-finite (e.g. inf) [fp: overflow] |
| reference | large        | float32,float64 | inf     | 2.3e+18 | got inf, reference 1.43262e+20 [fp: overflow]              |
| finite    | cancellation | float32,float64 | -       | -       | finite input gave 1/1 non-finite (e.g. inf) [fp: overflow] |
| reference | cancellation | float32,float64 | inf     | 4.5e+18 | got inf, reference 46345.6 [fp: overflow]                  |
| finite    | wide_range   | float32,float64 | -       | -       | finite input gave 1/1 non-finite (e.g. inf) [fp: overflow] |
| reference | wide_range   | float32,float64 | inf     | 1.5e+17 | got inf, reference 2.08048e+37 [fp: overflow]              |
+-----------+--------------+-----------------+---------+---------+------------------------------------------------------------+
```

The `cancellation` row means the naive version returns `inf` for ordinary data
with a large offset, such as `46341 + [0, 1, 2, 3, 4]` in float32 or
`2**30 + [0, ..., 4]` in float64 (timestamps, for example), where the correct
answers are about `46345.6` and `1.07e9`.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start): [Python API](#python-api), [pytest plugin](#pytest-plugin), [CLI](#cli)
- [Edge cases](#edge-cases)
- [Checks](#checks)
- [High-precision references and ULPs](#high-precision-references-and-ulps)
- [Eval mode: the bug corpus](#eval-mode-the-bug-corpus)
- [Roadmap](#roadmap)
- [Contributing](#contributing) · [License](#license)

## Install

numguard isn't published on PyPI yet. Install it from a checkout:

```bash
pip install -e .                     # numpy + click only
pip install -e ".[pytest]"           # + pytest (for the plugin)
pip install -e ".[mpmath]"           # + arbitrary-precision references
pip install -e ".[dev]"              # pytest, scipy, mpmath (for working on numguard)
```

numguard needs Python 3.10 or newer and NumPy. The only other runtime dependency
is click, for the CLI.

## Quick start

### Python API

```python
import numpy as np
import numguard

report = numguard.check(
    my_variance,                                  # function under test
    inputs=numguard.Array("n", domain="any"),     # optional; inferred from the signature
    reference=np.var,                             # optional trusted implementation
    properties=["translation_invariant", "scale_equivariant:2", "permutation_invariant"],
    dtypes=["float16", "float32", "float64"],     # default: float32, float64
    rtol=None, atol=None,                         # default: per-dtype tolerances
)
report.ok              # bool
report.failures        # list[CheckResult]: check, case, dtype, message, max abs/rel/ULP error, fp events
print(report.summary())
report.to_json()
report.raise_if_failed()   # raises numguard.NumGuardError (an AssertionError) with the table
```

Inputs are described per argument: a shape (ints, or symbolic sizes like `"n"`
that take each case's size), a **domain** (`any`, `positive`, `nonneg`,
`negative`, `nonpos`, `unit_interval`, interval notation like `"(-1, inf)"`, or
integer ranges like `"int[0, 255]"`), and optionally a fixed dtype. With no
`inputs`, numguard creates one 1-D array per required positional parameter.

```python
numguard.check(hypot, inputs={"x": "any", "y": "any"}, properties=["symmetric"])
numguard.check(log1p, domain="(-1, inf)", monotonic="increasing")
numguard.check(softmax, batched=True, bounds=(0, 1))      # f(stack(xs)) == stack(f(x))
numguard.check(lookup, inputs=[numguard.Array("n"), numguard.Array("n", domain="int[0, 9]")])
```

Other options: `nan="propagate" | "raise" | "ignore"`, `allow_nonfinite=[...]`
and `may_raise=[...]` (case names, kinds, tags or globs), `out_dtype=` (default
`"preserve"`), `fp_signals="fail"` (treat overflow/invalid/divide-by-zero
signals on finite inputs as failures), `postconditions=[fn]`,
`extra_cases={"name": arrays}`, `include=`/`exclude=`, `ignore=["check:pattern"]`,
`kwargs=` (forwarded to the function), `n=`, `large_n=`, `seed=`.

**Decorator.** `@numguard.guard(...)` takes the same arguments and attaches the
spec to the function without changing its behaviour:

```python
@numguard.guard(reference=scipy.special.softmax, batched=True, bounds=(0, 1),
                properties=["translation_invariant"])
def softmax(x):
    ...

softmax.numguard_check().raise_if_failed()     # or let the pytest plugin collect it
```

### pytest plugin

Installing numguard registers a pytest plugin (`pytest11` entry point). You can
use it in two ways.

**Marker.** The test body runs first, then numguard checks `fn`. If anything
fails, the test fails with the summary table as its message:

```python
import pytest, scipy.special
from mypkg.ops import logsumexp

@pytest.mark.numguard(fn=logsumexp, reference=scipy.special.logsumexp,
                      properties=["translation_equivariant"])
def test_logsumexp_edge_cases():
    pass
```

**Auto-collection.** Every `@numguard.guard`-decorated function in the listed
modules becomes a test item:

```bash
pytest --numguard-modules mypkg.ops,mypkg.stats
pytest --numguard-dtypes float16,float32 --numguard-large-n 1000000
```

(`numguard_modules` can also be set in `[tool.pytest.ini_options]`.)

### CLI

```bash
numguard check mymodule:func --ref other:func --dtype float32 --dtype float64 \
    --prop translation_invariant --domain any        # prints the report, exit 1 on failure
numguard check mymodule:func --json                  # machine-readable
numguard check path/to/file.py:func --batched --fp-signals fail
numguard cases                                       # list edge-case kinds
python -m numguard --help
```

If the target has a spec attached with `@numguard.guard`, the CLI uses it, and
any command-line options override it.

A real example: with NumPy 2.5.3, `numguard.check(np.var, dtypes=["float16"])`
fails on the `large_n` case. `np.var` of 100,000 float16 values between 1 and 2
returns `inf`, because the reduction overflows the float16 accumulator. The true
variance is about 0.083.

**Limits to keep in mind.** Without a reference, the `finite` check can't tell a
result that legitimately overflows (such as `hypot(1.7e308, 1.7e308)`) from a bug.
Pass a reference, which is rounded to the output dtype before comparing, or
mark such cases with `allow_nonfinite`. numguard tests the cases it generates and
the contract you declare. It does not prove correctness.

## Edge cases

Each case gets a name, a kind, the input arrays, and tags. Cases are generated
for every dtype under test and are fully deterministic for a given seed.
Values that fall outside an argument's domain are never generated: a `positive`
argument never receives zeros or negatives, and on `(-1, inf)` the `huge` values
are positive. When a function has several arguments, each value kind is also
applied to one argument at a time (`zeros[b]`).

| kind | what it contains | tags |
|---|---|---|
| `random` | seeded normal (or in-domain) values | regular |
| `zeros`, `neg_zero` | all `0.0` / all `-0.0` | regular / signed_zero |
| `ones`, `constant` | constant arrays | regular |
| `empty`, `single`, `pair`, `large_n` | sizes 0, 1, 2 and `large_n` (default 100,000) | size |
| `nan`, `posinf`, `neginf` | one non-finite value among regular ones | nonfinite_input |
| `huge` | 25-50% of `finfo.max` | extreme, overflow_prone |
| `large` | a few × `sqrt(finfo.max)`, so squares overflow | extreme, overflow_prone |
| `small` | 1-100 × `eps` | extreme, precision |
| `tiny`, `subnormal` | just above `finfo.tiny` / below it | extreme, underflow_prone |
| `wide_range` | geometric sweep from `tiny` to `max` | extreme |
| `cancellation` | large offset + small spread: `16/sqrt(eps) + [0..4]` (`2**30 + ...` in float64) | extreme, precision |
| `alternating` | `1, -2, 3, -4, ...` | regular, mixed_sign |
| `duplicates`, `sorted`, `reverse_sorted` | orderings and ties | regular |
| `boundary` | closed/open interval endpoints (via `nextafter`) | |
| `int_min`, `int_max` | integer range limits | boundary |

Patterns in `include`, `exclude`, `allow_nonfinite`, `may_raise` and `ignore`
match a case name (glob), its kind, or any of its tags.

## Checks

| check | fails when |
|---|---|
| `error` | the function raises (unless the case matches `may_raise`, or the reference raises too) |
| `mutation` | the function modifies its input arrays in place |
| `dtype` | a floating output does not keep the input dtype (or does not match `out_dtype`) |
| `finite` | finite inputs produce inf/NaN that the reference (rounded to the output dtype) does not |
| `nan` | `nan="propagate"`: NaN input gives an output with no NaN. `nan="raise"`: no exception |
| `reference` | values differ from the reference beyond `rtol`/`atol` (reports max abs/rel error and ULPs) |
| `fp_signals` | opt-in: overflow / invalid / divide-by-zero signal raised on finite input |
| `bounds` | opt-in: finite output outside `bounds=(lo, hi)` |
| `post:<name>` | opt-in: a user postcondition `post(args, outputs)` returns False or a message |
| `batch` | `batched=True`: `f(stack(xs, axis)) != stack(f(x) for x in xs)` |
| `monotonic` | `monotonic="increasing"/"decreasing"`: the order is violated on sorted inputs |
| `prop:<name>` | a metamorphic property fails (see below) |

Floating-point exceptions raised while the function runs are recorded on every
result (`[fp: overflow]` in the table), as evidence. They only cause failures
when `fp_signals="fail"` is set.

**Properties.** These can be given as strings, with an optional `:value`, or as
`numguard.Property` objects:
`permutation_invariant`, `permutation_equivariant`, `translation_invariant[:c]`,
`translation_equivariant[:c]`, `scale_invariant[:factor]`,
`scale_equivariant[:exponent]` (the default factor of 2 is exact in binary FP), `symmetric`,
`odd`, `even`. They're checked on regular cases. A transformed input that would
leave the declared domain is skipped.

**Default tolerances.** rtol is 1e-2 for float16, 1e-5 for float32 and 1e-12
for float64. Relative errors are measured against `max(|expected|, finfo.tiny)`.
Pass `rtol`/`atol` as numbers, or as dicts keyed by dtype name such as
`{"float32": 1e-4, "uint8": 1}`.

## High-precision references and ULPs

A plain reference gets float64-upcast inputs. To get a reference that is
genuinely more accurate than float64:

```python
ref = numguard.longdouble(np.log1p)           # evaluates on np.longdouble inputs
ref = numguard.highprec(numpy_fn=np.log1p,    # mpmath if installed, else longdouble
                        mp_fn=lambda x: mpmath.log1p(x), dps=50)
numguard.check(my_log1p, reference=ref, domain="(-1, inf)")
```

`np.longdouble` is 80-bit extended precision on most x86-64 Linux builds. On
platforms where it is plain float64 (Windows, macOS arm64), numguard falls back
to float64 and labels the reference that way. `numguard.ulp_distance(a, b)`
counts the representable values between `a` and `b` (signed zeros count as
equal, and so do two NaNs).

## Eval mode: the bug corpus

`corpus/` contains 12 self-contained tasks. Each one is **inspired by a real
class of numerical bug** that shows up in AI-written and human-written array
code. Each task folder contains:

- `task.md`: the prompt a coding agent receives (function name, signature, contract)
- `buggy.py`: a plausible naive implementation that **passes a happy-path test**
- `reference.py`: a correct implementation
- `spec.py`: the numguard spec (inputs, domain, properties, reference, happy-path test)

| task | bug class |
|---|---|
| `quantize_dynamic` | dynamic uint8 quantization: all-zero input gives `scale = 0`, which turns into NaN downstream |
| `logsumexp`, `softmax` | `exp` overflow on large logits; softmax normalized over the whole batch |
| `variance_onepass` | `E[x²] - E[x]²` cancellation: `1e9 + [1, 2, 3]` gives `0.0` in float64 and `-9.2e10` in float32 |
| `mean_float16` | float16 accumulator overflows on 100k elements |
| `sample_std` | `n = 1` gives NaN instead of an error, plus overflow/underflow of squared deviations |
| `log1p` | `log(1 + x)` loses all precision for tiny `x` |
| `cosine_similarity`, `normalize` | division by a zero norm, and overflow in `‖x‖²` |
| `sigmoid` | `exp(-x)` overflows for very negative inputs |
| `hypot` | `sqrt(x² + y²)` overflows and underflows |
| `loo_mean` | closed-form leave-one-out mean cancels catastrophically when one value dominates, and `n = 1` |

The quantization task is modelled on the all-zero-input case of ONNX
`DynamicQuantizeLinear`, where the textbook formula gives a zero scale. The
author contributed a fix for this case to the ONNX specification. The other
tasks are generic bug classes that don't refer to specific upstream issues.

Grade the bundled versions:

```bash
numguard eval --baseline buggy        # exit 0 only if every buggy version is caught
numguard eval --baseline reference    # exit 0 only if every reference passes
```

Real output from `numguard eval --baseline buggy`:

```text
+-------------------+-----------------+----------+----------------------------------------------------+-------------------------------+
| task              | happy-path test | numguard | failed checks                                      | failing cases                 |
+-------------------+-----------------+----------+----------------------------------------------------+-------------------------------+
| cosine_similarity | pass            | FAIL     | finite, reference                                  | zeros, zeros[a], zeros[b] +23 |
| hypot             | pass            | FAIL     | finite, reference                                  | huge, huge[x], huge[y] +8     |
| log1p             | pass            | FAIL     | reference                                          | small, tiny, wide_range       |
| logsumexp         | pass            | FAIL     | finite, reference                                  | huge, large, cancellation +1  |
| loo_mean          | pass            | FAIL     | finite, reference                                  | single, huge, large +2        |
| mean_float16      | pass            | FAIL     | finite, reference                                  | empty, large_n, huge          |
| normalize         | pass            | FAIL     | finite, reference                                  | zeros, neg_zero, huge +6      |
| quantize_dynamic  | pass            | FAIL     | finite, reference, fp_signals, post:scale_positive | zeros, neg_zero, subnormal +1 |
| sample_std        | pass            | FAIL     | finite, reference                                  | empty, single, huge +4        |
| sigmoid           | pass            | FAIL     | fp_signals                                         | huge, large, wide_range       |
| softmax           | pass            | FAIL     | finite, reference, batch                           | random, zeros, ones +11       |
| variance_onepass  | pass            | FAIL     | finite, reference, bounds                          | empty, huge, large +3         |
+-------------------+-----------------+----------+----------------------------------------------------+-------------------------------+
baseline:buggy: numguard passed 0/12 tasks; happy-path tests passed 12/12
```

Real output from `numguard eval --baseline reference`:

```text
+-------------------+-----------------+----------+---------------+---------------+
| task              | happy-path test | numguard | failed checks | failing cases |
+-------------------+-----------------+----------+---------------+---------------+
| cosine_similarity | pass            | pass     |               |               |
| hypot             | pass            | pass     |               |               |
| log1p             | pass            | pass     |               |               |
| logsumexp         | pass            | pass     |               |               |
| loo_mean          | pass            | pass     |               |               |
| mean_float16      | pass            | pass     |               |               |
| normalize         | pass            | pass     |               |               |
| quantize_dynamic  | pass            | pass     |               |               |
| sample_std        | pass            | pass     |               |               |
| sigmoid           | pass            | pass     |               |               |
| softmax           | pass            | pass     |               |               |
| variance_onepass  | pass            | pass     |               |               |
+-------------------+-----------------+----------+---------------+---------------+
baseline:reference: numguard passed 12/12 tasks; happy-path tests passed 12/12
```

All 12 buggy versions pass their happy-path test, and numguard catches all 12.

### Grading any coding agent

The harness is agent-agnostic. It only reads files, so any agent (or a person)
can be graded:

1. For each task, give the agent the contents of `task.md`
   (`numguard tasks --show <task>` prints it).
2. Save the agent's answer as `DIR/<task>/solution.py`. The file has to define
   the function named in the task.
3. Run `numguard eval --candidate-dir DIR [--json-out results.json]`.

Missing solutions are reported as `missing`. Solutions that fail to import are
reported as `error`. The JSON output includes the failed checks and cases for
every task, and `--json-out` also writes the full per-case reports. Use
`--task NAME` (repeatable) to grade a subset, and `--solution-name` if the agent
writes to another file name. Note that `solution.py` is imported and run
in-process: grade untrusted code inside a sandbox or container.

## Roadmap

- **Other array libraries** via the [Python array API standard](https://data-apis.org/array-api/):
  run the same cases against JAX, PyTorch and CuPy functions.
- **Accelerator dtypes**: bfloat16 and float8 variants, plus TF32-style reduced-precision matmul.
- **Hypothesis integration**: numguard cases as explicit examples, plus a strategy
  for domain-aware arrays.
- **Gradients**: check `jax.grad`/`torch.autograd` outputs at domain boundaries and support edges.
- **A GitHub Action** that runs `numguard check`/`eval` and posts the table on pull requests.
- More corpus tasks (matrix ops, special functions, quantile/median, numerically
  stable losses).

## Contributing

Contributions are welcome, especially new corpus tasks and edge-case kinds. See
[CONTRIBUTING.md](CONTRIBUTING.md). In short: `pip install -e ".[dev]"`, then
`python -m unittest discover -s tests -v` (or `pytest`), and make sure
`numguard eval --baseline reference` and `numguard eval --baseline buggy` both
exit 0.

## License

MIT © 2026 Bhaskar Gurram. See [LICENSE](LICENSE).
