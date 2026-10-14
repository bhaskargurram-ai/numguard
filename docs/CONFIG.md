# Configuration reference: `numguard.toml`

`numguard.toml` lets numguard check functions without touching their source
(no decorators): `numguard check` reads it from the current directory (or
`--config PATH`), and the GitHub workflow reads it from the pull request's
**base** commit, so a pull request cannot weaken its own checks.

```toml
[defaults]                                  # optional, merged into every target
dtypes = ["float32", "float64"]

[[target]]
name = "logsumexp"                          # optional; default: the function name
function = "mypkg.ops:logsumexp"            # or "path/to/file.py:logsumexp"
inputs = ["any"]                            # one entry per required positional argument
properties = ["translation_equivariant"]
reference = "special.logsumexp(x)"          # an expression over the argument names

[[target]]
function = "mypkg.ops:log1p"
inputs = ["(-1, inf)"]
reference = "mpmath.log1p(x)"
reference_precision = "mpmath"              # elementwise, 50 digits (needs mpmath)
monotonic = "increasing"

[pytest]                                    # optional: also run pytest
args = ["tests/test_numerics.py"]
timeout = 1800

[agent]                                     # optional: suggested fixes (experimental)
enabled = false
```

Validation is strict: an unknown key, a wrong type or an unknown value
(dtype, domain, property, enum) is an error that names the table and key
(`target[2].dtypes: ...`). Keys whose names look like secrets (`api_key`,
`token`, `secret`, `password`) are refused anywhere in the file; the LLM key
is only ever read from `NUMGUARD_LLM_API_KEY`.

## Resolving targets

`function` is `package.module:function` or
`relative/path/file.py:function`. Module names are imported with the
configuration file's directory (and its `src/`) on `sys.path`; file paths
are relative to that directory. `numguard check --root DIR` (and the
workflows) resolve against `DIR` instead. Targets are imported and run
in-process by `numguard check` (your own code), and in a subprocess by the
agent commands.

## `[[target]]` keys

Every key except `function` is optional. Keys that are also
`numguard.check()` arguments mean the same thing (see the docstring of
`numguard.check`).

| key | type | meaning |
|---|---|---|
| `function` | string | the function to check (required) |
| `name` | string | label in reports and for `--target` (default: the function's name; must be unique) |
| `description` | string | free text, ignored |
| `inputs` | list | one entry per required positional argument: a domain string, or a table `{ name, domain, shape, dtype }` |
| `domain` | string | domain for inputs inferred from the signature when `inputs` is absent |
| `dtypes` | list | floating dtypes under test: `float16`, `float32`, `float64`, `bfloat16` (default `float32`, `float64`) |
| `properties` | list | metamorphic properties, e.g. `"translation_invariant"`, `"scale_equivariant:2"` |
| `reference` | string | reference expression (below) |
| `reference_precision` | string | `float64` (default), `longdouble` or `mpmath` |
| `reference_function` | string | instead of `reference`: a trusted callable `module:function`, called like `numguard.check(reference=...)` |
| `rtol`, `atol` | number or table | tolerance, or per output dtype: `rtol = { float32 = 1e-4 }` |
| `nan` | string | `propagate` (default), `raise` or `ignore` |
| `finite` | bool | finite inputs must give finite outputs (default `true`) |
| `allow_nonfinite`, `may_raise`, `exclude`, `include` | list | case names, kinds or tags (globs allowed) |
| `ignore` | list | `"check:case-pattern"` entries downgraded to *skip*, e.g. `"reference:empty"` |
| `bounds` | `[lo, hi]` | every finite output value must lie in this interval |
| `monotonic`, `monotonic_arg` | string, int | `increasing` / `decreasing` in argument `monotonic_arg` (default 0) |
| `batched`, `batch_axis` | bool, int | `f(stack(xs)) == stack(f(x))` along `batch_axis` |
| `fp_signals` | string | `ignore` (default) or `fail` on overflow / invalid / divide-by-zero for finite inputs |
| `out_dtype` | string or list | `"preserve"` (default), `"none"` (no dtype check), a dtype, or one per output |
| `check_mutation` | bool | fail when the function modifies its inputs in place (default `true`) |
| `kwargs` | table | extra keyword arguments for the function (scalars only) |
| `n`, `large_n`, `seed` | int | array length of ordinary cases, size of the `large_n` case, generator seed |
| `ref_upcast` | bool | call a plain reference with float64-upcast inputs (default `true`) |
| `extra_cases` | table | hand-written inputs: `name = [values]` (or a list per argument) |
| `backend` | string | `numpy` (default), `torch`, `jax` or `cupy` (below) |

Input entries: `domain` is `any`, `positive`, `nonneg`, `negative`,
`nonpos`, `unit_interval`, interval notation such as `"[-1, inf)"` or
`"(0, 1]"`, or an integer range `"int[0, 255]"`. `shape` is a list of ints and
symbolic sizes (default `["n"]`; `[]` is a scalar; all symbolic sizes take the
case's size). `dtype` pins an argument (default: the dtype under test).

`numguard check --dtype ...`, `--large-n`, `--seed` and `--backend` override
these keys for every target.

## Reference expressions

`reference` is a Python **expression** over the function's argument names,
for example `"special.logsumexp(x)"`, `"np.hypot(x, y)"`,
`"np.sqrt(np.sum(x * x))"` or `"mpmath.log1p(x)"`. It is checked statically
and evaluated with only these names available:

* `np` / `numpy`, `math`, `special` (`scipy.special`, if SciPy is installed;
  `scipy.special` also works), `mpmath` (if installed);
* the builtins `abs min max sum len range float int bool tuple list zip
  enumerate round pow all any sorted reversed complex`.

Rejected: statements, `import`, lambdas, names starting with `__`,
private attributes, `open`/`eval`/`exec`/`getattr`/..., and NumPy/mpmath
attributes that touch files, the OS or global state (`np.load`, `np.save`,
`tofile`, `memmap`, `lib`, `seterr`, `mp`, ...). In your own config the
expression is compiled in-process; an expression written by the agent is
only ever evaluated in the subprocess worker.

`reference_precision`:

* `float64`: the expression gets float64-upcast arrays;
* `longdouble`: it gets `np.longdouble` arrays (80-bit on most x86-64 Linux
  builds; the same as float64 elsewhere);
* `mpmath`: it is evaluated **elementwise**, each name bound to an
  `mpmath.mpf` scalar at 50 digits. Use it for elementwise functions
  (`mpmath.log1p(x)`, `mpmath.exp(x) / (1 + mpmath.exp(x))`), not reductions.

A wrong reference produces wrong failures. `numguard suggest` therefore
checks the reference against the function on benign inputs and calls it
*suspect* when they disagree there; when you write references by hand, look
at failures on ordinary (`random`, `ones`, `sorted`, ...) cases first.

## `[pytest]`

```toml
[pytest]
args = ["tests", "-k", "numerics", "--numguard-modules", "mypkg.ops"]
timeout = 1800      # seconds
enabled = true
```

`numguard check --config` (without `--target`) then also runs
`python -m pytest <args>` in the root directory, with the numguard plugin
active, and records passed / failed / errors / skipped from its JUnit report.
A failing or crashing pytest run makes the overall status `fail` / `error`.
`--no-pytest` skips it.

## `[agent]`

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | `numguard agent propose` (the CI agent workflow) does nothing unless `true` |
| `rounds` | 3 | refinement rounds of local `numguard fix` (when `--rounds` is not given) |
| `max_tool_calls` | 25 | read-only tool calls per model conversation (`fix`, `agent propose`) |
| `max_targets` | 3 | failing targets the CI agent tries per pull request |
| `variants` | 3 | independent rewrites the CI agent proposes per target (1–10); the verify job runs all and keeps the best |
| `timeout` | 120 | seconds for each sandboxed check run in `agent verify` |

See [AGENT.md](AGENT.md).

## Backends (PyTorch, JAX, CuPy)

`backend = "torch"` (or `--backend torch`, or `pytest --numguard-backend
torch`) checks a function that takes and returns that library's arrays.
numguard still generates every case and compares every result in NumPy: an
adapter converts the inputs to the backend before each call and converts the
outputs back (`np.from_dlpack` / `np.asarray` / the backend's export),
recording the backend dtype of each output for the `dtype` check. The
reference is always called with NumPy arrays.

* `torch`: CPU tensors by default (`NUMGUARD_TORCH_DEVICE=cuda` moves them to
  a GPU); in-place modification of inputs is detected.
* `jax`: arrays are immutable (the `mutation` check always passes). JAX
  computes in float32 unless `JAX_ENABLE_X64=1`; the `dtype` check reports
  float64 cases that silently became float32.
* `cupy`: GPU arrays; no bfloat16.

These libraries are optional and not installed with numguard; a missing one
is an error. numguard's own test suite exercises the adapters with fake
`torch` and `jax` modules (the real libraries are not installed in CI).

### bfloat16

`dtypes = ["bfloat16"]` uses bfloat16's own limits (max about 3.39e38, eps
2^-7, smallest normal 2^-126, smallest subnormal 2^-133) to build cases, stores
them as float32 values that are exactly representable in bfloat16, and hands
them to the function as `torch.bfloat16`, `jax.numpy.bfloat16` or (NumPy
backend) `ml_dtypes.bfloat16` arrays (`pip install 'numguard[bfloat16]'`).
Default tolerance: `rtol = 6e-2`; ULP errors are in bfloat16 units.

## Hypothesis strategies

`numguard.strategies` (optional; `pip install 'numguard[hypothesis]'`) turns a
spec into Hypothesis strategies:

```python
from hypothesis import given
import numguard.strategies as ngst

@given(ngst.inputs_for(my_softmax, dtypes=["float32"]))
def test_softmax_sums_to_one(args):
    (x,) = args
    ...
```

* `cases(target, **spec)` draws `numguard.Case` objects (name, kind, tags,
  fresh copies of the arrays) from the edge-case catalogue;
* `edge_cases(...)` draws argument tuples from the catalogue;
* `random_inputs(..., min_size=0, max_size=32)` draws random in-domain arrays
  (bounds rounded inwards to the storage precision; no NaN or infinities);
* `inputs_for(...)` draws from both.

`target` is a `numguard.Spec`, a function decorated with `@numguard.guard`,
or a function plus the keyword arguments of `numguard.check`.
