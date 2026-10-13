# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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
