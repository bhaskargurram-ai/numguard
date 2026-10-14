"""Backend adapters (torch / jax / cupy) and the emulated bfloat16 dtype."""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers  # noqa: F401 - sets up sys.path
from tests.fake_backends import installed, make_fake_jax, make_fake_torch

import numguard
from numguard.backends import BackendCall, BackendError, get_backend, wrap
from numguard.cases import ArraySpec, generate_cases
from numguard.dtypes import bf16_bits, finfo, nextafter_in, round_bf16
from numguard.reference import default_tolerance, ulp_distance


def _ref_lse(x):
    m = np.max(x) if x.size else -np.inf
    if x.size == 0:
        return np.float64(-np.inf)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(x - m)))


class TorchBackendTests(unittest.TestCase):
    def setUp(self):
        self.torch = make_fake_torch()

    def test_naive_torch_logsumexp_fails_stable_passes(self):
        torch = self.torch

        def naive(x):
            return torch.log(torch.sum(torch.exp(x)))

        def stable(x):
            if x._a.size == 0:
                return x._wrap(np.array(-np.inf))
            m = torch.max(x)
            if not np.isfinite(m._a):
                return m
            return m + torch.log(torch.sum(torch.exp(x - m)))

        with installed(torch=torch):
            bad = numguard.check(naive, reference=_ref_lse, backend="torch")
            good = numguard.check(stable, reference=_ref_lse, backend="torch")
        self.assertFalse(bad.ok)
        self.assertIn("finite", bad.failed_checks())
        self.assertEqual(bad.backend, "torch")
        self.assertIn("backend: torch", bad.summary())
        self.assertTrue(good.ok, good.summary())

    def test_bfloat16_dtype_preserved_and_upcast_detected(self):
        torch = self.torch

        def absval(x):
            return torch.abs(x)

        def upcasting_abs(x):
            return torch.abs(x).to(torch.float32)

        with installed(torch=torch):
            ok = numguard.check(absval, reference=np.abs, dtypes=["bfloat16"], backend="torch")
            bad = numguard.check(upcasting_abs, reference=np.abs, dtypes=["bfloat16"], backend="torch")
        self.assertTrue(ok.ok, ok.summary())
        self.assertEqual(ok.dtypes, ["bfloat16"])
        self.assertEqual(bad.failed_checks(), ["dtype"])
        self.assertIn("got float32, expected bfloat16", bad.failures[0].message)

    def test_mutation_detected_through_backend(self):
        torch = self.torch

        def inplace(x):
            return x.add_(1.0)

        with installed(torch=torch):
            for dt in ("float32", "bfloat16"):  # shared memory / converted copy
                rep = numguard.check(inplace, dtypes=[dt], backend="torch", exclude=["nonfinite_input"])
                self.assertIn("mutation", rep.failed_checks(), dt)

    def test_torch_not_installed(self):
        with installed(torch=None):
            import sys

            sys.modules.pop("torch", None)
            with self.assertRaises(BackendError):
                get_backend("torch")
        with self.assertRaises(BackendError):
            get_backend("tensorflow")
        with self.assertRaises(ValueError):
            numguard.check(np.abs, backend="tensorflow")


class JaxBackendTests(unittest.TestCase):
    def test_silent_float32_downcast_is_a_dtype_failure(self):
        jax, jnp = make_fake_jax()

        def jabs(x):
            return jnp.abs(x)

        with installed(jax=jax, **{"jax.numpy": jnp}):
            rep = numguard.check(jabs, dtypes=["float32", "float64"], backend="jax",
                                 exclude=["overflow_prone", "nonfinite_input"])
        self.assertEqual(rep.failed_checks(), ["dtype"])
        self.assertTrue(all(r.dtype == "float64" for r in rep.failures))
        # jax arrays are immutable: no mutation is ever reported
        self.assertNotIn("mutation", rep.failed_checks())


class NumpyBackendTests(unittest.TestCase):
    def test_numpy_backend_is_a_no_op(self):
        f = np.abs
        self.assertIs(wrap(f, "numpy", ["float32"]), f)

    def test_bfloat16_with_numpy_needs_ml_dtypes(self):
        from numguard.dtypes import ml_dtypes_bfloat16

        if ml_dtypes_bfloat16() is not None:  # pragma: no cover - ml_dtypes installed
            self.skipTest("ml_dtypes is installed")
        with self.assertRaises(BackendError):
            numguard.check(np.abs, dtypes=["bfloat16"])

    def test_backend_call_converts_tuple_and_dict_outputs(self):
        torch = make_fake_torch()
        with installed(torch=torch):
            be = get_backend("torch")
            call = BackendCall(lambda x: {"b": x, "a": (x + 1.0)}, be)
            call.set_case(["bfloat16"])
            out = call(np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(sorted(out), ["a", "b"])
        self.assertEqual(call.last_out_dtypes(), ["bfloat16", "bfloat16"])
        self.assertEqual(out["a"].dtype, np.float32)


class BFloat16Tests(unittest.TestCase):
    def test_rounding_and_bits(self):
        self.assertEqual(float(round_bf16(np.float32(1.0 + 2 ** -9))), 1.0)
        self.assertEqual(float(round_bf16(np.float32(1.0 + 3 * 2 ** -9))), 1.0 + 2 ** -7)
        self.assertEqual(int(bf16_bits(np.float32(1.0))), 0x3F80)
        self.assertTrue(np.isnan(round_bf16(np.float32("nan"))))
        self.assertTrue(np.isinf(round_bf16(np.float32(3.4e38))))  # beyond bf16 max -> inf
        self.assertEqual(nextafter_in(1.0, "bfloat16", 2.0), 1.0 + 2 ** -7)
        self.assertEqual(nextafter_in(0.0, "bfloat16", 1.0), finfo("bfloat16").smallest_subnormal)

    def test_ulp_and_tolerance(self):
        self.assertEqual(float(ulp_distance(1.0, 1.0 + 2 ** -7, "bfloat16")), 1.0)
        self.assertEqual(float(ulp_distance(-0.0, 0.0, "bfloat16")), 0.0)
        rtol, tiny = default_tolerance("bfloat16")
        self.assertGreater(rtol, default_tolerance("float16")[0])
        self.assertEqual(tiny, 2.0 ** -126)

    def test_cases_are_bfloat16_representable(self):
        cases = generate_cases([ArraySpec()], ["bfloat16"], large_n=64)
        kinds = {c.kind for c in cases}
        self.assertTrue({"huge", "subnormal", "tiny", "small"} <= kinds)
        for c in cases:
            a = c.args[0]
            self.assertEqual(a.dtype, np.float32)
            np.testing.assert_array_equal(round_bf16(a), a, err_msg=c.name)
        huge = next(c for c in cases if c.kind == "huge").args[0]
        self.assertLessEqual(float(np.max(np.abs(huge))), finfo("bfloat16").max)
        self.assertGreater(float(np.max(np.abs(huge))), 1e37)


if __name__ == "__main__":
    unittest.main()
