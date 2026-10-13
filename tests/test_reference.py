"""Tests for ULP computation and high-precision references."""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers  # noqa: F401

from numguard.reference import (
    HAS_MPMATH,
    LONGDOUBLE_IS_EXTENDED,
    HighPrecReference,
    default_tolerance,
    highprec,
    longdouble,
    mp_scalar,
    ulp,
    ulp_distance,
)


class UlpTests(unittest.TestCase):
    def test_adjacent_values_are_one_ulp_apart(self):
        for dt in (np.float16, np.float32, np.float64):
            one = dt(1)
            nxt = np.nextafter(one, dt(2))
            self.assertEqual(float(ulp_distance(one, nxt)), 1.0, dt)
            self.assertEqual(float(ulp_distance(nxt, one)), 1.0, dt)
            self.assertEqual(float(ulp_distance(one, one)), 0.0, dt)

    def test_many_steps(self):
        x = np.float32(3.0)
        y = x
        for _ in range(17):
            y = np.nextafter(y, np.float32(10))
        self.assertEqual(float(ulp_distance(x, y)), 17.0)

    def test_signed_zero_and_sign_crossing(self):
        self.assertEqual(float(ulp_distance(0.0, -0.0)), 0.0)
        tiny = np.float32(np.finfo(np.float32).smallest_subnormal)
        self.assertEqual(float(ulp_distance(-tiny, tiny)), 2.0)
        self.assertEqual(float(ulp_distance(-tiny, np.float32(0))), 1.0)

    def test_nan_and_inf(self):
        self.assertEqual(float(ulp_distance(np.nan, np.nan)), 0.0)
        self.assertTrue(np.isinf(ulp_distance(np.nan, 1.0)))
        fmax = np.finfo(np.float64).max
        self.assertEqual(float(ulp_distance(fmax, np.inf)), 1.0)
        # float16 has 2**15 - 1024 finite magnitudes +1 for inf on each side
        self.assertEqual(float(ulp_distance(-np.inf, np.inf, np.float16)), 2 * 31744.0)

    def test_vectorized_and_dtype_argument(self):
        a = np.array([1.0, 2.0, 4.0])
        b = np.array([1.0, np.nextafter(2.0, 3.0), 4.0])
        np.testing.assert_array_equal(ulp_distance(a, b), [0.0, 1.0, 0.0])
        # rounding to float32 first makes 1 and 1 + 1e-12 identical
        self.assertEqual(float(ulp_distance(1.0, 1.0 + 1e-12, np.float32)), 0.0)
        with self.assertRaises(TypeError):
            ulp_distance(np.array([1], dtype=np.int32), np.array([2], dtype=np.int32))

    def test_ulp_size(self):
        self.assertEqual(float(ulp(1.0)), np.finfo(np.float64).eps)
        self.assertEqual(float(ulp(np.float32(1.0))), np.finfo(np.float32).eps)

    def test_default_tolerances(self):
        self.assertEqual(default_tolerance("float32")[0], 1e-5)
        self.assertEqual(default_tolerance("float16")[1], float(np.finfo(np.float16).tiny))
        self.assertEqual(default_tolerance("int32"), (0.0, 0.0))


class HighPrecTests(unittest.TestCase):
    def test_longdouble_wrapper_runs_in_extended_precision(self):
        seen = {}

        def probe(x):
            seen["dtype"] = x.dtype
            return x

        ref = longdouble(probe)
        self.assertIsInstance(ref, HighPrecReference)
        ref(np.array([1.0], dtype=np.float32))
        self.assertEqual(seen["dtype"], np.dtype(np.longdouble))
        self.assertEqual(ref.__name__, "probe")

    @unittest.skipUnless(LONGDOUBLE_IS_EXTENDED, "np.longdouble is float64 on this platform")
    def test_longdouble_is_more_accurate_than_float64(self):
        x = np.array([1e-10])
        exact = 9.9999999995e-11  # log1p(1e-10) to 11 digits
        naive64 = np.log(1 + x)[0]
        ld = longdouble(lambda v: np.log(1 + v))(x)[0]
        self.assertGreater(abs(naive64 - exact) / exact, 1e-8)
        self.assertLess(abs(float(ld) - exact) / exact, 1e-8)

    def test_highprec_falls_back_without_mpmath(self):
        ref = highprec(numpy_fn=np.log1p, mp_fn=lambda x: x)
        if HAS_MPMATH:
            self.assertIn("mpmath", ref.precision)
        else:
            self.assertIn("longdouble", ref.precision)
        np.testing.assert_allclose(np.asarray(ref(np.array([0.5])), dtype=np.float64),
                                   np.log1p([0.5]))

    def test_highprec_requires_something(self):
        with self.assertRaises(ValueError):
            highprec()
        if not HAS_MPMATH:
            with self.assertRaises(ImportError):
                highprec(mp_fn=lambda x: x)
            with self.assertRaises(ImportError):
                mp_scalar(lambda x: x)

    @unittest.skipUnless(HAS_MPMATH, "mpmath not installed")
    def test_mp_scalar(self):  # pragma: no cover - environment dependent
        import mpmath

        ref = mp_scalar(lambda x: mpmath.log1p(x), dps=40)
        out = ref(np.array([1e-10, 0.5]))
        self.assertAlmostEqual(float(out[0]) / 9.9999999995e-11, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
