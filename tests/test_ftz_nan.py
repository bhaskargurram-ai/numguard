"""Flush-to-zero differences and NaN handling that the reference shares are not failures."""

from __future__ import annotations

import unittest

import numpy as np

from tests import helpers as h  # noqa: F401  (puts src on sys.path)

import numguard


def failed(rep):
    return set(rep.failed_checks())


def _ftz(x):
    x = np.asarray(x)
    tiny = np.finfo(x.dtype).tiny
    return np.where(np.abs(x) < tiny, np.copysign(np.zeros_like(x), x), x)


class FlushToZeroTests(unittest.TestCase):
    def test_backend_that_flushes_subnormals_is_not_a_reference_failure(self):
        def double_ftz(x):  # like XLA on CPU: subnormal inputs and results become zero
            return _ftz(_ftz(x) * np.asarray(2, dtype=np.asarray(x).dtype))

        rep = numguard.check(double_ftz, reference=lambda x: x * 2, domain="any", elementwise=True,
                             dtypes=["float32", "float64"])
        self.assertNotIn("reference", failed(rep), rep.summary())
        skipped = [r for r in rep.results if r.check == "reference" and r.status == "skip"]
        self.assertTrue(skipped and "flushed to zero" in skipped[0].message)

    def test_real_errors_near_subnormals_still_fail(self):
        def wrong(x):
            return _ftz(x) * 2 + np.asarray(1e-3, dtype=np.asarray(x).dtype)

        rep = numguard.check(wrong, reference=lambda x: x * 2, domain="any", elementwise=True,
                             dtypes=["float32"])
        self.assertIn("reference", failed(rep))


class NanPolicyTests(unittest.TestCase):
    def test_ignoring_nan_like_the_reference_is_fine(self):
        rep = numguard.check(lambda x: np.nanmax(x), reference=np.nanmax, domain="any", dtypes=["float64"])
        self.assertNotIn("nan", failed(rep), rep.summary())

    def test_swallowing_nan_the_reference_propagates_fails(self):
        rep = numguard.check(lambda x: np.nanmax(x), reference=np.max, domain="any", dtypes=["float64"])
        self.assertIn("nan", failed(rep))

    def test_without_reference_nan_must_propagate(self):
        rep = numguard.check(lambda x: np.nan_to_num(x), domain="any", dtypes=["float64"], elementwise=True)
        self.assertIn("nan", failed(rep))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
