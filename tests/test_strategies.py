"""numguard.strategies: Hypothesis strategies from numguard specs.

The composition logic is exercised with a small fake Hypothesis (always), and
with the real Hypothesis when it is installed.
"""

from __future__ import annotations

import importlib
import unittest

import numpy as np

from tests import helpers  # noqa: F401
from tests.fake_backends import installed
from tests.fake_hypothesis import make_modules

import numguard
from numguard import strategies as ngst
from numguard.cases import Domain

try:
    import hypothesis  # noqa: F401

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


def positive_log(x):
    return np.log(x)


@numguard.guard(domain="[-1, 1]", dtypes=["float16"])
def guarded(x):
    return x


class ElementKwargsTests(unittest.TestCase):
    def test_open_unit_interval_float16(self):
        kw = ngst.element_kwargs(Domain.parse("(0, 1]"), "float16")
        self.assertEqual(kw["min_value"], 0.0)
        self.assertTrue(kw["exclude_min"])
        self.assertEqual(kw["max_value"], 1.0)
        self.assertFalse(kw["allow_nan"])
        self.assertEqual(kw["width"], 16)

    def test_bounds_are_rounded_inwards(self):
        kw = ngst.element_kwargs(Domain.parse("[0.1, 0.3]"), "float16")
        self.assertGreaterEqual(kw["min_value"], 0.1)
        self.assertLessEqual(kw["max_value"], 0.3)
        self.assertEqual(float(np.float16(kw["min_value"])), kw["min_value"])
        kw = ngst.element_kwargs(Domain.parse("[0.1, 0.3]"), "bfloat16")
        self.assertGreaterEqual(kw["min_value"], 0.1)

    def test_unbounded_and_integer(self):
        kw = ngst.element_kwargs(Domain.parse("any"), "float32")
        self.assertEqual(kw["max_value"], float(np.finfo(np.float32).max))
        kw = ngst.element_kwargs(Domain.parse("int[0, 255]"), "int64")
        self.assertEqual((kw["kind"], kw["min_value"], kw["max_value"]), ("int", 0, 255))


class FakeHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _with_fake(self):
        mods = make_modules()
        ctx = installed(**mods)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        importlib.reload(ngst)
        self.addCleanup(importlib.reload, ngst)

    def test_inputs_for_respects_domain_and_dtypes(self):
        if HAS_HYPOTHESIS:  # pragma: no cover - the real one is tested below
            self.skipTest("real Hypothesis installed")
        self._with_fake()
        strat = ngst.inputs_for(positive_log, domain="positive", dtypes=["float32"], max_size=6)
        seen_kinds = set()
        for _ in range(300):
            (x,) = strat.draw(self.rng)
            self.assertEqual(x.dtype, np.float32)
            finite = x[np.isfinite(x)]
            self.assertTrue(np.all(finite > 0), x)
            seen_kinds.add("empty" if x.size == 0 else ("nonfinite" if finite.size < x.size else "finite"))
        self.assertEqual(seen_kinds, {"empty", "nonfinite", "finite"})

    def test_cases_from_guarded_function_are_fresh_copies(self):
        if HAS_HYPOTHESIS:  # pragma: no cover
            self.skipTest("real Hypothesis installed")
        self._with_fake()
        strat = ngst.cases(guarded)
        c1 = strat.draw(np.random.default_rng(1))
        self.assertIsInstance(c1, numguard.Case)
        self.assertEqual(c1.dtype, "float16")
        catalogue = {c.name: c for c in guarded.__numguard__.cases()}
        if c1.args[0].size:
            c1.args[0][...] = 99
            self.assertFalse(np.any(catalogue[c1.name].args[0] == 99))

    def test_random_inputs_multi_argument_shares_symbolic_size(self):
        if HAS_HYPOTHESIS:  # pragma: no cover
            self.skipTest("real Hypothesis installed")
        self._with_fake()
        strat = ngst.random_inputs(inputs=[numguard.Array("n"), numguard.Array("n", 3, domain="nonneg")],
                                   dtypes=["float64", "bfloat16"], max_size=5)
        for _ in range(50):
            a, b = strat.draw(self.rng)
            self.assertEqual(b.shape, (a.shape[0], 3))
            self.assertTrue(np.all(b >= 0))
            if a.dtype == np.float32:  # bfloat16 storage
                np.testing.assert_array_equal(numguard.dtypes.round_bf16(a), a)

    def test_missing_hypothesis_gives_actionable_error(self):
        if HAS_HYPOTHESIS:  # pragma: no cover
            self.skipTest("real Hypothesis installed")
        with self.assertRaisesRegex(ImportError, "numguard-ci\\[hypothesis\\]"):
            ngst.inputs_for(positive_log)


@unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
class RealHypothesisTests(unittest.TestCase):  # pragma: no cover - needs hypothesis
    def test_given_with_inputs_for(self):
        from hypothesis import given, settings

        seen = []

        @settings(max_examples=60, deadline=None)
        @given(ngst.inputs_for(positive_log, domain="positive", dtypes=["float32"]))
        def prop(args):
            (x,) = args
            seen.append(x)
            finite = x[np.isfinite(x)]
            assert np.all(finite > 0)

        prop()
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()
