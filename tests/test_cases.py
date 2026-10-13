"""Tests for edge-case generation (numguard.cases)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - sets sys.path

from numguard.cases import (
    KINDS,
    Array,
    ArraySpec,
    Domain,
    case_matches,
    generate_cases,
    integers,
    interval,
)


def by_name(cases, dtype=None):
    return {c.name: c for c in cases if dtype is None or c.dtype == dtype}


class DomainTests(unittest.TestCase):
    def test_named_domains(self):
        pos = Domain.parse("positive")
        self.assertFalse(pos.contains(0.0))
        self.assertTrue(pos.contains(1e-300))
        self.assertFalse(pos.contains(-1.0))
        self.assertTrue(pos.contains(math.inf))
        self.assertTrue(Domain.parse("nonneg").contains(0.0))
        self.assertTrue(Domain.parse("nonneg").contains(-0.0))
        unit = Domain.parse("unit_interval")
        self.assertTrue(unit.contains(1.0) and unit.contains(0.0))
        self.assertFalse(unit.contains(1.0000001))
        self.assertFalse(Domain.parse("any").contains(math.nan))

    def test_interval_notation(self):
        d = Domain.parse("(-1, inf)")
        self.assertFalse(d.contains(-1.0))
        self.assertTrue(d.contains(-0.999))
        self.assertTrue(d.contains(math.inf))
        self.assertFalse(d.contains(-math.inf))
        d2 = Domain.parse("[0, 2)")
        self.assertTrue(d2.contains(0.0))
        self.assertFalse(d2.contains(2.0))
        self.assertEqual(d2.describe(), "[0, 2)")

    def test_integer_domain(self):
        d = Domain.parse("int[0, 255]")
        self.assertTrue(d.integer)
        self.assertTrue(d.contains(255))
        self.assertFalse(d.contains(2.5))
        self.assertFalse(d.contains(256))
        self.assertEqual(integers(0, 255), d)

    def test_invalid_domains(self):
        with self.assertRaises(ValueError):
            Domain.parse("sometimes")
        with self.assertRaises(ValueError):
            interval(2, 1)
        with self.assertRaises(ValueError):
            interval(0, 1, closed="sideways")


class GenerateCasesTests(unittest.TestCase):
    def test_all_kinds_for_unconstrained_input(self):
        cases = generate_cases([Array("n")], ["float64"])
        names = set(by_name(cases))
        expected = set(KINDS) - {"boundary", "int_min", "int_max"}
        self.assertEqual(names, expected)

    def test_dtypes_are_respected(self):
        for dt in ("float16", "float32", "float64"):
            cases = generate_cases([Array("n")], [dt])
            self.assertTrue(cases)
            for c in cases:
                self.assertEqual(c.dtype, dt)
                self.assertEqual(c.args[0].dtype, np.dtype(dt), c.name)
                self.assertIn(dt, c.tags)

    def test_positive_domain_never_gets_nonpositive_values(self):
        cases = generate_cases([Array("n", domain="positive")], ["float16", "float32", "float64"])
        names = {c.name for c in cases}
        self.assertNotIn("zeros", names)
        self.assertNotIn("neg_zero", names)
        self.assertNotIn("alternating", names)
        self.assertNotIn("neginf", names)
        for c in cases:
            x = c.args[0]
            finite = x[~np.isnan(x)]
            self.assertTrue(np.all(finite > 0), f"{c.name}/{c.dtype}: {finite}")

    def test_unit_interval_and_boundaries(self):
        cases = by_name(generate_cases([Array("n", domain="unit_interval")], ["float32"]))
        for c in cases.values():
            x = c.args[0][~np.isnan(c.args[0])]
            self.assertTrue(np.all((x >= 0) & (x <= 1)), c.name)
        self.assertIn("boundary", cases)
        self.assertEqual(set(cases["boundary"].args[0].tolist()), {0.0, 1.0})
        self.assertNotIn("huge", cases)
        self.assertNotIn("posinf", cases)

    def test_open_interval_boundary_uses_nextafter(self):
        cases = by_name(generate_cases([Array("n", domain="(-1, inf)")], ["float64"]))
        b = cases["boundary"].args[0]
        self.assertTrue(np.all(b > -1))
        self.assertEqual(b[0], np.nextafter(-1.0, 0.0))
        # huge values must be flipped to the positive side to stay in-domain
        self.assertTrue(np.all(cases["huge"].args[0] > 0))

    def test_extreme_magnitudes_per_dtype(self):
        for dt in ("float16", "float32", "float64"):
            fi = np.finfo(dt)
            cases = by_name(generate_cases([Array("n")], [dt]))
            huge = np.abs(cases["huge"].args[0].astype(np.float64))
            self.assertTrue(np.all(huge >= float(fi.max) / 4.01), dt)
            self.assertTrue(np.all(np.isfinite(huge)))
            sub = np.abs(cases["subnormal"].args[0].astype(np.float64))
            self.assertTrue(np.all((sub > 0) & (sub < float(fi.tiny))), dt)
            tiny = np.abs(cases["tiny"].args[0].astype(np.float64))
            self.assertTrue(np.all(tiny >= float(fi.tiny)), dt)

    def test_special_values(self):
        cases = by_name(generate_cases([Array("n")], ["float32"]))
        self.assertTrue(np.isnan(cases["nan"].args[0]).any())
        self.assertIn("nonfinite_input", cases["nan"].tags)
        self.assertIn("nan_input", cases["nan"].tags)
        self.assertTrue(np.isposinf(cases["posinf"].args[0]).any())
        self.assertTrue(np.isneginf(cases["neginf"].args[0]).any())
        nz = cases["neg_zero"].args[0]
        self.assertTrue(np.all(nz == 0) and np.all(np.signbit(nz)))
        self.assertEqual(cases["empty"].args[0].shape, (0,))
        self.assertEqual(cases["single"].args[0].shape, (1,))
        self.assertTrue(cases["random"].finite_inputs)
        self.assertTrue(cases["random"].regular)

    def test_cancellation_case_has_large_offset_small_spread(self):
        c = by_name(generate_cases([Array("n")], ["float64"]))["cancellation"]
        x = c.args[0]
        self.assertGreater(x.min(), 1e8)
        self.assertEqual(np.ptp(x), 4.0)
        self.assertIn("precision", c.tags)

    def test_large_n_configurable(self):
        c = by_name(generate_cases([Array("n")], ["float32"], large_n=1234))["large_n"]
        self.assertEqual(c.args[0].shape, (1234,))
        self.assertTrue(np.all(c.args[0] >= 1))

    def test_deterministic(self):
        a = generate_cases([Array("n"), Array("n")], ["float32"], seed=7)
        b = generate_cases([Array("n"), Array("n")], ["float32"], seed=7)
        c = generate_cases([Array("n"), Array("n")], ["float32"], seed=8)
        self.assertEqual([x.name for x in a], [x.name for x in b])
        for x, y in zip(a, b):
            for p, q in zip(x.args, y.args):
                np.testing.assert_array_equal(p, q)
        ra = by_name(a)["random"].args[0]
        rc = by_name(c)["random"].args[0]
        self.assertFalse(np.array_equal(ra, rc))

    def test_multi_argument_variants(self):
        cases = by_name(generate_cases([Array("n", name="a"), Array("n", name="b")], ["float64"]))
        self.assertIn("zeros", cases)
        self.assertIn("zeros[a]", cases)
        self.assertIn("zeros[b]", cases)
        za = cases["zeros[a]"]
        self.assertTrue(np.all(za.args[0] == 0))
        self.assertFalse(np.all(za.args[1] == 0))
        # symbolic sizes are shared between arguments
        self.assertEqual(cases["pair"].args[0].shape, cases["pair"].args[1].shape)

    def test_mixed_domains_skip_joint_case_when_one_argument_cannot_take_it(self):
        cases = by_name(generate_cases(
            [Array("n", domain="positive", name="p"), Array("n", name="q")], ["float64"]))
        self.assertNotIn("zeros", cases)
        self.assertIn("zeros[q]", cases)
        self.assertNotIn("zeros[p]", cases)

    def test_fixed_shapes_and_integer_inputs(self):
        cases = generate_cases([ArraySpec(shape=(2, 3)), Array("n", domain="int[0, 255]")],
                               ["float32"])
        names = {c.name for c in cases}
        self.assertNotIn("int_max", names)  # not joint: arg0 is float
        self.assertIn("int_max[arg1]", names)
        for c in cases:
            self.assertEqual(c.args[0].shape, (2, 3))
            self.assertEqual(c.args[1].dtype, np.int64)
            self.assertTrue(np.all((c.args[1] >= 0) & (c.args[1] <= 255)), c.name)

    def test_fixed_shape_only_has_no_size_cases(self):
        names = {c.name for c in generate_cases([ArraySpec(shape=(4,))], ["float64"])}
        self.assertFalse(names & {"empty", "single", "pair", "large_n"})

    def test_include_exclude_and_matching(self):
        cases = generate_cases([Array("n")], ["float64"], include=["nonfinite_input"])
        self.assertEqual({c.name for c in cases}, {"nan", "posinf", "neginf"})
        cases = generate_cases([Array("n")], ["float64"], exclude=["*inf", "extreme"])
        names = {c.name for c in cases}
        self.assertFalse(names & {"posinf", "neginf", "huge", "subnormal"})
        c = generate_cases([Array("n")], ["float64"], include=["zeros"])[0]
        self.assertTrue(case_matches(c, ["zer*"]))
        self.assertTrue(case_matches(c, ["regular"]))
        self.assertFalse(case_matches(c, ["nan"]))


if __name__ == "__main__":
    unittest.main()
