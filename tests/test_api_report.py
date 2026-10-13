"""Tests for the public API (check, guard, Spec) and Report formatting."""

from __future__ import annotations

import json
import unittest

import numpy as np
import scipy.special as sp

from tests import helpers as h

import numguard
from numguard.api import make_spec
from numguard.report import FAIL, PASS, SKIP, CheckResult, Report


class CheckApiTests(unittest.TestCase):
    def test_inputs_inferred_from_signature(self):
        def two_args(a, b, scale=1.0):
            return a * b * scale

        spec = make_spec(two_args)
        self.assertEqual([s.name for s in spec.inputs], ["a", "b"])
        self.assertEqual(spec.arg_names, ("a", "b"))

    def test_input_forms(self):
        self.assertEqual(make_spec(np.sum, inputs="positive").inputs[0].domain.name, "positive")
        self.assertEqual(make_spec(np.sum, inputs=(3, "n")).inputs[0].shape, (3, "n"))
        spec = make_spec(np.add, inputs={"x": "nonneg", "y": numguard.Array("n")})
        self.assertEqual(spec.arg_names, ("x", "y"))
        spec = make_spec(np.add, inputs=["positive", "any"])
        self.assertEqual(len(spec.inputs), 2)
        self.assertEqual(make_spec(np.sum, domain="unit_interval").inputs[0].domain.name,
                         "unit_interval")

    def test_kwargs_are_forwarded(self):
        rep = numguard.check(np.sum, kwargs={"axis": -1}, include=["random"])
        self.assertTrue(rep.ok)
        with self.assertRaises(TypeError):
            numguard.check(np.sum, not_an_option=True)
        with self.assertRaises(ValueError):
            numguard.check(np.sum, nan="sometimes")

    def test_dtypes_argument(self):
        rep = numguard.check(np.sum, dtypes="float16", include=["random"])
        self.assertEqual(rep.dtypes, ["float16"])
        rep = numguard.check(np.sum, dtypes=["float16", "float64"], include=["random"])
        self.assertEqual(rep.dtypes, ["float16", "float64"])


class GuardTests(unittest.TestCase):
    def test_bare_decorator_attaches_default_spec(self):
        @numguard.guard
        def double(x):
            return x * 2

        spec = numguard.get_spec(double)
        self.assertIsInstance(spec, numguard.Spec)
        self.assertEqual(double(np.array([1.0])).tolist(), [2.0])  # behaviour unchanged
        self.assertTrue(double.numguard_check(include=["random"]).ok)

    def test_decorator_with_arguments(self):
        @numguard.guard(reference=sp.logsumexp, properties=["translation_equivariant"],
                        dtypes=["float64"])
        def lse(x):
            return np.log(np.sum(np.exp(x)))

        spec = numguard.get_spec(lse)
        self.assertIs(spec.reference, sp.logsumexp)
        self.assertEqual(spec.dtypes, ("float64",))
        self.assertEqual(len(spec.properties), 1)
        report = lse.numguard_check()
        self.assertFalse(report.ok)
        self.assertEqual(report.dtypes, ["float64"])
        # overrides
        self.assertTrue(lse.numguard_check(include=["random", "zeros"]).ok)

    def test_check_with_attached_spec_and_overrides(self):
        @numguard.guard(domain="positive", dtypes=["float32"])
        def f(x):
            return np.log(x)

        rep = numguard.check(f, spec=numguard.get_spec(f), dtypes=["float64"])
        self.assertEqual(rep.dtypes, ["float64"])
        self.assertIsNone(numguard.get_spec(np.log))

    def test_raise_if_failed(self):
        rep = numguard.check(h.naive_logsumexp, reference=sp.logsumexp)
        with self.assertRaises(numguard.NumGuardError) as cm:
            rep.raise_if_failed()
        self.assertIsInstance(cm.exception, AssertionError)
        self.assertIn("naive_logsumexp FAILED", str(cm.exception))
        self.assertIs(cm.exception.report, rep)
        ok = numguard.check(h.stable_logsumexp, include=["random"])
        self.assertIs(ok.raise_if_failed(), ok)


class ReportTests(unittest.TestCase):
    def make_report(self):
        rep = Report(function="f", dtypes=["float32", "float64"], n_cases=3)
        rep.results = [
            CheckResult("finite", "huge", "float32", FAIL, "finite input gave inf",
                        fp_events=["overflow"]),
            CheckResult("finite", "huge", "float64", FAIL, "finite input gave inf"),
            CheckResult("reference", "huge", "float32", FAIL, "got inf", max_abs_err=float("inf"),
                        max_rel_err=float("inf"), max_ulp=1e9),
            CheckResult("reference", "zeros", "float32", PASS, max_abs_err=0.0,
                        max_rel_err=0.0, max_ulp=0.0),
            CheckResult("nan", "nan", "float32", SKIP, "ignored"),
        ]
        return rep

    def test_counts_and_queries(self):
        rep = self.make_report()
        self.assertFalse(rep.ok)
        self.assertFalse(bool(rep))
        self.assertEqual(len(rep.failures), 3)
        self.assertEqual(rep.failed_checks(), ["finite", "reference"])
        self.assertEqual(rep.failed_cases(), ["huge"])
        self.assertEqual(rep.counts()["reference"], {"pass": 1, "fail": 1, "skip": 0})

    def test_summary_table(self):
        text = self.make_report().summary()
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("numguard: f FAILED - 1 passed, 3 failed"))
        self.assertIn("| check", text)
        # failures are grouped by (check, case) with their dtypes
        self.assertIn("float32,float64", text)
        self.assertIn("[fp: overflow]", text)
        self.assertIn("1e+09", text)
        widths = {len(line) for line in lines[1:]}
        self.assertEqual(len(widths), 1, "table rows must be aligned")

    def test_summary_truncation(self):
        rep = Report(function="g")
        rep.results = [CheckResult("finite", f"case{i}", "float64", FAIL, "x" * 300)
                       for i in range(30)]
        text = rep.summary(max_rows=5, width=40)
        self.assertIn("... and 25 more failing", text)
        self.assertNotIn("x" * 41, text)

    def test_passing_summary(self):
        rep = Report(function="ok", dtypes=["float64"], n_cases=1,
                     results=[CheckResult("finite", "random", "float64", PASS)])
        self.assertEqual(rep.summary(), "numguard: ok PASSED - 1 passed, 0 failed "
                                        "(1 cases; dtypes: float64)")

    def test_to_json(self):
        rep = self.make_report()
        data = json.loads(rep.to_json())
        self.assertFalse(data["ok"])
        self.assertEqual(data["function"], "f")
        self.assertEqual(len(data["results"]), 5)
        self.assertEqual(data["results"][2]["max_rel_err"], "inf")
        only = json.loads(rep.to_json(failures_only=True))
        self.assertEqual(len(only["results"]), 3)

    def test_real_report_json_roundtrip(self):
        rep = numguard.check(h.naive_logsumexp, reference=sp.logsumexp)
        data = json.loads(rep.to_json(indent=None))
        self.assertEqual(data["n_cases"], rep.n_cases)
        self.assertIn("finite", data["counts"])


if __name__ == "__main__":
    unittest.main()
