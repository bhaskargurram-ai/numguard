"""Each check catches a known-bad function and passes a known-good one."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.special as sp

from tests import helpers as h

import numguard
from numguard.checks import call, compare


def failed_checks(report):
    return set(report.failed_checks())


class CompareTests(unittest.TestCase):
    def test_within_tolerance(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        c = compare(a, a.astype(np.float64) * (1 + 1e-7))
        self.assertTrue(c.ok)
        self.assertLessEqual(c.max_ulp, 2)

    def test_outside_tolerance_reports_errors(self):
        c = compare(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
        self.assertFalse(c.ok)
        self.assertEqual(c.n_bad, 1)
        self.assertAlmostEqual(c.max_rel_err, 0.1 / 2.1, places=6)
        self.assertIn("got 2", c.message)

    def test_nonfinite_matching(self):
        self.assertTrue(compare(np.array([np.nan, np.inf]), np.array([np.nan, np.inf])).ok)
        self.assertFalse(compare(np.array([np.inf]), np.array([-np.inf])).ok)
        self.assertFalse(compare(np.array([np.nan]), np.array([1.0])).ok)
        # a float64 reference beyond float32 range is expected to be inf in float32
        self.assertTrue(compare(np.array([np.inf], dtype=np.float32), np.array([1e39])).ok)

    def test_shape_mismatch_and_integers(self):
        self.assertFalse(compare(np.zeros(3), np.zeros(4)).ok)
        self.assertFalse(compare(np.array([1, 2], dtype=np.uint8), np.array([1, 3])).ok)
        self.assertTrue(compare(np.array([1, 2], dtype=np.uint8), np.array([1, 3]), atol=1).ok)

    def test_dict_tolerances(self):
        a = np.array([1.0], dtype=np.float32)
        e = np.array([1.001])
        self.assertFalse(compare(a, e).ok)
        self.assertTrue(compare(a, e, rtol={"float32": 1e-2}).ok)

    def test_subnormal_errors_measured_against_tiny(self):
        tiny = float(np.finfo(np.float64).tiny)
        self.assertTrue(compare(np.array([0.0]), np.array([tiny * 1e-14])).ok)
        self.assertFalse(compare(np.array([0.0]), np.array([tiny])).ok)


class CallTests(unittest.TestCase):
    def test_records_fp_events_and_errors(self):
        r = call(np.exp, [np.array([1000.0])])
        self.assertIn("overflow", r.fp_events)
        r = call(lambda x: 1 / 0, [np.array([1.0])])
        self.assertIsInstance(r.error, ZeroDivisionError)
        self.assertIn("ZeroDivisionError", r.error_text())

    def test_detects_mutation_and_normalizes_outputs(self):
        x = np.array([-1.0, 2.0])
        r = call(h.mutating_abs, [x])
        self.assertEqual(r.mutated, [0])
        np.testing.assert_array_equal(x, [-1.0, 2.0])  # original untouched
        r = call(lambda x: (x, 1.0), [x])
        self.assertEqual(len(r.outputs), 2)
        self.assertEqual(r.weak, [False, True])


class FiniteAndReferenceTests(unittest.TestCase):
    def test_naive_logsumexp_fails_finite_and_reference(self):
        rep = numguard.check(h.naive_logsumexp, reference=sp.logsumexp)
        self.assertFalse(rep.ok)
        self.assertTrue({"finite", "reference"} <= failed_checks(rep))
        self.assertIn("huge", rep.failed_cases())
        overflow = [r for r in rep.failures if "overflow" in r.fp_events]
        self.assertTrue(overflow, "overflow warnings should be recorded as evidence")

    def test_stable_logsumexp_passes(self):
        rep = numguard.check(h.stable_logsumexp, reference=sp.logsumexp,
                             properties=["translation_equivariant"])
        self.assertTrue(rep.ok, rep.summary())

    def test_finite_check_without_reference(self):
        rep = numguard.check(h.naive_logsumexp)
        self.assertEqual(failed_checks(rep), {"finite"})

    def test_allow_nonfinite(self):
        rep = numguard.check(np.exp, allow_nonfinite=["overflow_prone", "precision"])
        self.assertTrue(rep.ok, rep.summary())

    def test_reference_agreeing_on_inf_is_not_a_finite_failure(self):
        rep = numguard.check(np.exp, reference=np.exp)
        self.assertNotIn("finite", failed_checks(rep))

    def test_reference_ulp_reported(self):
        def sloppy_exp(x):
            return (np.exp(x) * (1 + 1e-9)).astype(x.dtype)

        rep = numguard.check(sloppy_exp, reference=np.exp, dtypes=["float64"],
                             include=["random"])
        self.assertFalse(rep.ok)
        r = rep.failures[0]
        self.assertEqual(r.check, "reference")
        self.assertGreater(r.max_ulp, 1e5)
        self.assertAlmostEqual(r.max_rel_err, 1e-9, delta=1e-10)
        rep = numguard.check(sloppy_exp, reference=np.exp, dtypes=["float32"],
                             include=["random"])
        self.assertTrue(rep.ok, "1e-9 relative error is below float32 resolution")

    def test_highprec_reference(self):
        def naive_log1p(x):
            return np.log(1 + x)

        ref = numguard.highprec(numpy_fn=np.log1p)
        rep = numguard.check(naive_log1p, reference=ref, domain="(-1, inf)")
        self.assertIn("reference", failed_checks(rep))
        self.assertIn("small", rep.failed_cases())
        rep = numguard.check(np.log1p, reference=ref, domain="(-1, inf)")
        self.assertTrue(rep.ok, rep.summary())

    def test_exceptions(self):
        def strict_mean(x):
            if x.size == 0:
                raise ValueError("empty")
            return np.mean(x)

        # (np.mean itself overflows on near-max inputs: exclude those here)
        rep = numguard.check(strict_mean, exclude=["overflow_prone"])
        self.assertEqual(failed_checks(rep), {"error"})
        self.assertEqual(rep.failed_cases(), ["empty"])
        self.assertTrue(numguard.check(strict_mean, may_raise=["empty"],
                                       exclude=["overflow_prone"]).ok)


class DtypeTests(unittest.TestCase):
    def test_upcasting_is_caught(self):
        rep = numguard.check(h.upcasting_square, dtypes=["float32"], exclude=["extreme"])
        self.assertEqual(failed_checks(rep), {"dtype"})
        self.assertIn("got float64, expected float32", rep.failures[0].message)

    def test_preserving_passes_and_declared_dtype(self):
        self.assertTrue(numguard.check(h.float_square, dtypes=["float32"],
                                       exclude=["extreme"]).ok)
        rep = numguard.check(h.upcasting_square, dtypes=["float32"], exclude=["extreme"],
                             out_dtype="float64")
        self.assertTrue(rep.ok, rep.summary())


class NanTests(unittest.TestCase):
    def test_nan_swallowing_is_caught(self):
        rep = numguard.check(h.nan_swallowing_sum, exclude=["extreme", "*inf"])
        self.assertEqual(failed_checks(rep), {"nan"})

    def test_nan_propagation_passes(self):
        self.assertTrue(numguard.check(np.sum, exclude=["extreme", "*inf"]).ok)

    def test_nan_raise_policy(self):
        def raising(x):
            if np.isnan(x).any():
                raise ValueError("NaN input")
            return np.sum(x)

        rep = numguard.check(raising, nan="raise", exclude=["extreme"])
        self.assertTrue(rep.ok, rep.summary())
        rep = numguard.check(np.sum, nan="raise", exclude=["extreme"])
        self.assertEqual(failed_checks(rep), {"nan"})

    def test_nan_ignore_policy(self):
        self.assertTrue(numguard.check(h.nan_swallowing_sum, nan="ignore",
                                       exclude=["extreme", "*inf"]).ok)


class MutationTests(unittest.TestCase):
    def test_in_place_modification_is_caught(self):
        rep = numguard.check(h.mutating_abs, include=["random"])
        self.assertEqual(failed_checks(rep), {"mutation"})
        self.assertTrue(numguard.check(h.mutating_abs, include=["random"],
                                       check_mutation=False).ok)


class BatchTests(unittest.TestCase):
    def test_global_softmax_fails_batch(self):
        rep = numguard.check(h.global_softmax, batched=True,
                             exclude=["extreme", "nonfinite_input", "empty"])
        self.assertEqual(failed_checks(rep), {"batch"})
        self.assertIn("f(stack(xs)) != stack(f(x))", rep.failures[0].message)

    def test_rowwise_softmax_passes_batch(self):
        rep = numguard.check(h.rowwise_softmax, batched=True,
                             exclude=["extreme", "nonfinite_input", "empty"])
        self.assertTrue(rep.ok, rep.summary())
        self.assertGreater(rep.counts()["batch"]["pass"], 0)

    def test_batch_axis(self):
        def colwise(x):
            return x - np.mean(x, axis=0)

        rep = numguard.check(colwise, batched=True, batch_axis=-1, include=["random", "pair"])
        self.assertTrue(rep.ok, rep.summary())
        rep = numguard.check(colwise, batched=True, batch_axis=0, include=["random"])
        self.assertIn("batch", failed_checks(rep))


class PropertyTests(unittest.TestCase):
    def test_onepass_variance_fails_translation_invariance(self):
        rep = numguard.check(h.onepass_variance, properties=["translation_invariant:1e4"],
                             dtypes=["float32"], include=["regular"])
        self.assertIn("prop:translation_invariant(c=10000)", failed_checks(rep))

    def test_twopass_variance_passes_invariances(self):
        rep = numguard.check(h.twopass_variance,
                             properties=["translation_invariant", "scale_equivariant:2",
                                         "permutation_invariant"],
                             include=["regular"])
        self.assertTrue(rep.ok, rep.summary())

    def test_permutation_invariance_catches_order_dependence(self):
        rep = numguard.check(lambda x: x[0] if x.size else x.sum(),
                             properties=["permutation_invariant"], include=["random"])
        self.assertEqual(failed_checks(rep), {"prop:permutation_invariant"})

    def test_symmetry(self):
        both = {"a": "any", "b": "any"}
        self.assertTrue(numguard.check(np.add, inputs=both, properties=["symmetric"],
                                       include=["regular"]).ok)
        rep = numguard.check(np.subtract, inputs=both, properties=["symmetric"],
                             include=["random"])
        self.assertEqual(failed_checks(rep), {"prop:symmetric"})

    def test_odd_even_scale(self):
        self.assertTrue(numguard.check(np.sin, properties=["odd"], include=["regular"]).ok)
        self.assertTrue(numguard.check(np.cos, properties=["even"], include=["regular"]).ok)
        self.assertFalse(numguard.check(np.cos, properties=["odd"], include=["random"]).ok)
        norm = numguard.check(np.linalg.norm, properties=["scale_equivariant:1"],
                              include=["regular"])
        self.assertTrue(norm.ok, norm.summary())

    def test_properties_respect_domain(self):
        # translating by +1 would push unit_interval values out of the domain:
        # such transformed cases are skipped instead of reported
        rep = numguard.check(np.sqrt, domain="unit_interval",
                             properties=["translation_invariant"], include=["regular"])
        props = rep.counts().get("prop:translation_invariant(c=1)", {"pass": 0, "fail": 0})
        # only the all-zeros case fits (once per dtype)
        self.assertEqual(props["fail"] + props["pass"], len(rep.dtypes))

    def test_unknown_property(self):
        with self.assertRaises(ValueError):
            numguard.check(np.sum, properties=["commutative-ish"])

    def test_custom_property_object(self):
        from numguard.properties import Property

        idempotent = Property(
            "idempotent",
            transform=lambda args, specs: (np.sort(args[0]),),
            expect=lambda outs, args: outs,
            description="f(sort(x)) == f(x)",
        )
        self.assertTrue(numguard.check(np.sort, properties=[idempotent],
                                       include=["regular"]).ok)


class MonotonicBoundsAndSignalsTests(unittest.TestCase):
    def test_monotonic(self):
        self.assertTrue(numguard.check(np.tanh, monotonic="increasing",
                                       exclude=["nonfinite_input"]).ok)
        rep = numguard.check(h.bumpy, monotonic="increasing", include=["random"])
        self.assertEqual(failed_checks(rep), {"monotonic"})
        self.assertTrue(numguard.check(lambda x: -x, monotonic="decreasing",
                                       include=["random"]).ok)

    def test_bounds(self):
        rep = numguard.check(lambda x: x * 2, bounds=(-1, 1), include=["random"])
        self.assertEqual(failed_checks(rep), {"bounds"})
        self.assertTrue(numguard.check(np.tanh, bounds=(-1, 1)).ok)

    def test_fp_signals(self):
        def naive_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        rep = numguard.check(naive_sigmoid, fp_signals="fail")
        self.assertEqual(failed_checks(rep), {"fp_signals"})
        self.assertTrue(numguard.check(sp.expit, fp_signals="fail").ok)

    def test_postconditions_and_ignore(self):
        def nonneg(args, outs):
            return None if np.all(outs[0][~np.isnan(outs[0])] >= 0) else "negative output"

        rep = numguard.check(np.sin, postconditions=[nonneg], include=["random"])
        self.assertEqual(failed_checks(rep), {"post:nonneg"})
        rep = numguard.check(np.sin, postconditions=nonneg, include=["random"],
                             ignore=["post:*"])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.counts()["post:nonneg"]["skip"], len(rep.dtypes))

    def test_extra_cases(self):
        rep = numguard.check(h.onepass_variance, reference=np.var, dtypes=["float64"],
                             include=["extra"],
                             extra_cases={"offset_1e9": [1e9 + 1, 1e9 + 2, 1e9 + 3]})
        self.assertEqual(rep.n_cases, 1)
        self.assertFalse(rep.ok)


if __name__ == "__main__":
    unittest.main()
