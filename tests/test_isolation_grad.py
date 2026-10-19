"""The isolation check and the gradient checks."""

from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np
import scipy.special as sp

from tests import helpers as h  # noqa: F401  (puts src on sys.path)

import numguard
from numguard.gradcheck import finite_difference_grad


def failed(report):
    return set(report.failed_checks())


def shared_convergence_exp(x):
    """exp by its Taylor series with one stopping test for the whole array.

    The loop stops when the first element has converged, so elements that
    need more terms come out wrong, and how wrong depends on their
    neighbours: the bug pattern of ``jax.scipy.special.betainc`` before
    jax-ml/jax#41440.
    """
    x = np.asarray(x)
    term = np.ones_like(x)
    total = np.ones_like(x)
    with np.errstate(all="ignore"):
        for k in range(1, 400):
            term = term * x / k
            total = total + term
            if np.any(np.abs(term) <= np.finfo(x.dtype).eps * np.abs(total)):
                break
    return total


def per_element_exp(x):
    """The same series, with a convergence test per element (the fix)."""
    x = np.asarray(x)
    term = np.ones_like(x)
    total = np.ones_like(x)
    done = np.zeros(x.shape, bool)
    with np.errstate(all="ignore"):
        for k in range(1, 400):
            term = np.where(done, 0, term * x / k)
            total = total + term
            done |= np.abs(term) <= np.finfo(x.dtype).eps * np.abs(total)
            if done.all():
                break
    return total


class IsolationTests(unittest.TestCase):
    def test_shared_convergence_fails_isolation(self):
        rep = numguard.check(shared_convergence_exp, inputs="[-5, 5]", reference=np.exp,
                             elementwise=True, dtypes=["float64"])
        self.assertIn("isolation", failed(rep))
        msg = next(r.message for r in rep.failures if r.check == "isolation")
        self.assertIn("depends on the other elements", msg)

    def test_per_element_convergence_passes(self):
        rep = numguard.check(per_element_exp, inputs="[-5, 5]", reference=np.exp,
                             elementwise=True, dtypes=["float64"])
        self.assertNotIn("isolation", failed(rep), rep.summary())
        self.assertGreater(rep.counts()["isolation"]["pass"], 0)

    def test_ufuncs_pass(self):
        for fn in (np.exp, np.tanh, sp.expit):
            rep = numguard.check(fn, reference=fn, elementwise=True, allow_nonfinite="*")
            self.assertNotIn("isolation", failed(rep), (fn, rep.summary()))

    def test_batch_normalization_fails(self):
        def standardize(x):
            return (x - np.mean(x)) / (np.std(x) + 1)

        rep = numguard.check(standardize, elementwise=True, include=["random"], dtypes=["float64"])
        self.assertIn("isolation", failed(rep))

    def test_reduction_declared_elementwise_fails(self):
        rep = numguard.check(lambda x: np.sum(x), elementwise=True, include=["random"],
                             dtypes=["float64"])
        self.assertIn("isolation", failed(rep))
        self.assertIn("output shape", rep.failures[0].message)

    def test_off_by_default(self):
        rep = numguard.check(shared_convergence_exp, inputs="[-5, 5]", dtypes=["float64"])
        self.assertNotIn("isolation", rep.counts())


class FiniteDifferenceTests(unittest.TestCase):
    def test_smooth_function(self):
        x = np.array([-3.0, -0.5, 0.25, 2.0, 30.0])
        est, unc = finite_difference_grad(np.sin, [x], 0, None, {}, elementwise=True)
        np.testing.assert_allclose(est, np.cos(x), rtol=1e-9)
        self.assertTrue(np.all(unc < 1e-8))

    def test_untrustworthy_points_are_nan(self):
        # far beyond the period of sin, a step cannot resolve the derivative
        x = np.array([8.319698978042759e307, -1.27517537e38, 6e19])
        est, _ = finite_difference_grad(np.cos, [x], 0, None, {}, elementwise=True)
        self.assertTrue(np.all(np.isnan(est)))
        # a kink: the one-sided slopes differ
        est, _ = finite_difference_grad(np.abs, [np.array([0.0])], 0, None, {}, elementwise=True)
        self.assertTrue(np.isnan(est[0]))

    def test_non_finite_values_and_step_functions_are_skipped(self):
        # scipy's gammaln returns inf at subnormal arguments; |gammaln| is symmetric about
        # the pole at 0, so differences across it agree on a meaningless -0.577
        def lgamma_inf_at_tiny(v):
            return np.where(np.abs(v) < 1e-300, np.inf, sp.gammaln(v))

        est, _ = finite_difference_grad(lgamma_inf_at_tiny, [np.array([-7.9e-323, 2.5])], 0, None, {},
                                        elementwise=True)
        self.assertTrue(np.isnan(est[0]))
        self.assertAlmostEqual(est[1], sp.digamma(2.5), places=8)
        # round is the identity above 2**52 and a staircase finer than the step at 1e9
        x = np.array([-6.46e307, 1.07374e9, 154591.0])
        est, _ = finite_difference_grad(np.round, [x], 0, None, {}, elementwise=True)
        self.assertTrue(np.all(np.isnan(est)), est)
        # an integer-valued point of a smooth function is still differentiated
        est, _ = finite_difference_grad(lambda v: v * 1.0, [np.array([5.0])], 0, None, {}, elementwise=True)
        self.assertAlmostEqual(est[0], 1.0, places=10)

    def test_domain_boundary_is_skipped(self):
        spec = numguard.api.make_spec(np.sqrt, domain="nonneg")
        x = np.array([0.0, 1e-300, 4.0])
        est, _ = finite_difference_grad(np.sqrt, [x], 0, spec.inputs[0], {}, elementwise=True)
        self.assertTrue(np.isnan(est[0]) and np.isnan(est[1]))
        self.assertAlmostEqual(est[2], 0.25, places=10)

    def test_reduction_uses_spread_scale(self):
        x = np.array([46341.0, 46342.0, 46343.0])
        est, _ = finite_difference_grad(lambda v: sp.logsumexp(v), [x], 0, None, {},
                                        elementwise=False)
        np.testing.assert_allclose(est, sp.softmax(x), rtol=1e-7)


class GradientCheckTests(unittest.TestCase):
    def test_correct_analytic_gradients_pass(self):
        cases = [(np.sin, np.cos, "any"), (np.log, lambda x: (1 / x).astype(x.dtype), "positive"),
                 (sp.gammaln, sp.digamma, "positive"), (np.arctan, lambda x: (1 / (1 + x * x)).astype(x.dtype), "any")]
        for f, g, dom in cases:
            rep = numguard.check(f, domain=dom, reference=f, grad=g, elementwise=True,
                                 allow_nonfinite="*", dtypes=["float32", "float64"])
            self.assertNotIn("grad", failed(rep), (f.__name__, rep.summary()))
            self.assertGreater(rep.counts()["grad"]["pass"], 0)

    def test_wrong_gradient_fails_with_location(self):
        def bad(x):
            return (np.cos(x) * np.where(np.abs(x) > 3, 1.01, 1.0)).astype(x.dtype)

        rep = numguard.check(np.sin, reference=np.sin, grad=bad, elementwise=True)
        self.assertIn("grad", failed(rep))
        msg = next(r.message for r in rep.failures if r.check == "grad")
        self.assertIn("d/dx at x=", msg)

    def test_overflowing_softmax_gradient_fails(self):
        # the gradient of logsumexp is softmax; computing it as exp(x) / sum(exp(x))
        # overflows where logsumexp itself does not
        def lse(x):
            return sp.logsumexp(x)

        def naive_softmax(x):
            with np.errstate(all="ignore"):
                return (np.exp(x) / np.sum(np.exp(x))).astype(x.dtype)

        rep = numguard.check(lse, inputs="any", reference=lse, grad=naive_softmax)
        self.assertIn("grad", failed(rep))
        rep = numguard.check(lse, inputs="any", reference=lse,
                             grad=lambda x: sp.softmax(x).astype(x.dtype))
        self.assertNotIn("grad", failed(rep), rep.summary())

    def test_grad_reference(self):
        rep = numguard.check(np.sin, reference=np.sin, grad=np.cos, grad_reference=np.cos,
                             elementwise=True)
        self.assertNotIn("grad", failed(rep), rep.summary())
        rep = numguard.check(np.sin, reference=np.sin, grad=lambda x: -np.cos(x),
                             grad_reference=np.cos, elementwise=True)
        self.assertIn("grad", failed(rep))
        msg = next(r.message for r in rep.failures if r.check == "grad")
        self.assertIn("grad_reference", msg)

    def test_two_arguments(self):
        def f(x, y):
            return x * np.exp(y)

        def g(x, y):
            return np.exp(y), x * np.exp(y)

        rep = numguard.check(f, inputs=["[-3, 3]", "[-3, 3]"], reference=f, grad=g,
                             elementwise=True, dtypes=["float64"])
        self.assertNotIn("grad", failed(rep), rep.summary())
        rep = numguard.check(f, inputs=["[-3, 3]", "[-3, 3]"], reference=f,
                             grad=lambda x, y: (np.exp(y), np.exp(y)), elementwise=True,
                             dtypes=["float64"])
        self.assertIn("grad", failed(rep))
        msg = next(r.message for r in rep.failures if r.check == "grad")
        self.assertIn("d/dy", msg)

    def test_gradient_function_errors_are_reported(self):
        def broken(x):
            raise RuntimeError("no gradient")

        rep = numguard.check(np.sin, grad=broken, include=["random"])
        self.assertIn("grad", failed(rep))
        rep = numguard.check(np.sin, grad=lambda x: (np.cos(x), np.cos(x)), include=["random"])
        self.assertIn("returned 2 array(s)", rep.failures[0].message)

    def test_spec_validation(self):
        with self.assertRaisesRegex(ValueError, "backend='jax'"):
            numguard.check(np.sin, grad=True)
        with self.assertRaisesRegex(TypeError, "grad must be"):
            numguard.check(np.sin, grad="yes")
        with self.assertRaisesRegex(ValueError, "grad_reference needs"):
            numguard.check(np.sin, grad_reference=np.cos)


@unittest.skipUnless(importlib.util.find_spec("jax"), "jax is not installed")
class JaxGradTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        import jax  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)

    def test_jax_functions_pass(self):
        import jax.numpy as jnp  # noqa: PLC0415
        import jax.scipy.special as jsp  # noqa: PLC0415

        for f, ref, dom in ((jnp.sin, np.sin, "any"), (jnp.tanh, np.tanh, "any"),
                            (jsp.gammaln, sp.gammaln, "positive"), (jsp.expit, sp.expit, "any")):
            rep = numguard.check(f, domain=dom, reference=ref, backend="jax", grad=True,
                                 elementwise=True, allow_nonfinite="*", dtypes=["float32", "float64"])
            bad = {c for c in failed(rep) if c.startswith("grad")}
            self.assertFalse(bad, (f, rep.summary()))
            self.assertGreater(rep.counts()["grad"]["pass"], 0)
            self.assertGreater(rep.counts()["grad:primal"]["pass"], 0)

    def test_custom_jvp_with_wrong_primal_fails(self):
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        @jax.custom_jvp
        def f(x):
            return jnp.exp(x)

        @f.defjvp
        def f_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            # the primal should be exp(x); this rule returns something else
            return jnp.exp(x) * (1 + 1e-3), jnp.exp(x) * t

        rep = numguard.check(f, inputs="[-3, 3]", reference=np.exp, backend="jax", grad=True,
                             elementwise=True, dtypes=["float64"])
        self.assertIn("grad:primal", failed(rep))

    def test_custom_jvp_with_wrong_tangent_fails(self):
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        @jax.custom_jvp
        def f(x):
            return jnp.sin(x)

        @f.defjvp
        def f_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return jnp.sin(x), jnp.cos(x) * t * jnp.where(jnp.abs(x) > 2, 1.05, 1.0)

        rep = numguard.check(f, inputs="[-3, 3]", reference=np.sin, backend="jax", grad=True,
                             elementwise=True, dtypes=["float64"])
        self.assertIn("grad", failed(rep))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class TorchGradTests(unittest.TestCase):
    def test_torch_functions_pass(self):
        import torch  # noqa: PLC0415

        for f, ref, dom in ((torch.sin, np.sin, "any"), (torch.tanh, np.tanh, "any"),
                            (torch.special.gammaln, sp.gammaln, "positive"),
                            (torch.sigmoid, sp.expit, "any")):
            rep = numguard.check(f, domain=dom, reference=ref, backend="torch", grad=True,
                                 elementwise=True, allow_nonfinite="*", dtypes=["float32", "float64"])
            bad = {c for c in failed(rep) if c.startswith("grad")}
            self.assertFalse(bad, (f, rep.summary()))
            self.assertGreater(rep.counts()["grad"]["pass"], 0)

    def test_custom_autograd_function_with_wrong_backward_fails(self):
        import torch  # noqa: PLC0415

        class BadExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.exp(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, g):
                (y,) = ctx.saved_tensors
                return g * y * 1.01

        rep = numguard.check(BadExp.apply, inputs="[-3, 3]", reference=np.exp, backend="torch",
                             grad=True, elementwise=True, dtypes=["float64"])
        self.assertIn("grad", failed(rep))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
