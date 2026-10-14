"""numguard.toml, spec documents, reference expressions and `numguard check --config`."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from tests import helpers as h

import numguard
from numguard.cli import main
from numguard.config import ConfigError, load_config, parse_config
from numguard.reference import HighPrecReference
from numguard.result import validate_result
from numguard.specdoc import (
    MODEL_KEYS,
    SpecDocError,
    check_expression,
    compile_reference,
    render_decorator,
    render_python_spec,
    render_toml_target,
    spec_kwargs,
    validate_spec_doc,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def _pytest_available() -> bool:
    proc = subprocess.run([sys.executable, "-c", "import pytest"], capture_output=True, check=False)
    return proc.returncode == 0


class ReferenceExpressionTests(unittest.TestCase):
    def test_allowed_expressions(self):
        for expr in ("special.logsumexp(x)", "np.sqrt(a * a + b * b)", "math.fsum(x) / len(x)",
                     "np.where(x > 0, x, 0.0)", "sum(v for v in x)", "x[..., ::-1]"):
            self.assertEqual(check_expression(expr, ["x", "a", "b"]), [], expr)

    def test_rejected_expressions(self):
        cases = {
            "__import__('os').system('true')": "not allowed",
            "np.load('secrets.npy')": "'load' is not allowed",
            "x.__class__": "private attribute",
            "(lambda: 1)()": "Lambda is not allowed",
            "open('f', 'w')": "'open' is not allowed",
            "import os": "not a valid Python expression",
            "y + 1": "name 'y' is not available",
            "np.lib.format": "'lib' is not allowed",
            "getattr(np, 'load')": "'getattr' is not allowed",
            "x" + "+x" * 300: "too large",
            "x" + " + x" * 600: "longer than",
        }
        for expr, needle in cases.items():
            problems = check_expression(expr, ["x"])
            self.assertTrue(any(needle in p for p in problems), (expr, problems))

    def test_unknown_argument_names_allowed_before_import(self):
        self.assertEqual(check_expression("np.hypot(u, v)", None), [])
        self.assertTrue(check_expression("open(u)", None))

    def test_compile_and_precisions(self):
        ref = compile_reference("np.log1p(x)", ["x"])
        np.testing.assert_allclose(ref(np.array([1e-10])), [1e-10])
        self.assertEqual(ref.__name__, "np.log1p(x)")
        ld = compile_reference("np.log1p(x)", ["x"], "longdouble")
        self.assertIsInstance(ld, HighPrecReference)
        with self.assertRaises(SpecDocError):
            compile_reference("np.log1p(y)", ["x"])
        try:
            import mpmath  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(SpecDocError, "mpmath"):
                compile_reference("mpmath.log1p(x)", ["x"], "mpmath")
        else:  # pragma: no cover - mpmath installed
            mp = compile_reference("mpmath.log1p(x)", ["x"], "mpmath")
            self.assertAlmostEqual(float(mp(np.array([1e-10]))[0]), 1e-10)

    def test_builtins_are_restricted(self):
        ref = compile_reference("max(x) + abs(-1)", ["x"])
        self.assertEqual(ref([1.0, 2.0]), 3.0)
        # eval globals expose no real builtins: only the safe subset
        ref = compile_reference("len(x)", ["x"])
        self.assertEqual(ref([1, 2]), 2)


class SpecDocTests(unittest.TestCase):
    def test_validate_normalizes_and_rejects(self):
        doc = validate_spec_doc({"inputs": ["positive", {"name": "y", "domain": "[0, 1]", "shape": ["n", 3]}],
                                 "dtypes": ["float32"], "rtol": {"float32": 1e-4}, "bounds": [0, 1],
                                 "out_dtype": "none", "rationale": None})
        self.assertEqual(doc["inputs"][0], {"domain": "positive"})
        self.assertEqual(doc["bounds"], [0.0, 1.0])
        bad = {"dtypes": ["float128"], "nan": "maybe", "inputs": ["nope"], "bogus": 1, "large_n": -1,
               "properties": ["frobnicate"], "bounds": [2, 1], "kwargs": {"a b": 1}}
        with self.assertRaises(SpecDocError) as ctx:
            validate_spec_doc(bad)
        text = str(ctx.exception)
        for needle in ("unknown key(s): bogus", "dtypes", "nan", "inputs[0].domain", "large_n",
                       "unknown property", "bounds", "kwargs"):
            self.assertIn(needle, text)

    def test_model_keys_cannot_name_functions(self):
        with self.assertRaisesRegex(SpecDocError, "function"):
            validate_spec_doc({"function": "os:system"}, allowed=MODEL_KEYS)

    def test_inputs_must_match_arity(self):
        with self.assertRaisesRegex(SpecDocError, "2 required positional"):
            validate_spec_doc({"inputs": ["any"]}, arg_names=["x", "y"])

    def test_spec_kwargs_runs_like_check(self):
        doc = validate_spec_doc({"reference": "special.logsumexp(x)", "properties": ["translation_equivariant"],
                                 "dtypes": ["float64"]})
        rep = numguard.check(h.naive_logsumexp, **spec_kwargs(doc, h.naive_logsumexp))
        self.assertFalse(rep.ok)
        rep = numguard.check(h.stable_logsumexp, **spec_kwargs(doc, h.stable_logsumexp))
        self.assertTrue(rep.ok, rep.summary())


class RenderTests(unittest.TestCase):
    DOC = {"inputs": [{"name": "x", "domain": "any"}], "dtypes": ["float32", "float64"],
           "properties": ["translation_equivariant"], "reference": "special.logsumexp(x)",
           "nan": "propagate", "bounds": [-np.inf, np.inf]}

    def test_decorator_is_valid_python_and_equivalent(self):
        src = render_decorator(self.DOC, ["x"])
        self.assertIn("@numguard.guard(", src)
        self.assertIn("import scipy.special as special", src)
        ns: dict = {}
        exec(src + "def lse(x):\n    return np.log(np.sum(np.exp(x)))\n", {"np": np, **ns}, ns)  # noqa: S102
        rep = ns["lse"].numguard_check()
        self.assertFalse(rep.ok)
        self.assertIn("finite", rep.failed_checks())

    def test_toml_round_trip(self):
        text = render_toml_target(self.DOC, "pkg.mod:lse", name="lse")
        data = tomllib.loads(text)
        cfg = parse_config(data)
        self.assertEqual(cfg.targets[0]["name"], "lse")
        self.assertEqual(cfg.targets[0]["bounds"], [-np.inf, np.inf])
        self.assertEqual(cfg.targets[0]["reference"], "special.logsumexp(x)")

    def test_python_spec_file_runs(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "mymod.py").write_text("import numpy as np\n\ndef lse(x):\n    return np.log(np.sum(np.exp(x)))\n")
        text = render_python_spec(self.DOC, "mymod.py:lse", ["x"], header="Suggested by a test.")
        compile(text, "spec.py", "exec")
        (tmp / "spec_lse.py").write_text(text)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ns: dict = {"__name__": "spec_lse"}
            exec(compile(text, str(tmp / "spec_lse.py"), "exec"), ns)  # noqa: S102
            with self.assertRaises(numguard.NumGuardError):
                ns["test_lse_numguard"]()
        finally:
            os.chdir(cwd)


class ConfigTests(unittest.TestCase):
    def test_parse_defaults_names_and_sections(self):
        cfg = parse_config({
            "defaults": {"dtypes": ["float64"]},
            "target": [{"function": "a.b:f"}, {"function": "c.py:g", "dtypes": ["float32"], "name": "gg"}],
            "pytest": {"args": ["-k", "num"], "timeout": 60},
            "agent": {"enabled": True, "rounds": 2},
        })
        self.assertEqual([t["name"] for t in cfg.targets], ["f", "gg"])
        self.assertEqual(cfg.targets[0]["dtypes"], ["float64"])
        self.assertEqual(cfg.targets[1]["dtypes"], ["float32"])
        self.assertEqual(cfg.pytest.args, ["-k", "num"])
        self.assertTrue(cfg.agent.enabled)
        self.assertEqual(cfg.agent.rounds, 2)

    def test_errors(self):
        for data, needle in (
            ({"target": [{"function": "a:f"}, {"function": "b:f"}]}, "duplicate target name"),
            ({"target": [{"dtypes": ["float32"]}]}, "function: is required"),
            ({"target": [{"function": "a:f", "dtype": "float32"}]}, "unknown key(s): dtype"),
            ({"targets": []}, "unknown top-level"),
            ({"agent": {"api_key": "sk-123"}}, "never put API keys"),
            ({"pytest": {"args": "tests"}}, "pytest.args"),
            ({"agent": {"rounds": 99}}, "agent.rounds"),
        ):
            with self.assertRaises(ConfigError) as ctx:
                parse_config(data)
            self.assertIn(needle, str(ctx.exception))

    def test_invalid_toml(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "numguard.toml").write_text("[[target]\n")
        with self.assertRaisesRegex(ConfigError, "invalid TOML"):
            load_config(tmp / "numguard.toml")

    def test_repository_dogfood_config_is_valid(self):
        cfg = load_config(h.ROOT / "numguard.toml")
        self.assertGreaterEqual(len(cfg.targets), 5)


BUGGY_CONFIG = """
[defaults]
dtypes = ["float32", "float64"]

[[target]]
name = "lse_buggy"
function = "{corpus}/logsumexp/buggy.py:logsumexp"
properties = ["translation_equivariant"]
reference = "special.logsumexp(x)"

[[target]]
name = "lse_reference"
function = "{corpus}/logsumexp/reference.py:logsumexp"
properties = ["translation_equivariant"]
reference_function = "scipy.special:logsumexp"

[[target]]
name = "missing"
function = "no_such_module_xyz:f"
"""


class CheckConfigCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "numguard.toml").write_text(BUGGY_CONFIG.format(corpus=h.CORPUS.as_posix()))
        self.runner = CliRunner()

    def invoke(self, *args):
        return self.runner.invoke(main, ["check", "--config", str(self.tmp / "numguard.toml"), *args])

    def test_failures_errors_json_and_card(self):
        out = self.tmp / "out" / "result.json"
        card = self.tmp / "out" / "card.md"
        res = self.invoke("--json-out", str(out), "--card-out", str(card), "--pr-number", "7",
                          "--head-sha", "a" * 40, "--base-sha", "nothex")
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("lse_buggy FAILED", res.output)
        self.assertIn("missing ERROR", res.output)
        self.assertIn("1/3 targets passed; overall: ERROR", res.output)
        doc = validate_result(json.loads(out.read_text()))
        self.assertEqual(doc["status"], "error")
        self.assertEqual(doc["pr_number"], 7)
        self.assertEqual(doc["head"], "a" * 40)
        self.assertIsNone(doc["base"])
        by = {t["name"]: t for t in doc["targets"]}
        self.assertEqual(by["lse_buggy"]["status"], "fail")
        self.assertIn("finite", by["lse_buggy"]["failed_checks"])
        self.assertEqual(by["lse_reference"]["status"], "pass")
        self.assertIn("cannot import module", by["missing"]["error"])
        text = card.read_text()
        self.assertIn("<!-- numguard:card -->", text)
        self.assertIn("lse\\_buggy", text)
        self.assertTrue(text.rstrip().endswith("-->"))

    def test_target_selection_and_fail_on(self):
        res = self.invoke("--target", "lse_reference")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("1/1 targets passed", res.output)
        res = self.invoke("--target", "lse_buggy", "--fail-on", "never")
        self.assertEqual(res.exit_code, 0, res.output)
        res = self.invoke("--target", "lse_buggy", "--fail-on", "error")
        self.assertEqual(res.exit_code, 0, res.output)
        res = self.invoke("--target", "nope")
        self.assertEqual(res.exit_code, 2)
        self.assertIn("no target named", res.output)

    def test_json_to_stdout_and_dtype_override(self):
        res = self.invoke("--target", "lse_reference", "--json", "--dtype", "float64")
        self.assertEqual(res.exit_code, 0, res.output)
        doc = json.loads(res.output)
        self.assertEqual(doc["targets"][0]["dtypes"], ["float64"])

    def test_default_config_in_cwd_and_usage_errors(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            res = self.runner.invoke(main, ["check", "--target", "lse_reference"])
            self.assertEqual(res.exit_code, 0, res.output)
        finally:
            os.chdir(cwd)
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty)
        os.chdir(empty)
        try:
            res = self.runner.invoke(main, ["check"])
            self.assertEqual(res.exit_code, 2)
            self.assertIn("no numguard.toml", res.output)
        finally:
            os.chdir(cwd)
        res = self.runner.invoke(main, ["check", "numpy:sum", "--config", "x.toml"])
        self.assertEqual(res.exit_code, 2)

    @unittest.skipUnless(_pytest_available(), "pytest not importable in a subprocess")
    def test_pytest_section(self):
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "test_ok.py").write_text("def test_a():\n    assert True\n\n"
                                                        "def test_b():\n    assert 1 == 2\n")
        (self.tmp / "numguard.toml").write_text(textwrap.dedent(f"""
            [[target]]
            function = "{h.CORPUS.as_posix()}/logsumexp/reference.py:logsumexp"
            reference_function = "scipy.special:logsumexp"

            [pytest]
            args = ["tests"]
        """))
        out = self.tmp / "r.json"
        res = self.invoke("--json-out", str(out))
        self.assertEqual(res.exit_code, 1, res.output)
        doc = validate_result(json.loads(out.read_text()))
        self.assertEqual(doc["pytest"]["status"], "fail")
        self.assertEqual((doc["pytest"]["passed"], doc["pytest"]["failed"]), (1, 1))
        self.assertEqual(doc["status"], "fail")
        res = self.invoke("--no-pytest")
        self.assertEqual(res.exit_code, 0, res.output)

    def test_repository_config_passes(self):
        res = self.runner.invoke(main, ["check", "--config", str(h.ROOT / "numguard.toml"), "--no-pytest"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("overall: PASS", res.output)


if __name__ == "__main__":
    unittest.main()
