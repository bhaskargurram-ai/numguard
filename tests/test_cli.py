"""CLI tests via click's CliRunner."""

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

from click.testing import CliRunner

from tests import helpers as h

from numguard.cli import main, resolve_callable


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "ngdemo_mod.py").write_text(textwrap.dedent("""
            import numpy as np
            import numguard

            def naive_lse(x):
                return np.log(np.sum(np.exp(x)))

            def stable_lse(x):
                if x.size == 0:
                    return x.dtype.type(-np.inf)
                m = np.max(x)
                if not np.isfinite(m):
                    return m
                return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)

            def variance(x):
                return np.var(x)

            @numguard.guard(domain="positive", dtypes=["float64"], reference=np.log)
            def guarded_log(x):
                return np.log(x)
        """))
        sys.path.insert(0, str(self.tmp))
        self.runner = CliRunner()

    def tearDown(self):
        sys.path.remove(str(self.tmp))
        sys.modules.pop("ngdemo_mod", None)
        shutil.rmtree(self.tmp)


class CheckCommandTests(CliTestCase):
    def test_failing_check_exits_1_with_table(self):
        res = self.runner.invoke(main, ["check", "ngdemo_mod:naive_lse",
                                        "--ref", "ngdemo_mod:stable_lse",
                                        "--prop", "translation_equivariant"])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("FAILED", res.output)
        self.assertIn("| finite", res.output)

    def test_passing_check_exits_0(self):
        res = self.runner.invoke(main, ["check", "ngdemo_mod:variance", "--dtype", "float64",
                                        "--prop", "translation_invariant", "--domain", "any",
                                        "--exclude", "overflow_prone",
                                        # np.var([]) is NaN by design: allow it explicitly
                                        "--allow-nonfinite", "empty"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("PASSED", res.output)
        self.assertIn("dtypes: float64", res.output)

    def test_json_output(self):
        res = self.runner.invoke(main, ["check", "ngdemo_mod:naive_lse", "--json",
                                        "--dtype", "float32"])
        self.assertEqual(res.exit_code, 1)
        data = json.loads(res.output)
        self.assertFalse(data["ok"])
        self.assertEqual(data["dtypes"], ["float32"])

    def test_guarded_function_uses_attached_spec(self):
        res = self.runner.invoke(main, ["check", "ngdemo_mod:guarded_log"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("dtypes: float64", res.output)
        self.assertIn("reference: log", res.output)

    def test_file_path_target_and_bad_targets(self):
        res = self.runner.invoke(main, ["check", f"{self.tmp / 'ngdemo_mod.py'}:variance",
                                        "--dtype", "float64", "--exclude", "overflow_prone"])
        self.assertEqual(res.exit_code, 1, res.output)  # np.var([]) -> NaN is reported
        self.assertIn("| finite | empty", res.output)
        res = self.runner.invoke(main, ["check", "ngdemo_mod:nope"])
        self.assertEqual(res.exit_code, 2)
        res = self.runner.invoke(main, ["check", "no_such_module_xyz:f"])
        self.assertEqual(res.exit_code, 2)
        res = self.runner.invoke(main, ["check", "ngdemo_mod:variance", "--prop", "bogus"])
        self.assertEqual(res.exit_code, 2)

    def test_resolve_callable_forms(self):
        import numpy as np

        self.assertIs(resolve_callable("numpy:sum"), np.sum)
        self.assertIs(resolve_callable("numpy.linalg:norm"), np.linalg.norm)
        self.assertIs(resolve_callable("numpy.sum"), np.sum)


class EvalCommandTests(CliTestCase):
    def test_baseline_buggy_detects_everything(self):
        res = self.runner.invoke(main, ["eval", "--baseline", "buggy"])
        self.assertEqual(res.exit_code, 0, res.output)  # default expectation: all fail
        self.assertIn("numguard passed 0/12 tasks", res.output)
        res = self.runner.invoke(main, ["eval", "--baseline", "buggy", "--expect", "pass"])
        self.assertEqual(res.exit_code, 1)

    def test_baseline_reference_passes(self):
        res = self.runner.invoke(main, ["eval", "--baseline", "reference", "--json"])
        self.assertEqual(res.exit_code, 0, res.output)
        data = json.loads(res.output)
        self.assertEqual(data["n_pass"], data["n_tasks"])

    def test_candidate_dir_and_json_out(self):
        cand = self.tmp / "cand"
        (cand / "sigmoid").mkdir(parents=True)
        shutil.copy(h.CORPUS / "sigmoid" / "buggy.py", cand / "sigmoid" / "solution.py")
        out = self.tmp / "results.json"
        res = self.runner.invoke(main, ["eval", "--candidate-dir", str(cand),
                                        "--task", "sigmoid", "--json-out", str(out)])
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("fp_signals", res.output)
        data = json.loads(out.read_text())
        self.assertEqual(data["tasks"][0]["status"], "fail")
        self.assertIn("report", data["tasks"][0])

    def test_eval_usage_errors(self):
        self.assertEqual(self.runner.invoke(main, ["eval"]).exit_code, 2)
        res = self.runner.invoke(main, ["eval", "--baseline", "buggy", "--task", "nope"])
        self.assertEqual(res.exit_code, 2)

    def test_tasks_and_cases_commands(self):
        res = self.runner.invoke(main, ["tasks"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("quantize_dynamic", res.output)
        res = self.runner.invoke(main, ["tasks", "--show", "hypot"])
        self.assertIn("def hypot", res.output)
        res = self.runner.invoke(main, ["cases"])
        self.assertIn("subnormal", res.output)


class ModuleEntryPointTests(unittest.TestCase):
    def test_python_m_numguard_help(self):
        env = dict(os.environ, PYTHONPATH=str(h.SRC))
        proc = subprocess.run([sys.executable, "-m", "numguard", "--help"],
                              capture_output=True, text=True, env=env, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("eval", proc.stdout)
        self.assertIn("check", proc.stdout)


if __name__ == "__main__":
    unittest.main()
