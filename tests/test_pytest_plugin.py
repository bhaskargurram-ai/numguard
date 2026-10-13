"""Tests for the pytest plugin (skipped when pytest is not installed).

They run pytest in a subprocess on generated test files with the plugin
loaded explicitly (``-p numguard.pytest_plugin``), so they also work from a
source checkout where the ``pytest11`` entry point is not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests import helpers as h

HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def run_pytest(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    paths = [str(h.SRC), str(tmp)] + [p for p in sys.path if p]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "numguard.pytest_plugin", "-q",
         "-p", "no:cacheprovider", *args],
        cwd=tmp, capture_output=True, text=True, env=env, check=False, timeout=300,
    )


@unittest.skipUnless(HAS_PYTEST, "pytest is not installed")
class PytestPluginTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "ngops.py").write_text(textwrap.dedent("""
            import numpy as np
            import numguard

            @numguard.guard(dtypes=["float64"])
            def naive_lse(x):
                return np.log(np.sum(np.exp(x)))

            @numguard.guard(domain="positive")
            def safe_log(x):
                return np.log(x)

            def not_guarded(x):
                return x
        """))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_test(self, body: str) -> None:
        (self.tmp / "test_generated.py").write_text(textwrap.dedent(body))

    def test_marker_failure_shows_summary_table(self):
        self.write_test("""
            import numpy as np
            import pytest
            import scipy.special as sp
            from ngops import naive_lse

            @pytest.mark.numguard(fn=lambda x: np.log(np.sum(np.exp(x))), reference=sp.logsumexp)
            def test_lse():
                pass

            @pytest.mark.numguard(np.tanh, monotonic="increasing", bounds=(-1, 1))
            def test_tanh():
                pass
        """)
        proc = run_pytest(self.tmp)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("1 failed, 1 passed", proc.stdout)
        self.assertIn("| finite", proc.stdout)
        self.assertIn("FAILED", proc.stdout)

    def test_marker_registered_and_strict_markers(self):
        self.write_test("""
            import numpy as np
            import pytest

            @pytest.mark.numguard(fn=np.sin, properties=["odd"])
            def test_sin():
                assert True
        """)
        proc = run_pytest(self.tmp, "--strict-markers")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        proc = run_pytest(self.tmp, "--markers")
        self.assertIn("@pytest.mark.numguard", proc.stdout)

    def test_auto_collection_of_guarded_functions(self):
        self.write_test("""
            def test_placeholder():
                pass
        """)
        proc = run_pytest(self.tmp, "--numguard-modules", "ngops", "-rA")
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, out)
        self.assertIn("1 failed, 2 passed", out)
        self.assertIn("naive_lse", out)
        self.assertNotIn("not_guarded", out)
        self.assertIn("| finite", out)

    def test_dtype_and_large_n_options(self):
        self.write_test("""
            import numpy as np
            import pytest

            def half_mean(x):
                return (np.sum(x, dtype=x.dtype) / x.dtype.type(x.size)).astype(x.dtype)

            @pytest.mark.numguard(fn=half_mean, may_raise=["empty"],
                                  allow_nonfinite=["empty", "overflow_prone"])
            def test_mean():
                pass
        """)
        # float64 with a moderate large_n: fine
        proc = run_pytest(self.tmp, "--numguard-dtypes", "float64", "--numguard-large-n", "1000")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # float16 with 100k elements: the sum overflows
        proc = run_pytest(self.tmp, "--numguard-dtypes", "float16",
                          "--numguard-large-n", "100000")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("large_n", proc.stdout)

    def test_bad_marker_usage(self):
        self.write_test("""
            import pytest

            @pytest.mark.numguard()
            def test_nothing():
                pass
        """)
        proc = run_pytest(self.tmp)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("needs a function", proc.stdout + proc.stderr)


class PluginModuleWithoutPytestTests(unittest.TestCase):
    def test_entry_point_declared(self):
        text = (h.ROOT / "pyproject.toml").read_text()
        self.assertIn('[project.entry-points.pytest11]', text)
        self.assertIn('numguard = "numguard.pytest_plugin"', text)

    @unittest.skipIf(HAS_PYTEST, "only meaningful without pytest")
    def test_plugin_import_requires_pytest(self):
        with self.assertRaises(ImportError):
            importlib.import_module("numguard.pytest_plugin")


if __name__ == "__main__":
    unittest.main()
