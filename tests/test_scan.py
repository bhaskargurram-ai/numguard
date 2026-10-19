"""Scan mode: probing, scanning a module, and the CLI command."""

from __future__ import annotations

import json
import shutil
import signal
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from tests import helpers as h  # noqa: F401  (puts src on sys.path)

from numguard.cli import main
from numguard.scan import probe, public_functions, scan_module

MODULE = textwrap.dedent('''
    import numpy as np

    __all__ = ["good_exp", "shared_loop_exp", "mean_center", "logsumexp", "four_args",
               "describe", "save_table", "safe_log", "Thing", "builtin_exp", "builtin_hypot"]


    class _Builtin:
        """Like a C builtin (torch.special.*): callable, but without a readable signature."""

        def __init__(self, f, nargs):
            self.f, self.nargs = f, nargs

        @property
        def __signature__(self):
            raise ValueError("no signature found for builtin")

        def __call__(self, *args):
            if len(args) != self.nargs:
                raise TypeError(f"takes {self.nargs} arguments ({len(args)} given)")
            return self.f(*args)

    def good_exp(x):
        return np.exp(x)

    def shared_loop_exp(x):
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

    def mean_center(x):
        return x - np.mean(x)

    def logsumexp(x):
        return np.log(np.sum(np.exp(x)))

    def four_args(a, b, c, d):
        return a + b + c + d

    def describe(x):
        return "an array"

    def save_table(x):
        raise AssertionError("must not be called")

    def safe_log(x):
        return np.log(x)

    class Thing:
        pass

    builtin_exp = _Builtin(np.exp, 1)
    builtin_hypot = _Builtin(np.hypot, 2)
''')

REFERENCE = textwrap.dedent('''
    import numpy as np
    from scipy.special import logsumexp as _lse

    def good_exp(x):
        return np.exp(x)

    def shared_loop_exp(x):
        return np.exp(x)

    def logsumexp(x):
        return _lse(x)
''')


HANG_MODULE = textwrap.dedent('''
    import os
    import signal
    import time

    import numpy as np

    __all__ = ["fine_exp", "crash", "stuck"]

    def fine_exp(x):
        return np.exp(x)

    def crash(x):
        os._exit(3)  # like a segfault in native code: the process just ends

    def stuck(x):
        # like a C loop that never returns to Python: the timer signal is never handled
        if hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            time.sleep(120)
        return np.exp(x)
''')


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="numguard-scan-"))
        (cls.tmp / "scan_target_mod.py").write_text(MODULE, encoding="utf-8")
        (cls.tmp / "scan_reference_mod.py").write_text(REFERENCE, encoding="utf-8")
        (cls.tmp / "scan_hang_mod.py").write_text(HANG_MODULE, encoding="utf-8")
        sys.path.insert(0, str(cls.tmp))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.tmp))
        for m in ("scan_target_mod", "scan_reference_mod", "scan_hang_mod"):
            sys.modules.pop(m, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_public_functions_respects_all_and_skips_classes(self):
        import scan_target_mod  # noqa: PLC0415

        names = [n for n, _ in public_functions(scan_target_mod)]
        self.assertIn("good_exp", names)
        self.assertNotIn("Thing", names)

    def test_probe(self):
        self.assertEqual(probe(np.exp, 1), ("any", True))
        self.assertEqual(probe(np.log, 1), ("positive", True))
        self.assertEqual(probe(lambda p: np.log(p / (1 - p)) + np.sqrt(1 - p) * np.sqrt(p), 1)[0], "unit_interval")
        self.assertEqual(probe(lambda x: x - np.mean(x), 1), ("any", False))
        self.assertIsInstance(probe(lambda x: "text", 1), str)

    def test_scan_without_reference(self):
        res = scan_module("scan_target_mod")
        by = {e.name: e for e in res.entries}
        self.assertEqual(by["good_exp"].status, "checked")
        self.assertEqual(by["good_exp"].failed_checks, [])
        self.assertTrue(by["good_exp"].elementwise)
        # probed alone, a function whose elements depend on each other does not look elementwise
        self.assertFalse(by["shared_loop_exp"].elementwise)
        self.assertFalse(by["mean_center"].elementwise)
        self.assertEqual(by["safe_log"].domain, "positive")
        self.assertEqual(by["four_args"].status, "skipped")
        self.assertEqual(by["describe"].status, "skipped")
        self.assertEqual(by["save_table"].reason, "not a numerical function (by name)")
        # no reference: the finite check is off, so overflow alone is not flagged
        self.assertNotIn("finite", by["logsumexp"].failed_checks)

    def test_functions_without_a_signature_are_probed_for_their_arity(self):
        res = scan_module("scan_target_mod", include=["^builtin_"])
        by = {e.name: e for e in res.entries}
        self.assertEqual((by["builtin_exp"].status, by["builtin_exp"].arity), ("checked", 1))
        self.assertEqual((by["builtin_hypot"].status, by["builtin_hypot"].arity), ("checked", 2))
        self.assertEqual(by["builtin_hypot"].failed_checks, [])

    def test_isolated_scan_survives_hangs_and_crashes(self):
        from numguard import scan as scan_mod  # noqa: PLC0415

        old = scan_mod.WORKER_GRACE_S
        scan_mod.WORKER_GRACE_S = 20.0
        try:
            res = scan_module("scan_hang_mod", isolate=True, timeout_s=2.0, dtypes=["float64"])
        finally:
            scan_mod.WORKER_GRACE_S = old
        by = {e.name: e for e in res.entries}
        self.assertEqual((by["fine_exp"].status, by["fine_exp"].failed_checks), ("checked", []))
        self.assertEqual(by["crash"].status, "error")
        self.assertIn("worker process died", by["crash"].reason)
        if hasattr(signal, "pthread_sigmask"):
            self.assertEqual(by["stuck"].status, "error")
            self.assertIn("killed after 22 s", by["stuck"].reason)

    def test_scan_with_reference_module(self):
        res = scan_module("scan_target_mod", reference_module="scan_reference_mod",
                          include=["^(good_exp|logsumexp|mean_center|shared_loop_exp)$"])
        by = {e.name: e for e in res.entries}
        # the reference is elementwise, so the isolation check runs and catches the shared loop
        self.assertTrue(by["shared_loop_exp"].elementwise)
        self.assertIn("isolation", by["shared_loop_exp"].failed_checks)
        self.assertEqual(by["good_exp"].reference, "scan_reference_mod.good_exp")
        self.assertEqual(by["good_exp"].failed_checks, [])
        self.assertTrue({"reference", "finite"} & set(by["logsumexp"].failed_checks))
        self.assertEqual(by["mean_center"].reference, "")
        self.assertEqual({e.name for e in res.flagged}, {"logsumexp", "shared_loop_exp"})

    def test_cli(self):
        runner = CliRunner()
        out = self.tmp / "scan.json"
        res = runner.invoke(main, ["scan", "scan_target_mod", "--ref", "scan_reference_mod",
                                   "--include", "^(good_exp|shared_loop_exp)$", "--json-out", str(out)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("FLAG shared_loop_exp", res.output)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], "numguard/scan")
        self.assertEqual(doc["n_flagged"], 1)
        res = runner.invoke(main, ["scan", "scan_target_mod", "--ref", "scan_reference_mod",
                                   "--include", "^shared_loop_exp$", "--fail-on-flag"])
        self.assertEqual(res.exit_code, 1, res.output)
        res = runner.invoke(main, ["scan", "no_such_module_xyz"])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("cannot import", res.output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ScanTimeoutTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(__import__("signal"), "setitimer"), "needs POSIX timers")
    def test_slow_function_is_recorded_and_the_scan_continues(self):
        import time as _time  # noqa: PLC0415
        import types  # noqa: PLC0415

        mod = types.ModuleType("scan_slow_mod")

        def slow(x):
            if np.size(x) > 1:
                _time.sleep(5)
            return np.exp(x)

        mod.slow = slow
        mod.fast = np.exp
        sys.modules["scan_slow_mod"] = mod
        try:
            t0 = _time.perf_counter()
            res = scan_module("scan_slow_mod", timeout_s=1.0)
            self.assertLess(_time.perf_counter() - t0, 4)
        finally:
            sys.modules.pop("scan_slow_mod", None)
        by = {e.name: e for e in res.entries}
        self.assertEqual(by["slow"].status, "error")
        self.assertIn("longer than", by["slow"].reason)
        self.assertEqual(by["fast"].status, "checked")
