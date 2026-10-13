"""Every corpus task: the buggy version fails numguard, the reference passes."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import helpers as h

from numguard.corpus import (
    EvalResult,
    corpus_dir,
    grade_task,
    list_tasks,
    load_task,
    run_eval,
)

EXPECTED_TASKS = {
    "quantize_dynamic", "logsumexp", "softmax", "variance_onepass", "mean_float16",
    "sample_std", "log1p", "cosine_similarity", "normalize", "sigmoid", "hypot", "loo_mean",
}

#: The check(s) each buggy implementation is expected to trip (at minimum).
EXPECTED_FAILURES = {
    "quantize_dynamic": {"post:scale_positive"},
    "logsumexp": {"finite"},
    "softmax": {"batch", "finite"},
    "variance_onepass": {"reference", "bounds"},
    "mean_float16": {"finite"},
    "sample_std": {"reference"},
    "log1p": {"reference"},
    "cosine_similarity": {"finite"},
    "normalize": {"finite"},
    "sigmoid": {"fp_signals"},
    "hypot": {"finite"},
    "loo_mean": {"reference"},
}


class CorpusLayoutTests(unittest.TestCase):
    def test_corpus_found_and_complete(self):
        root = corpus_dir()
        self.assertEqual(root.resolve(), h.CORPUS.resolve())
        self.assertEqual(set(list_tasks()), EXPECTED_TASKS)
        for name in EXPECTED_TASKS:
            for f in ("task.md", "buggy.py", "reference.py", "spec.py"):
                self.assertTrue((root / name / f).is_file(), f"{name}/{f}")

    def test_task_metadata(self):
        task = load_task("logsumexp")
        self.assertEqual(task.function, "logsumexp")
        self.assertIn("def logsumexp", task.prompt)
        self.assertEqual(task.title(), "logsumexp")
        with self.assertRaises(FileNotFoundError):
            load_task("no_such_task")


class CorpusTaskTests(unittest.TestCase):
    """One subtest per task keeps failures readable."""

    def test_buggy_versions_fail_and_pass_happy_path(self):
        for name in sorted(EXPECTED_TASKS):
            with self.subTest(task=name):
                task = load_task(name)
                res = grade_task(task, task.path / "buggy.py")
                self.assertEqual(res.status, "fail", f"{name}: buggy version not detected")
                self.assertTrue(res.happy_path, f"{name}: buggy version should pass "
                                                f"the happy-path test")
                self.assertTrue(EXPECTED_FAILURES[name] <= set(res.failed_checks),
                                f"{name}: {res.failed_checks}")

    def test_reference_versions_pass(self):
        for name in sorted(EXPECTED_TASKS):
            with self.subTest(task=name):
                task = load_task(name)
                res = grade_task(task, task.path / "reference.py")
                self.assertEqual(res.status, "pass",
                                 res.report.summary() if res.report else res.error)
                self.assertTrue(res.happy_path)


class EvalHarnessTests(unittest.TestCase):
    def test_baselines(self):
        buggy = run_eval(baseline="buggy")
        ref = run_eval(baseline="reference")
        self.assertIsInstance(buggy, EvalResult)
        self.assertEqual(buggy.n_pass, 0)
        self.assertEqual(ref.n_pass, ref.n_tasks)
        self.assertEqual(buggy.n_happy, buggy.n_tasks)
        board = buggy.scoreboard()
        self.assertIn("numguard passed 0/12 tasks", board)
        data = json.loads(ref.to_json())
        self.assertEqual(data["n_pass"], 12)

    def test_candidate_dir_with_missing_and_broken_solutions(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "logsumexp").mkdir()
            shutil.copy(h.CORPUS / "logsumexp" / "reference.py", tmp / "logsumexp" / "solution.py")
            (tmp / "softmax").mkdir()
            (tmp / "softmax" / "solution.py").write_text("raise RuntimeError('boom')\n")
            (tmp / "hypot").mkdir()
            (tmp / "hypot" / "solution.py").write_text("def not_hypot(x, y):\n    return x\n")
            res = run_eval(candidate_dir=tmp, tasks=["logsumexp", "softmax", "hypot", "sigmoid"])
            status = {r.task: r.status for r in res.results}
            self.assertEqual(status, {"logsumexp": "pass", "softmax": "error",
                                      "hypot": "error", "sigmoid": "missing"})
            self.assertIn("RuntimeError", next(r for r in res.results
                                               if r.task == "softmax").error)
        finally:
            shutil.rmtree(tmp)

    def test_run_eval_argument_validation(self):
        with self.assertRaises(ValueError):
            run_eval()
        with self.assertRaises(ValueError):
            run_eval(baseline="buggy", candidate_dir=".")
        with self.assertRaises(ValueError):
            run_eval(baseline="perfect")


if __name__ == "__main__":
    unittest.main()
