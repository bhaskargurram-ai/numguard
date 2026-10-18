"""`numguard bench` with a scripted model."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from tests import helpers as h
from tests.agent_fixtures import fake_llm, make_llm
from tests.fake_llm import ScriptedTransport

from numguard.agent.bench import extract_code_block, run_bench
from numguard.cli import main


def corpus(task: str, which: str) -> str:
    return (h.CORPUS / task / f"{which}.py").read_text(encoding="utf-8")


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def script(self, provider="anthropic"):
        def answer_with(which_by_task):
            def step(payload):
                text = json.dumps(payload)
                for task, how in which_by_task.items():
                    if f"Implement `{task}(" in text:
                        return how
                raise AssertionError("unexpected task prompt")
            return step

        plan = {
            "logsumexp": [("submit_solution", {"code": corpus("logsumexp", "reference")})],
            "sigmoid": [("submit_solution", {"code": corpus("sigmoid", "buggy")})],
            "hypot": [("submit_solution", {"code": "import os\n" + corpus("hypot", "reference")})],
            "softmax": "Here you go:\n```python\n" + corpus("softmax", "reference") + "```\n",
        }
        return ScriptedTransport(provider, [answer_with(plan) for _ in plan])

    def test_bench_statuses_and_leaderboard(self):
        t = self.script()
        res = run_bench(make_llm(t), tasks=["logsumexp", "sigmoid", "hypot", "softmax"],
                        solutions_dir=self.tmp / "sol", timeout=120)
        by = {r.task: r for r in res.tasks}
        self.assertEqual(by["logsumexp"].status, "pass")
        self.assertEqual(by["sigmoid"].status, "fail")
        self.assertTrue(by["sigmoid"].happy_path)
        self.assertIn("fp_signals", by["sigmoid"].failed_checks)
        self.assertEqual(by["hypot"].status, "refused")
        self.assertIn("'os'", by["hypot"].reason)
        self.assertTrue((self.tmp / "sol" / "hypot" / "solution.py.refused").exists())
        self.assertFalse((self.tmp / "sol" / "hypot" / "solution.py").exists())
        self.assertEqual(by["softmax"].status, "pass")  # code block fallback
        d = res.to_dict()
        self.assertEqual((d["n_tasks"], d["n_pass"], d["pass_at_1"]), (4, 2, 0.5))
        self.assertEqual(d["usage"]["requests"], 4)
        row = res.leaderboard_row()
        self.assertTrue(row.startswith("| fake-model | 2/4 (50%) |"), row)
        self.assertIn("hypot: refused before execution", row)
        # every request forced the submit tool and carried the task prompt only
        payload = t.requests[0]["payload"]
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "submit_solution"})
        self.assertEqual([tool["name"] for tool in payload["tools"]], ["submit_solution"])

    def test_cli(self):
        t = self.script("openai")
        out = self.tmp / "bench.json"
        with fake_llm(t):
            res = CliRunner().invoke(main, ["bench", "--tasks", "logsumexp,sigmoid", "--task", "hypot",
                                            "--task", "softmax", "--out-dir", str(self.tmp / "s"),
                                            "--json-out", str(out)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("| model | pass@1 | happy-path | failures |", res.output)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["n_pass"], 2)
        res = CliRunner().invoke(main, ["bench", "--task", "nope", "--model", "m"])
        self.assertEqual(res.exit_code, 2)

    def test_extract_code_block(self):
        self.assertEqual(extract_code_block("x\n```python\na = 1\n```\n"), "a = 1\n")
        self.assertIsNone(extract_code_block("no code"))


if __name__ == "__main__":
    unittest.main()
