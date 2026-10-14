"""`numguard suggest` with a scripted model."""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

from click.testing import CliRunner

from tests import helpers  # noqa: F401
from tests.agent_fixtures import Repo, fake_llm, make_llm
from tests.fake_llm import ScriptedTransport

import numguard.specdoc
from numguard.agent.suggest import run_suggest
from numguard.cli import main
from numguard.config import load_config

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

GOOD_SPEC = {"inputs": [{"name": "x", "domain": "any"}], "dtypes": ["float32", "float64"],
             "properties": ["translation_equivariant"], "reference": "special.logsumexp(x)",
             "large_n": 2000, "rationale": "logsumexp is translation-equivariant; scipy's is stable."}


class SuggestTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.cleanup)

    def test_spec_found_reference_consistent_bugs_reported(self):
        t = ScriptedTransport("anthropic", [
            [("read_function", {}), ("list_catalogue", {})],
            [("grep", {"pattern": r"\blogsumexp\b"})],
            [("submit_spec", GOOD_SPEC)],
        ])
        # the model-written reference must only ever be evaluated in the subprocess worker
        with mock.patch.object(numguard.specdoc, "compile_reference", side_effect=AssertionError("in-process")):
            out = run_suggest("numops:logsumexp", llm=make_llm(t), root=self.repo.root, timeout=90)
        self.assertEqual(out.status, "ok", out.error)
        self.assertEqual(out.reference_status, "consistent")
        self.assertFalse(out.check_ok)
        self.assertIn("finite", {g["check"] for g in out.groups})
        self.assertIn("@numguard.guard(", out.decorator)
        self.assertIn("reference=lambda x: special.logsumexp(x)", out.decorator)
        self.assertEqual(tomllib.loads(out.toml)["target"][0]["function"], "numops:logsumexp")
        # the module was only imported by the subprocess worker, never by the process holding the key
        self.assertNotIn("numops", sys.modules)
        self.assertTrue(self.repo.imported)
        # tool outputs reached the model as delimited untrusted data; the catalogue is trusted text
        second = t.request_text(1)
        self.assertIn("untrusted-data id=", second)
        self.assertIn("def logsumexp(x):", second)
        self.assertIn("translation_equivariant", second)

    def test_invalid_spec_is_rejected_with_reasons(self):
        bad = dict(GOOD_SPEC, reference="__import__('os').system('touch PWNED')", dtypes=["float128"])
        t = ScriptedTransport("openai", [[("submit_spec", bad)], [("submit_spec", GOOD_SPEC)]])
        out = run_suggest("numops:logsumexp", llm=make_llm(t), root=self.repo.root, timeout=90)
        self.assertEqual(out.status, "ok")
        rejected = t.request_text(1)
        self.assertIn("REJECTED", rejected)
        self.assertIn("__import__", rejected)
        self.assertIn("dtypes", rejected)
        self.assertFalse((self.repo.root / "PWNED").exists())

    def test_reference_disagreeing_on_benign_inputs_is_suspect(self):
        wrong = {"inputs": ["any"], "reference": "x", "dtypes": ["float64"], "large_n": 2000,
                 "rationale": "identity"}
        t = ScriptedTransport("anthropic", [[("submit_spec", wrong)], [("submit_spec", wrong)]])
        out = run_suggest("numops:double", llm=make_llm(t), root=self.repo.root, rounds=2, timeout=90)
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.reference_status, "suspect")
        self.assertIn("random", out.suspect_cases)
        self.assertTrue(any("SUSPECT" in n for n in out.notes))
        # the reference's disagreements are not reported as bugs (the spec is re-run without it);
        # genuine findings remain: 2 * x overflows for the largest finite inputs
        checks = {g["check"] for g in out.groups}
        self.assertNotIn("reference", checks)
        self.assertEqual(checks, {"finite"})
        self.assertIn("ORDINARY inputs", t.request_text(1))

    def test_model_that_never_submits(self):
        t = ScriptedTransport("anthropic", ["hmm", "let me think", "no", "no"])
        out = run_suggest("numops:logsumexp", llm=make_llm(t), root=self.repo.root, timeout=90)
        self.assertEqual(out.status, "no_submission")

    def test_cli_writes_python_spec_and_toml(self):
        runner = CliRunner()
        cwd = os.getcwd()
        os.chdir(self.repo.root)
        try:
            t = ScriptedTransport("anthropic", [[("submit_spec", GOOD_SPEC)]])
            with fake_llm(t):
                res = runner.invoke(main, ["suggest", "numops:logsumexp", "--write", "spec_lse.py",
                                           "--json-out", "s.json"])
            self.assertEqual(res.exit_code, 1, res.output)  # the naive function fails its checks
            self.assertIn("reference: consistent", res.output)
            self.assertIn("# --- decorator", res.output)
            text = (self.repo.root / "spec_lse.py").read_text()
            self.assertIn("from numops import logsumexp", text)
            self.assertEqual(json.loads((self.repo.root / "s.json").read_text())["reference_status"], "consistent")
            # the written spec file is itself a runnable test, and `fix --spec` accepts it
            ns: dict = {"__name__": "spec_lse"}
            sys.path.insert(0, str(self.repo.root))
            try:
                exec(compile(text, "spec_lse.py", "exec"), ns)  # noqa: S102
                with self.assertRaises(AssertionError):
                    ns["test_logsumexp_numguard"]()
            finally:
                sys.path.remove(str(self.repo.root))
            t = ScriptedTransport("anthropic", [[("submit_spec", GOOD_SPEC)]])
            with fake_llm(t):
                res = runner.invoke(main, ["suggest", "numops:logsumexp", "--write", "extra.toml"])
            self.assertEqual(res.exit_code, 1, res.output)
            self.assertEqual(load_config(self.repo.root / "extra.toml").targets[0]["name"], "logsumexp")
            t = ScriptedTransport("anthropic", [[("submit_spec", GOOD_SPEC)]])
            with fake_llm(t):
                res = runner.invoke(main, ["suggest", "numops:logsumexp", "--write", "numguard.toml"])
            self.assertNotEqual(res.exit_code, 0)
            self.assertIn("already has a target named 'logsumexp'", res.output)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
