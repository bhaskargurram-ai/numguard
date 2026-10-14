"""`numguard fix` and the CI split (`agent propose` / `agent verify`) with a scripted model."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from click.testing import CliRunner

from tests import helpers  # noqa: F401
from tests.agent_fixtures import DENIED, MODULE, STABLE, WRONG, Repo, fake_llm, make_llm
from tests.fake_llm import ScriptedTransport

from numguard.agent.ci import ManifestError, load_manifest
from numguard.agent.fix import apply_rewrite, run_fix
from numguard.carddata import parse_card_data
from numguard.cli import main
from numguard.resolve import locate_source
from numguard.result import validate_result

SPEC = {"source": "doc", "doc": {"inputs": ["any"], "dtypes": ["float32", "float64"],
                                 "properties": ["translation_equivariant"],
                                 "reference": "special.logsumexp(x)", "large_n": 2000}}


class FixLoopTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.cleanup)

    def test_wrong_rewrite_rejected_then_stable_rewrite_verified(self):
        t = ScriptedTransport("anthropic", [
            [("read_file", {"path": "numops.py"}), ("show_failures", {})],
            [("submit_rewrite", {"code": WRONG, "rationale": "shift by the maximum"})],
            [("submit_rewrite", {"code": STABLE, "rationale": "also guard empty and non-finite inputs"})],
        ])
        out = run_fix("numops:logsumexp", llm=make_llm(t), root=self.repo.root, spec_req=SPEC, rounds=3, timeout=90)
        self.assertEqual(out.status, "verified", [a.reasons for a in out.attempts])
        self.assertEqual([a.status for a in out.attempts], ["rejected", "verified"])
        # the wrong rewrite was executed and rejected for its own new failures
        self.assertTrue(any("new failure" in r for r in out.attempts[0].reasons), out.attempts[0].reasons)
        self.assertGreater(out.before["n_failures"], 0)
        self.assertEqual(out.verified.n_failures, 0)
        self.assertIn("+    m = np.max(x)", out.diff)
        self.assertIn("--- a/numops.py", out.diff)
        entry = out.fix_entry()
        self.assertEqual(entry["status"], "verified")
        self.assertIn("finite", entry["fixed_checks"])
        self.assertEqual(entry["after_failures"], 0)
        # what the model saw: the failure table and source as delimited untrusted data
        first = t.request_text(0)
        self.assertIn("untrusted-report id=", first)
        self.assertIn("untrusted-source id=", first)
        self.assertIn("finite", first)
        # the feedback after round 1 contains the rewrite's own failure table
        feedback = t.request_text(2)
        self.assertIn("NOT accepted", feedback)
        self.assertIn("new failure", feedback)
        self.assertEqual(out.usage.requests, 3)

    def test_denylisted_rewrite_is_refused_before_execution(self):
        t = ScriptedTransport("openai", [
            [("submit_rewrite", {"code": DENIED, "rationale": "x"})],
            "I cannot do better.", "Still no.", "No.",
        ])
        out = run_fix("numops:logsumexp", llm=make_llm(t), root=self.repo.root, spec_req=SPEC, rounds=2, timeout=90)
        self.assertEqual(out.status, "refused")
        self.assertFalse((self.repo.root / "PWNED").exists())
        self.assertTrue(any("import of 'os' is not allowed" in r for r in out.attempts[0].reasons))
        # the model was told why, as the tool result of its submission
        self.assertIn("REJECTED", t.request_text(1))
        self.assertIn("NOT executed", t.request_text(1))

    def test_nothing_to_fix_makes_no_llm_call(self):
        t = ScriptedTransport("anthropic", [])
        spec = {"source": "doc", "doc": {"reference": "special.logsumexp(x)", "large_n": 2000}}
        out = run_fix("numops:stable_lse", llm=make_llm(t), root=self.repo.root, spec_req=spec, timeout=90)
        self.assertEqual(out.status, "nothing_to_fix")
        self.assertEqual(t.requests, [])

    def test_cli_fix_apply_then_check_passes(self):
        t = ScriptedTransport("anthropic", [[("submit_rewrite", {"code": STABLE, "rationale": "max-shift"})]])
        runner = CliRunner()
        cwd = os.getcwd()
        os.chdir(self.repo.root)
        try:
            with fake_llm(t):
                res = runner.invoke(main, ["fix", "logsumexp", "--apply", "--json-out", "fix.json"])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("VERIFIED after 1 round", res.output)
            self.assertIn("| finite", res.output)
            text = (self.repo.root / "numops.py").read_text()
            self.assertIn("m = np.max(x)", text)
            self.assertIn("def double(x):", text)  # the rest of the module is untouched
            self.assertIn('pathlib.Path(__file__).with_name("IMPORTED")', text)
            data = json.loads((self.repo.root / "fix.json").read_text())
            self.assertEqual(data["status"], "verified")
            res = runner.invoke(main, ["check", "--target", "logsumexp"])
            self.assertEqual(res.exit_code, 0, res.output)
        finally:
            os.chdir(cwd)

    def test_cli_fix_without_model_is_a_usage_error(self):
        res = CliRunner().invoke(main, ["fix", "numops:logsumexp", "--root", str(self.repo.root)],
                                 env={"NUMGUARD_LLM_PROVIDER": "anthropic", "NUMGUARD_LLM_API_KEY": "k",
                                      "NUMGUARD_LLM_MODEL": ""})
        self.assertEqual(res.exit_code, 2)
        self.assertIn("NUMGUARD_LLM_MODEL is not set", res.output)


class ApplyRewriteTests(unittest.TestCase):
    def test_keeps_decorators_and_adds_missing_imports(self):
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))
        (tmp / "m.py").write_text('"""doc"""\nimport numpy as np\nimport numguard\n\n\n@numguard.guard(domain="any")\n'
                                  'def f(x):\n    return np.log(np.sum(np.exp(x)))\n\n\ndef g():\n    pass\n')
        info = locate_source("m:f", tmp)
        new = apply_rewrite(info, "import math\nimport numpy as np\n\n\ndef f(x):\n    return math.fsum(x)\n")
        self.assertIn('import numguard\nimport math\n', new)
        self.assertEqual(new.count("import numpy as np"), 1)
        self.assertIn('@numguard.guard(domain="any")\ndef f(x):\n    return math.fsum(x)\n', new)
        self.assertIn("def g():", new)
        compile(new, "m.py", "exec")


class CISplitTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.cleanup)
        self.runner = CliRunner()
        self.out = self.repo.root.parent / (self.repo.root.name + "-out")
        self.out.mkdir()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.out, ignore_errors=True))
        cfg = str(self.repo.root / "numguard.toml")
        res = self.runner.invoke(main, ["check", "--config", cfg, "--json-out", str(self.out / "result.json"),
                                        "--head-sha", "c" * 40])
        self.assertEqual(res.exit_code, 1, res.output)
        (self.repo.root / "IMPORTED").unlink()  # the check job ran the code; reset the marker

    def propose(self, steps, provider="anthropic", variants=1):
        t = ScriptedTransport(provider, steps)
        with fake_llm(t):
            res = self.runner.invoke(main, ["agent", "propose", "--config", str(self.repo.root / "numguard.toml"),
                                            "--result", str(self.out / "result.json"), "--repo", str(self.repo.root),
                                            "--out-dir", str(self.out / "cands"), "--variants", str(variants)])
        self.assertEqual(res.exit_code, 0, res.output)
        return t, res

    def verify(self):
        return self.runner.invoke(main, ["agent", "verify", "--candidates", str(self.out / "cands"),
                                         "--config", str(self.repo.root / "numguard.toml"),
                                         "--result", str(self.out / "result.json"), "--repo", str(self.repo.root),
                                         "--json-out", str(self.out / "final.json"),
                                         "--card-out", str(self.out / "card.md")])

    def test_propose_runs_no_repository_code_and_verify_proves_the_fix(self):
        t, res = self.propose([[("grep", {"pattern": "logsumexp"})],
                               [("submit_rewrite", {"code": STABLE, "rationale": "max-shift"})]])
        self.assertIn("1 candidate rewrite(s)", res.output)
        self.assertFalse(self.repo.imported, "propose imported repository code")
        manifest = json.loads((self.out / "cands" / "manifest.json").read_text())
        self.assertEqual(manifest["candidates"][0]["status"], "submitted")
        self.assertEqual(manifest["candidates"][0]["file"], "cand_0.py")
        self.assertIn("finite", t.request_text(0))  # failure table from result.json
        res = self.verify()
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("logsumexp: verified", res.output)
        self.assertTrue(self.repo.imported)  # verify executes the code (in its own subprocess)
        doc = validate_result(json.loads((self.out / "final.json").read_text()))
        fix = doc["fixes"][0]
        self.assertEqual((fix["status"], fix["after_failures"], fix["model"]), ("verified", 0, "fake-model"))
        self.assertIn("+    m = np.max(x)", fix["diff"])
        card = (self.out / "card.md").read_text()
        self.assertIn("suggested fix (verified): diff", card)
        self.assertEqual(parse_card_data(card)["fixes"][0]["status"], "verified")
        # the source in the PR is untouched: verify only reports
        self.assertEqual((self.repo.root / "numops.py").read_text(), MODULE)

    def test_wrong_candidate_is_rejected_by_verify(self):
        self.propose([[("submit_rewrite", {"code": WRONG, "rationale": "shift"})]])
        res = self.verify()
        self.assertEqual(res.exit_code, 0, res.output)
        fix = json.loads((self.out / "final.json").read_text())["fixes"][0]
        self.assertEqual(fix["status"], "rejected")
        self.assertTrue(any("new failure" in r for r in fix["reasons"]))

    def test_verify_repeats_the_static_check_and_validates_the_manifest(self):
        self.propose([[("submit_rewrite", {"code": STABLE, "rationale": "ok"})]])
        cands = self.out / "cands"
        (cands / "cand_0.py").write_text(DENIED)  # tampered after propose
        res = self.verify()
        self.assertEqual(res.exit_code, 0, res.output)
        fix = json.loads((self.out / "final.json").read_text())["fixes"][0]
        self.assertEqual(fix["status"], "refused")
        self.assertFalse((self.repo.root / "PWNED").exists())
        m = json.loads((cands / "manifest.json").read_text())
        m["candidates"][0]["file"] = "../numops.py"
        (cands / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(ManifestError):
            load_manifest(cands)
        self.assertNotEqual(self.verify().exit_code, 0)

    def test_variants_are_independent_and_verify_keeps_the_best(self):
        t, res = self.propose([[("submit_rewrite", {"code": WRONG, "rationale": "first"})],
                               [("submit_rewrite", {"code": STABLE, "rationale": "second"})],
                               [("submit_rewrite", {"code": WRONG, "rationale": "third"})]], variants=3)
        self.assertIn("3 candidate rewrite(s)", res.output)
        self.assertEqual(len(t.requests), 3)  # one fresh session per variant
        self.assertIn("input's own dtype", t.request_text(1))
        self.assertIn("cast the result back", t.request_text(2))
        res = self.verify()
        self.assertEqual(res.exit_code, 0, res.output)
        fixes = json.loads((self.out / "final.json").read_text())["fixes"]
        self.assertEqual(len(fixes), 1)  # one entry per target, not per variant
        self.assertEqual((fixes[0]["status"], fixes[0]["rounds"], fixes[0]["rationale"]), ("verified", 3, "second"))
        self.assertIn("3 attempts", (self.out / "card.md").read_text())

    def test_a_failing_variant_does_not_lose_the_others(self):
        # the second session gets no scripted answer (the model call fails): the first still counts
        _, res = self.propose([[("submit_rewrite", {"code": STABLE, "rationale": "ok"})]], variants=2)
        m = json.loads((self.out / "cands" / "manifest.json").read_text())
        self.assertEqual([c["status"] for c in m["candidates"]], ["submitted", "error"])
        res = self.verify()
        self.assertEqual(json.loads((self.out / "final.json").read_text())["fixes"][0]["status"], "verified")

    def test_propose_is_a_no_op_when_the_agent_is_disabled(self):
        cfg = self.repo.root / "numguard.toml"
        cfg.write_text(cfg.read_text().replace("enabled = true", "enabled = false"))
        t, res = self.propose([])
        self.assertIn("nothing to do", res.output)
        self.assertEqual(t.requests, [])


if __name__ == "__main__":
    unittest.main()
