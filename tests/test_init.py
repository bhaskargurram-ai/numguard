"""`numguard init` and the examples it must stay in sync with."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from tests import helpers as h

from numguard import __version__
from numguard.cli import main
from numguard.config import load_config
from numguard.init_cmd import InitError, init_repo

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class InitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_examples_are_exactly_what_init_writes(self):
        init_repo(self.tmp, agent=True)
        pairs = {
            ".github/workflows/numguard.yml": "examples/workflows/numguard.yml",
            ".github/workflows/numguard-comment.yml": "examples/workflows/numguard-comment.yml",
            ".github/workflows/numguard-agent.yml": "examples/workflows/numguard-agent.yml",
            "numguard.toml": "examples/numguard.toml",
        }
        for written, example in pairs.items():
            self.assertEqual((self.tmp / written).read_text(), (h.ROOT / example).read_text(),
                             f"{example} is out of date: regenerate it with `numguard init --agent`")
        self.assertEqual(sorted(p.name for p in (h.ROOT / "examples" / "workflows").glob("*.yml")),
                         sorted(Path(k).name for k in pairs if k.endswith(".yml")))

    def test_refuses_to_overwrite_and_writes_nothing(self):
        (self.tmp / "numguard.toml").write_text("# mine\n")
        with self.assertRaisesRegex(InitError, "refusing to overwrite existing file"):
            init_repo(self.tmp)
        self.assertFalse((self.tmp / ".github").exists())
        self.assertEqual((self.tmp / "numguard.toml").read_text(), "# mine\n")
        res = init_repo(self.tmp, force=True)
        self.assertIn("numguard.toml", res.written)
        self.assertNotIn(".github/workflows/numguard-agent.yml", res.written)

    def test_options_and_generated_config(self):
        res = init_repo(self.tmp, ref="v9.9.9", install_command="pip install .", targets=["pkg.ops:lse"])
        wf = (self.tmp / ".github/workflows/numguard.yml").read_text()
        self.assertIn("check.yml@v9.9.9", wf)
        self.assertIn("install-command: 'pip install .'", wf)
        cfg = load_config(self.tmp / "numguard.toml")
        self.assertEqual([t["function"] for t in cfg.targets], ["pkg.ops:lse"])
        self.assertFalse(cfg.agent.enabled)
        self.assertTrue(any("Open a pull request" in s for s in res.next_steps))
        for bad in ({"ref": "v1;rm -rf"}, {"ref": "../x"}, {"targets": ["no-colon"]},
                    {"install_command": "a\nb"}):
            with self.assertRaises(InitError):
                init_repo(self.tmp, force=True, **bad)

    def test_agent_config_is_enabled_and_default_ref_is_this_release(self):
        init_repo(self.tmp, agent=True)
        self.assertTrue(load_config(self.tmp / "numguard.toml").agent.enabled)
        self.assertIn(f"@v{__version__}", (self.tmp / ".github/workflows/numguard-agent.yml").read_text())

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_generated_workflows_parse(self):
        init_repo(self.tmp, agent=True, install_command="pip install 'a[b]'")
        for f in (self.tmp / ".github" / "workflows").glob("*.yml"):
            data = yaml.safe_load(f.read_text())
            self.assertIn("jobs", data)

    def test_cli(self):
        runner = CliRunner()
        res = runner.invoke(main, ["init", "--repo", str(self.tmp), "--agent"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("wrote .github/workflows/numguard-agent.yml", res.output)
        self.assertIn("gh secret set NUMGUARD_LLM_API_KEY", res.output)
        res = runner.invoke(main, ["init", "--repo", str(self.tmp)])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("--force", res.output)
        # a fresh config checks nothing, and says so
        res = runner.invoke(main, ["check", "--config", str(self.tmp / "numguard.toml")])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("nothing was checked", res.output)


if __name__ == "__main__":
    unittest.main()
