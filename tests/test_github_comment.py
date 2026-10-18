"""`numguard comment` / `numguard summary` against a mocked GitHub API."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from tests import helpers  # noqa: F401
from tests.fake_github import FakeGitHub
from tests.test_card import base_result

from numguard import github_api
from numguard.card import CARD_MARKER
from numguard.carddata import parse_card_data
from numguard.cli import main
from numguard.github_api import GitHubError, find_card_comment, harness_changes, validate_number


class CommentCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.result = self.tmp / "result.json"
        self.result.write_text(json.dumps(base_result()), encoding="utf-8")
        self.gh = FakeGitHub()
        patcher = mock.patch.object(github_api, "urllib_transport", self.gh)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.env = {"GITHUB_TOKEN": "t0k", "GITHUB_REPOSITORY": "o/r"}
        self.runner = CliRunner()

    def comment(self, *args):
        with mock.patch.dict(os.environ, self.env):
            return self.runner.invoke(main, ["comment", "--json", str(self.result), *args])

    def test_create_then_update_one_sticky_comment(self):
        res = self.comment("--pr", "12", "--expect-head-sha", "a" * 40)
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("created numguard card comment 100 on #12", res.output)
        self.assertEqual(len(self.gh.comments), 1)
        self.assertTrue(self.gh.comments[0]["body"].startswith(CARD_MARKER))
        res = self.comment("--pr", "12")
        self.assertIn("updated numguard card comment 100", res.output)
        self.assertEqual(len(self.gh.comments), 1)
        self.assertEqual(parse_card_data(self.gh.comments[0]["body"])["status"], "fail")

    def test_human_comment_with_marker_is_not_hijacked(self):
        self.gh.add_comment(CARD_MARKER + "\nfake card", login="mallory", typ="User")
        res = self.comment("--pr", "12")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("created", res.output)
        self.assertEqual(self.gh.comments[0]["body"], CARD_MARKER + "\nfake card")
        self.assertEqual(len(self.gh.comments), 2)

    def test_head_sha_mismatch_refuses(self):
        res = self.comment("--pr", "12", "--expect-head-sha", "b" * 40)
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("does not match", res.output)
        self.gh.head_sha = "c" * 40  # PR moved on since the run
        res = self.comment("--pr", "12", "--expect-head-sha", "a" * 40)
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("PR #12 head does not match", res.output)
        self.assertEqual(self.gh.comments, [])

    def test_harness_change_warning_and_labels(self):
        self.gh.files = ["numguard.toml", "src/x.py"]
        self.gh.issue_labels = ["numguard: pass", "keep-me"]
        res = self.comment("--pr", "12", "--labels")
        self.assertEqual(res.exit_code, 0, res.output)
        body = self.gh.comments[0]["body"]
        self.assertIn("[!WARNING]", body)
        self.assertIn("`numguard.toml`", body)
        self.assertEqual(parse_card_data(body)["warnings"], 1)
        self.assertEqual(sorted(self.gh.issue_labels), ["keep-me", "numguard: fail"])
        self.assertIn("numguard: fail", self.gh.labels)

    def test_invalid_inputs(self):
        self.assertNotEqual(self.comment("--pr", "12abc").exit_code, 0)
        self.result.write_text(json.dumps(base_result(status="pass")), encoding="utf-8")  # forged overall status
        res = self.comment("--pr", "12")
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("invalid result.json", res.output)
        self.result.write_text(json.dumps(base_result()), encoding="utf-8")
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            res = self.runner.invoke(main, ["comment", "--json", str(self.result), "--pr", "1"])
        self.assertIn("GITHUB_TOKEN is not set", res.output)

    def test_dry_run_and_pr_from_result(self):
        res = self.comment("--dry-run")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn(CARD_MARKER, res.output)
        self.assertEqual(self.gh.calls, [])

    def test_summary_writes_job_summary(self):
        out = self.tmp / "summary.md"
        with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(out)}):
            res = self.runner.invoke(main, ["summary", "--json", str(self.result)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn(CARD_MARKER, out.read_text(encoding="utf-8"))
        bad = self.tmp / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        res = self.runner.invoke(main, ["summary", "--json", str(bad)])
        self.assertNotEqual(res.exit_code, 0)


class GitHubHelpersTests(unittest.TestCase):
    def test_validate_number_and_harness(self):
        self.assertEqual(validate_number(" 42 "), 42)
        for bad in (True, 0, "-1", "1e3", "12345678901"):
            with self.assertRaises(GitHubError):
                validate_number(bad)
        self.assertEqual(harness_changes(["numguard.toml", "src/a.py", ".github/workflows/x.yml",
                                          "pkg/conftest.py"]),
                         [".github/workflows/x.yml", "numguard.toml", "pkg/conftest.py"])

    def test_find_card_comment_requires_bot(self):
        comments = [{"id": 1, "body": CARD_MARKER, "user": {"login": "me", "type": "User"}},
                    {"id": 2, "body": "  " + CARD_MARKER, "user": {"login": "bot", "type": "Bot"}}]
        self.assertEqual(find_card_comment(comments)["id"], 2)
        self.assertIsNone(find_card_comment(comments, bot_login="other"))


if __name__ == "__main__":
    unittest.main()
