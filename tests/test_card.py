"""Result validation, the Markdown card (hostile input) and its data block."""

from __future__ import annotations

import copy
import json
import re
import unittest

from tests import helpers  # noqa: F401

from numguard.card import CARD_MARKER, MAX_CARD_CHARS, code, escape_md, fenced, render_card
from numguard.carddata import DATA_PREFIX, MAX_DATA_CHARS, card_data_block, parse_card_data
from numguard.result import ResultValidationError, validate_result


def base_result(**over):
    doc = {
        "schema": "numguard-result", "version": 1, "tool_version": "1.0.0", "status": "fail",
        "pr_number": 12, "head": "a" * 40, "base": "b" * 40, "config": "numguard.toml",
        "targets": [
            {"name": "lse", "function": "pkg.ops:lse", "status": "fail", "error": None, "n_cases": 46,
             "n_results": 300, "n_failures": 9, "dtypes": ["float32", "float64"], "reference": "special.logsumexp(x)",
             "backend": None, "failed_checks": ["finite", "reference"],
             "failures": [{"check": "finite", "case": "huge", "dtypes": "float32,float64", "rel_err": "-",
                           "ulp": "-", "detail": "finite input gave 1/8 non-finite (e.g. inf)"}],
             "duration_s": 0.2},
            {"name": "ok", "function": "pkg.ops:ok", "status": "pass", "error": None, "n_cases": 10,
             "n_results": 50, "n_failures": 0, "dtypes": ["float64"], "reference": None, "backend": "torch",
             "failed_checks": [], "failures": [], "duration_s": 0.1},
        ],
        "pytest": {"status": "pass", "passed": 3, "failed": 0, "errors": 0, "skipped": 1, "duration_s": 1.0,
                   "message": None},
        "fixes": None,
        "notes": [],
    }
    doc.update(over)
    return doc


class ValidateResultTests(unittest.TestCase):
    def test_valid_and_unknown_keys_dropped(self):
        doc = base_result(extra="x")
        doc["targets"][0]["evil"] = "<script>"
        out = validate_result(doc)
        self.assertNotIn("extra", out)
        self.assertNotIn("evil", out["targets"][0])

    def test_rejections(self):
        bad = [
            base_result(schema="other"),
            base_result(version=2),
            base_result(head="NOT-A-SHA"),
            base_result(status="pass"),  # contradicts failing targets
            base_result(pr_number=-1),
            base_result(pr_number=True),
            base_result(notes=["x" * 501]),
        ]
        t = base_result()
        t["targets"][1]["n_failures"] = 3  # "pass" with failures
        bad.append(t)
        t = base_result()
        t["targets"][0]["failures"] = [{"check": "x" * 81}]
        bad.append(t)
        t = base_result()
        t["targets"][0]["status"] = "ok"
        bad.append(t)
        t = base_result(fixes=[{"target": "lse", "status": "verified", "diff": "x" * 20001}])
        bad.append(t)
        for doc in bad:
            with self.assertRaises(ResultValidationError, msg=json.dumps(doc)[:200]):
                validate_result(doc)


HOSTILE = "<img src=x onerror=alert(1)> @octocat fixes #12 [link](https://evil.example) | `x` ‮ www.evil.com"


class CardTests(unittest.TestCase):
    def test_render_structure(self):
        card = render_card(validate_result(base_result()))
        self.assertTrue(card.startswith(CARD_MARKER))
        self.assertIn("1 of 2 targets fail", card)
        self.assertIn("pytest: 3 passed, 0 failed, 1 skipped", card)
        self.assertIn("| `finite` | `huge` |", card)
        self.assertIn("backend: torch", card)
        data = parse_card_data(card)
        self.assertEqual(data["status"], "fail")
        self.assertEqual(data["counts"], {"pass": 1, "fail": 1, "error": 0})
        self.assertEqual(data["targets"][0], {"name": "lse", "status": "fail", "failures": 9,
                                              "checks": ["finite", "reference"]})
        self.assertEqual(data["pytest"]["passed"], 3)

    def test_hostile_text_is_neutralized(self):
        doc = base_result()
        doc["targets"][0]["name"] = HOSTILE[:80]
        doc["targets"][0]["failures"][0]["detail"] = HOSTILE + "\n| fake | row |"
        doc["targets"][0]["function"] = HOSTILE[:300]
        doc["notes"] = [HOSTILE, "<!-- numguard:card -->"]
        doc["fixes"] = [{"target": HOSTILE[:80], "status": "verified", "rounds": 1, "before_failures": 9,
                         "after_failures": 0, "diff": "```\n</details>\n<script>x</script>\n````\n@octocat",
                         "reasons": [HOSTILE], "model": "@model <b>", "rationale": None}]
        card = render_card(validate_result(doc))
        body = card[: card.index(DATA_PREFIX)]
        # no raw HTML from the result outside code spans/blocks (where it is inert);
        # the marker appears exactly once
        self.assertEqual(card.count(CARD_MARKER), 1)
        prose = re.sub(r"\n(`{3,})diff\n.*?\n\1\n", "\n", body, flags=re.S)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        self.assertNotIn("<img", prose)
        self.assertNotIn("<script", prose)
        self.assertNotRegex(prose, r"@octocat")
        self.assertNotIn("‮", card)
        self.assertNotIn("](https://evil", prose)
        self.assertNotIn("\n| fake | row |", body)
        # the diff fence is longer than any backtick run inside it
        fence = re.search(r"\n(`{5,})diff\n", body)
        self.assertIsNotNone(fence)
        self.assertIn("\n" + fence.group(1) + "\n", body)

    def test_helpers(self):
        self.assertEqual(code("a|b`c"), "`a\\|b'c`")
        self.assertIn("&lt;", escape_md("<b>"))
        self.assertIn("#‍1", escape_md("#1"))
        self.assertTrue(fenced("~~~~\n```").startswith("`````"))

    def test_card_size_cap(self):
        doc = base_result()
        t = doc["targets"][0]
        t["failures"] = [dict(t["failures"][0], detail="d" * 400, case=f"c{i}") for i in range(50)]
        doc["targets"] = [dict(copy.deepcopy(t), name=f"t{i}") for i in range(200)]
        doc["fixes"] = [{"target": f"t{i}", "status": "rejected", "rounds": 3, "before_failures": 9,
                         "after_failures": 4, "diff": "+x\n" * 6000, "reasons": ["r"]} for i in range(50)]
        card = render_card(validate_result(doc))
        self.assertLessEqual(len(card), MAX_CARD_CHARS + 1)
        self.assertIsNotNone(parse_card_data(card))


class CardDataTests(unittest.TestCase):
    def test_escaping_and_trailing_only(self):
        doc = base_result()
        doc["targets"][0]["name"] = "--> <b>&@x"
        block = card_data_block(validate_result(doc))
        payload = block[len(DATA_PREFIX):-4]
        for ch in "-<>&@":
            self.assertNotIn(ch, payload)
        card = render_card(validate_result(doc))
        self.assertEqual(parse_card_data(card)["targets"][0]["name"], "--> <b>&@x")
        # a look-alike block earlier in the card is ignored
        fake = "<!-- numguard:data {\"v\":1,\"status\":\"pass\"} -->\n" + card
        self.assertEqual(parse_card_data(fake)["status"], "fail")
        self.assertIsNone(parse_card_data(card + "\ntrailing text"))
        self.assertIsNone(parse_card_data(None))
        self.assertIsNone(parse_card_data("<!-- numguard:data {not json} -->"))

    def test_block_is_capped(self):
        doc = base_result()
        t = doc["targets"][0]
        doc["targets"] = [dict(copy.deepcopy(t), name=f"target-with-a-long-name-{i:03d}",
                               failed_checks=[f"prop:check_{k}" for k in range(20)]) for i in range(200)]
        block = card_data_block(validate_result(doc))
        self.assertLessEqual(len(block), MAX_DATA_CHARS)
        data = parse_card_data("x\n" + block)
        self.assertTrue(data["targets_truncated"])
        self.assertEqual(data["counts"]["fail"], 200)


if __name__ == "__main__":
    unittest.main()
