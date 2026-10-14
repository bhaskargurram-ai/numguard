"""The guard rails for model-written code: static denylist, scrubbed environment, subprocess worker."""

from __future__ import annotations

import os
import time
import unittest
from unittest import mock

from tests import helpers  # noqa: F401
from tests.agent_fixtures import HANGS, STABLE, Repo

from numguard.agent.sandbox import run_worker, scrubbed_env, static_check


class StaticCheckTests(unittest.TestCase):
    def test_accepts_plain_numerical_code(self):
        self.assertEqual(static_check(STABLE, "logsumexp"), [])
        code = "from scipy import special\nimport math\nEPS = 1e-12\n\ndef _h(x):\n    return x\n\n" \
               "def f(x):\n    with __import_guard__():\n        pass\n"
        self.assertTrue(any("dunder" in p for p in static_check(code, "f")))

    def test_denylist(self):
        cases = {
            "import os\ndef f(x): return x": "import of 'os' is not allowed",
            "import subprocess as sp\ndef f(x): return x": "'subprocess'",
            "from socket import socket\ndef f(x): return x": "'socket'",
            "import shutil\ndef f(x): return x": "'shutil'",
            "import ctypes\ndef f(x): return x": "'ctypes'",
            "import importlib\ndef f(x): return x": "'importlib'",
            "import requests\ndef f(x): return x": "'requests'",
            "import pandas\ndef f(x): return x": "import of 'pandas' is not allowed (allowed:",
            "def f(x):\n    open('out.txt', 'w').write('x')\n    return x": "open() is not allowed",
            "def f(x):\n    return eval('x')": "eval() is not allowed",
            "def f(x):\n    exec('y = 1')\n    return x": "exec() is not allowed",
            "def f(x):\n    return __import__('os')": "__import__() is not allowed",
            "def f(x):\n    return x.__class__.__bases__": "dunder attribute",
            "def f(x):\n    g = getattr\n    return x": "using 'getattr'",
            "import numpy as np\ndef f(x):\n    np.save('a.npy', x)\n    return x": "attribute 'save'",
            "def f(x):\n    global y\n    return x": "global statements",
            "from . import helpers\ndef f(x): return x": "relative imports",
            "print('hi')\ndef f(x): return x": "top-level Expr statements",
            "class A: pass\ndef f(x): return x": "top-level ClassDef",
            "def g(x): return x": "must define a top-level function named 'f'",
            "def f(x) return x": "syntax error",
            "": "non-empty",
        }
        for code, needle in cases.items():
            problems = static_check(code, "f")
            self.assertTrue(any(needle in p for p in problems), (code, problems))


class EnvTests(unittest.TestCase):
    def test_secrets_removed(self):
        env = scrubbed_env({"PATH": "/bin", "HOME": "/h", "NUMGUARD_LLM_API_KEY": "k", "NUMGUARD_LLM_MODEL": "m",
                            "GITHUB_TOKEN": "t", "AWS_SECRET_ACCESS_KEY": "s", "MY_PASSWORD": "p",
                            "ACTIONS_RUNTIME_TOKEN": "r", "PYTHONPATH": "/src", "NUMGUARD_CORPUS": "/c"})
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["PYTHONPATH"], "/src")
        self.assertEqual(env["NUMGUARD_CORPUS"], "/c")
        for gone in ("NUMGUARD_LLM_API_KEY", "NUMGUARD_LLM_MODEL", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                     "MY_PASSWORD", "ACTIONS_RUNTIME_TOKEN"):
            self.assertNotIn(gone, env)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.cleanup)
        self.req = {"root": str(self.repo.root), "target": "numops:logsumexp",
                    "spec": {"source": "doc", "doc": {"reference": "special.logsumexp(x)", "large_n": 1000,
                                                      "dtypes": ["float64"]}}}

    def test_checks_original_and_candidate(self):
        before = run_worker(self.req, timeout=60, cwd=self.repo.root)
        self.assertTrue(before.ok, before.error + before.stderr_tail)
        self.assertGreater(before.data["n_failures"], 0)
        self.assertIn("finite", {f[0] for f in before.data["failures"]})
        after = run_worker({**self.req, "candidate": {"code": STABLE, "name": "logsumexp"},
                            "compare_original": True}, timeout=60, cwd=self.repo.root)
        self.assertTrue(after.ok, after.error)
        self.assertEqual(after.data["n_failures"], 0, after.data["summary"])
        self.assertTrue(after.data["signature"]["match"])
        self.assertGreater(after.data["agreement"]["checked"], 0)
        self.assertEqual(after.data["agreement"]["mismatches"], [])

    def test_timeout_kills_the_worker(self):
        t0 = time.monotonic()
        res = run_worker({**self.req, "candidate": {"code": HANGS, "name": "logsumexp"}}, timeout=4,
                         cwd=self.repo.root)
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertIn("timed out after 4s", res.error)
        self.assertLess(time.monotonic() - t0, 30)

    def test_worker_never_sees_the_api_key(self):
        # simulate code that slipped past the static check and looks for the key
        leaky = ("import os\nimport numpy as np\n\ndef logsumexp(x):\n"
                 "    if os.environ.get('NUMGUARD_LLM_API_KEY') or os.environ.get('GITHUB_TOKEN'):\n"
                 "        raise RuntimeError('saw a secret')\n"
                 "    return np.log(np.sum(np.exp(x)))\n")
        with mock.patch.dict(os.environ, {"NUMGUARD_LLM_API_KEY": "sk-secret", "GITHUB_TOKEN": "ghs_x"}):
            res = run_worker({**self.req, "candidate": {"code": leaky, "name": "logsumexp"}}, timeout=60,
                             cwd=self.repo.root)
        self.assertTrue(res.ok, res.error)
        self.assertNotIn("saw a secret", json_text(res.data))

    def test_errors_are_reported_by_stage(self):
        res = run_worker({**self.req, "target": "numops:nope"}, timeout=60, cwd=self.repo.root)
        self.assertFalse(res.ok)
        self.assertEqual(res.data["stage"], "import")
        res = run_worker({**self.req, "candidate": {"code": "def other(x):\n    return x\n", "name": "logsumexp"}},
                         timeout=60, cwd=self.repo.root)
        self.assertEqual(res.data["stage"], "candidate")
        bad = dict(self.req, spec={"source": "doc", "doc": {"reference": "np.load('x')"}})
        res = run_worker(bad, timeout=60, cwd=self.repo.root)
        self.assertEqual(res.data["stage"], "spec")


def json_text(d):
    import json

    return json.dumps(d)


if __name__ == "__main__":
    unittest.main()
