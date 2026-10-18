"""A tiny repository with a numerically naive function, for the agent tests."""

from __future__ import annotations

import shutil
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from numguard.agent import llm as llm_mod

MODULE = '''"""Numerical helpers (test fixture)."""

import pathlib

import numpy as np

# Importing this module leaves a marker file: tests use it to prove that the
# agent's propose step never imports (runs) repository code.
pathlib.Path(__file__).with_name("IMPORTED").write_text("imported")


def logsumexp(x):
    """Return log(sum(exp(x))) of a 1-D array, in x's dtype."""
    return np.log(np.sum(np.exp(x)))


def double(x):
    """Twice x."""
    return 2 * x


def stable_lse(x):
    if x.size == 0:
        return x.dtype.type(-np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)
'''

CONFIG = '''
[[target]]
name = "logsumexp"
function = "numops:logsumexp"
inputs = ["any"]
dtypes = ["float32", "float64"]
properties = ["translation_equivariant"]
reference = "special.logsumexp(x)"
large_n = 2000

[[target]]
name = "stable"
function = "numops:stable_lse"
reference = "special.logsumexp(x)"
large_n = 2000

[agent]
enabled = true
timeout = 60
'''

#: The textbook fix: shift by the maximum, guard empty and non-finite inputs.
STABLE = textwrap.dedent('''
    import numpy as np


    def logsumexp(x):
        """Return log(sum(exp(x))) of a 1-D array, in x's dtype."""
        x = np.asarray(x)
        if x.size == 0:
            return x.dtype.type(-np.inf)
        m = np.max(x)
        if not np.isfinite(m):
            return m
        return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)
''').lstrip()

#: Plausible but wrong: max-shifted, without the empty / +inf guards.
WRONG = textwrap.dedent('''
    import numpy as np


    def logsumexp(x):
        m = np.max(x)
        return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)
''').lstrip()

#: Denylisted: would touch the file system if it ever ran.
DENIED = textwrap.dedent('''
    import os
    import numpy as np

    os.makedirs("PWNED", exist_ok=True)


    def logsumexp(x):
        return np.log(np.sum(np.exp(x)))
''').lstrip()

#: Never returns.
HANGS = textwrap.dedent('''
    def logsumexp(x):
        while True:
            pass
''').lstrip()


class Repo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="numguard-agent-test-"))
        (self.root / "numops.py").write_text(MODULE, encoding="utf-8")
        (self.root / "numguard.toml").write_text(CONFIG, encoding="utf-8")
        _forget()

    @property
    def imported(self) -> bool:
        return (self.root / "IMPORTED").exists()

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        _forget(self.root)


def _forget(root: Path | None = None) -> None:
    """In-process checks import ``numops``: drop it (and the repo's sys.path entries) between tests."""
    import sys

    sys.modules.pop("numops", None)
    if root is not None:
        for p in (str(root), str(root / "src"), str(root.resolve()), str(root.resolve() / "src")):
            while p in sys.path:
                sys.path.remove(p)


@contextmanager
def fake_llm(transport, **env):
    """Route :func:`client_from_env` to a scripted transport, with a complete fake config."""
    base = {"NUMGUARD_LLM_PROVIDER": transport.provider, "NUMGUARD_LLM_MODEL": "fake-model",
            "NUMGUARD_LLM_API_KEY": "sk-test-not-a-real-key"}
    base.update(env)
    with mock.patch.object(llm_mod, "TRANSPORT_OVERRIDE", transport), mock.patch.dict("os.environ", base):
        yield


def make_llm(transport):
    return llm_mod.ToolLLM(transport.provider, "fake-model", "sk-test", transport=transport, sleep=lambda s: None,
                           max_output_tokens=8192)
