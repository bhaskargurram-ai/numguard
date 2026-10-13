"""Shared test fixtures: small known-good and known-bad functions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORPUS = ROOT / "corpus"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("NUMGUARD_CORPUS", str(CORPUS))


def naive_logsumexp(x):
    return np.log(np.sum(np.exp(x)))


def stable_logsumexp(x):
    x = np.asarray(x)
    if x.size == 0:
        return x.dtype.type(-np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)


def nan_swallowing_sum(x):
    return np.nansum(x)


def upcasting_square(x):
    return np.asarray(x, dtype=np.float64) ** 2


def float_square(x):
    return x * x


def mutating_abs(x):
    np.abs(x, out=x)
    return x


def onepass_variance(x):
    n = x.size
    return ((np.sum(x * x) - np.sum(x) ** 2 / n) / n).astype(x.dtype)


def twopass_variance(x):
    return np.var(x)


def global_softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def rowwise_softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def sorted_desc(x):
    return -np.sort(-x)


def bumpy(x):
    """Non-monotonic elementwise function."""
    return np.sin(3 * x)
