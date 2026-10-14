"""A tiny stand-in for the parts of Hypothesis that numguard.strategies uses.

Each strategy is a sampler ``draw(rng)``; there is no shrinking or database.
It lets the test suite exercise numguard.strategies' composition logic when
the real Hypothesis is not installed (tests use the real one when it is).
"""

from __future__ import annotations

import types

import numpy as np


class Strategy:
    def __init__(self, draw):
        self._draw = draw

    def draw(self, rng):
        return self._draw(rng)

    def map(self, f):
        return Strategy(lambda rng: f(self.draw(rng)))

    def flatmap(self, f):
        return Strategy(lambda rng: f(self.draw(rng)).draw(rng))


def sampled_from(values):
    values = list(values)
    return Strategy(lambda rng: values[int(rng.integers(len(values)))])


def one_of(*strategies):
    return Strategy(lambda rng: strategies[int(rng.integers(len(strategies)))].draw(rng))


def integers(min_value, max_value):
    return Strategy(lambda rng: int(rng.integers(min_value, max_value, endpoint=True)))


def tuples(*strategies):
    return Strategy(lambda rng: tuple(s.draw(rng) for s in strategies))


def floats(min_value=None, max_value=None, *, width=64, allow_nan=True, allow_infinity=True,
           exclude_min=False, exclude_max=False):
    dt = {16: np.float16, 32: np.float32, 64: np.float64}[width]

    def draw(rng):
        lo = -1e6 if min_value is None else max(min_value, -1e6)
        hi = 1e6 if max_value is None else min(max_value, 1e6)
        choice = int(rng.integers(4))
        if choice == 0 and min_value is not None and not exclude_min:
            v = min_value
        elif choice == 1 and max_value is not None and not exclude_max:
            v = max_value
        else:
            v = rng.uniform(lo, hi)
            if exclude_min and v == min_value:
                v = (lo + hi) / 2
        out = float(dt(v))
        assert min_value is None or out >= min_value, (out, min_value)
        assert max_value is None or out <= max_value, (out, max_value)
        return out

    return Strategy(draw)


def arrays(dtype, shape, *, elements):
    def draw(rng):
        n = int(np.prod(shape, dtype=np.int64))
        vals = [elements.draw(rng) for _ in range(n)]
        return np.asarray(vals, dtype=np.float64).astype(dtype).reshape(shape) if n else np.zeros(shape, dtype)

    return Strategy(draw)


def make_modules():
    hyp = types.ModuleType("hypothesis")
    st = types.ModuleType("hypothesis.strategies")
    extra = types.ModuleType("hypothesis.extra")
    hnp = types.ModuleType("hypothesis.extra.numpy")
    for f in (sampled_from, one_of, integers, tuples, floats):
        setattr(st, f.__name__, f)
    hnp.arrays = arrays
    hyp.strategies = st
    hyp.extra = extra
    extra.numpy = hnp
    return {"hypothesis": hyp, "hypothesis.strategies": st, "hypothesis.extra": extra,
            "hypothesis.extra.numpy": hnp}
