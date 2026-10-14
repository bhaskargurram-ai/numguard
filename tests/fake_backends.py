"""Minimal fake ``torch`` and ``jax.numpy`` modules for testing the backend adapters.

They implement only what numguard's adapters and the test functions use:
tensors wrap NumPy arrays (``torch.from_numpy`` shares memory, like the real
thing), ``bfloat16`` values are emulated by rounding float32 values.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import numpy as np

from numguard.dtypes import round_bf16


class _DType:
    def __init__(self, name: str, np_dtype):
        self.name = name
        self.np_dtype = np_dtype

    def __repr__(self) -> str:
        return f"torch.{self.name}"


def make_fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    float16 = _DType("float16", np.float16)
    float32 = _DType("float32", np.float32)
    float64 = _DType("float64", np.float64)
    bfloat16 = _DType("bfloat16", np.float32)
    by_np = {np.dtype(np.float16): float16, np.dtype(np.float32): float32, np.dtype(np.float64): float64}

    class Tensor:
        def __init__(self, arr, dtype):
            self._a = arr
            self.dtype = dtype

        def _wrap(self, arr):
            arr = np.asarray(arr)
            if self.dtype is bfloat16:
                return Tensor(round_bf16(arr.astype(np.float32)), bfloat16)
            return Tensor(arr.astype(self.dtype.np_dtype), self.dtype)

        def to(self, target):
            if isinstance(target, _DType):
                if target is bfloat16:
                    return Tensor(round_bf16(self._a.astype(np.float32)), bfloat16)
                return Tensor(self._a.astype(target.np_dtype), target)
            return self  # a device

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            if self.dtype is bfloat16:
                raise TypeError("Got unsupported ScalarType BFloat16")
            return self._a

        def add_(self, value):
            self._a[...] = self._a + value
            return self

        def __sub__(self, other):
            o = other._a if isinstance(other, Tensor) else other
            return self._wrap(self._a - o)

        def __truediv__(self, other):
            o = other._a if isinstance(other, Tensor) else other
            return self._wrap(self._a / o)

        def __add__(self, other):
            o = other._a if isinstance(other, Tensor) else other
            return self._wrap(self._a + o)

    def from_numpy(arr):
        return Tensor(arr, by_np[arr.dtype])

    def _unary(f):
        def op(t, *args, **kw):
            with np.errstate(all="ignore"):
                return t._wrap(f(t._a, *args, **kw))
        return op

    torch.Tensor = Tensor
    torch.float16, torch.float32, torch.float64, torch.bfloat16 = float16, float32, float64, bfloat16
    torch.from_numpy = from_numpy
    torch.exp = _unary(np.exp)
    torch.log = _unary(np.log)
    torch.sum = _unary(np.sum)
    torch.max = _unary(np.max)
    torch.abs = _unary(np.abs)
    return torch


def make_fake_jax() -> tuple[types.ModuleType, types.ModuleType]:
    jax = types.ModuleType("jax")
    jnp = types.ModuleType("jax.numpy")

    class JaxArray:
        def __init__(self, arr):
            self._a = np.array(arr, copy=True)  # jax arrays never alias NumPy memory

        def __array__(self, dtype=None, copy=None):
            return self._a if dtype is None else self._a.astype(dtype)

    def asarray(a, dtype=None):
        a = np.asarray(a)
        if dtype == "bfloat16":
            raise TypeError("fake jax has no bfloat16")
        # like JAX without JAX_ENABLE_X64: float64 silently becomes float32
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        return JaxArray(a)

    jnp.asarray = asarray
    jnp.bfloat16 = "bfloat16"
    jnp.abs = lambda x: JaxArray(np.abs(np.asarray(x)))
    jnp.sum = lambda x: JaxArray(np.sum(np.asarray(x)))
    jax.numpy = jnp
    return jax, jnp


@contextmanager
def installed(**modules):
    """Temporarily put fake modules into ``sys.modules``."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
