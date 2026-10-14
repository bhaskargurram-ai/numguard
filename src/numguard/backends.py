"""Array-namespace backends: check PyTorch, JAX or CuPy functions with numguard.

numguard always *generates* cases and *compares* results in NumPy. A backend
adapter converts each NumPy input to the backend's array type before the
function under test is called, and converts the outputs back afterwards
(``np.from_dlpack`` / ``np.asarray`` / the backend's own export), recording
the backend dtype of every output so that the ``dtype`` check still sees,
for example, a ``torch.bfloat16`` result as bfloat16.

Supported names: ``numpy`` (default; no conversion), ``torch``, ``jax`` and
``cupy``. The non-NumPy backends are optional: they are imported lazily, and
selecting one that is not installed raises :class:`BackendError`.

The reference implementation (if any) is always called with NumPy arrays.

Notes per backend:

* **torch**: inputs become CPU tensors (``torch.from_numpy``; set
  ``NUMGUARD_TORCH_DEVICE=cuda`` to move them to a GPU). In-place
  modification of an input is detected as with NumPy.
* **jax**: arrays are immutable, so the ``mutation`` check always passes.
  JAX computes in float32 unless ``JAX_ENABLE_X64=1``; numguard's ``dtype``
  check reports a float64 case that silently became float32.
* **cupy**: arrays live on the GPU; bfloat16 is not supported.
* **bfloat16** (any backend): cases are stored as bfloat16-representable
  float32 values and converted to the backend's bfloat16 type
  (``ml_dtypes.bfloat16`` for NumPy, so that needs ``ml_dtypes``).
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dtypes import BFLOAT16, dtype_name, ml_dtypes_bfloat16

__all__ = ["BACKENDS", "Backend", "BackendError", "BackendCall", "get_backend", "wrap"]

BACKENDS: tuple[str, ...] = ("numpy", "torch", "jax", "cupy")


class BackendError(RuntimeError):
    """The requested backend cannot be used (not installed, unsupported dtype)."""


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise BackendError(f"backend {name.split('.')[0]!r} is not installed ({exc})") from exc


def _generic_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__dlpack__") and not isinstance(x, (np.generic, int, float, complex, bool)):
        try:
            return np.from_dlpack(x)
        except (TypeError, ValueError, RuntimeError, BufferError, AttributeError):
            pass
    return np.asarray(x)


class Backend:
    """Base adapter: NumPy in, NumPy out (no conversion)."""

    name = "numpy"

    def to_backend(self, a: np.ndarray, dtname: str) -> Any:
        if dtname == BFLOAT16 and a.dtype == np.float32:
            bf16 = ml_dtypes_bfloat16()
            if bf16 is None:
                raise BackendError("bfloat16 with the numpy backend needs ml_dtypes (pip install ml_dtypes)")
            return a.astype(bf16)
        return a

    def to_numpy(self, x: Any) -> tuple[Any, str | None]:
        """``(numpy_value, backend_dtype_name)``; python scalars are passed through."""
        if isinstance(x, (int, float, complex, bool)) and not isinstance(x, np.generic):
            return x, None
        arr = _generic_to_numpy(x)
        if str(arr.dtype) == BFLOAT16:
            return arr.astype(np.float32), BFLOAT16
        return arr, arr.dtype.name


class TorchBackend(Backend):
    name = "torch"

    def __init__(self) -> None:
        self.torch = _import("torch")
        self.device = os.environ.get("NUMGUARD_TORCH_DEVICE", "").strip() or None

    def to_backend(self, a: np.ndarray, dtname: str) -> Any:
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(a))
        if dtname == BFLOAT16 and a.dtype == np.float32:
            t = t.to(torch.bfloat16)
        if self.device:
            t = t.to(self.device)
        return t

    def to_numpy(self, x: Any) -> tuple[Any, str | None]:
        torch = self.torch
        if isinstance(x, torch.Tensor):
            t = x.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if t.dtype == torch.bfloat16:
                return t.to(torch.float32).numpy(), BFLOAT16
            arr = t.numpy()
            return arr, arr.dtype.name
        return super().to_numpy(x)


class JaxBackend(Backend):
    name = "jax"

    def __init__(self) -> None:
        self.jnp = _import("jax.numpy")

    def to_backend(self, a: np.ndarray, dtname: str) -> Any:
        if dtname == BFLOAT16 and a.dtype == np.float32:
            return self.jnp.asarray(a, dtype=self.jnp.bfloat16)
        return self.jnp.asarray(a)

    def to_numpy(self, x: Any) -> tuple[Any, str | None]:
        if isinstance(x, (int, float, complex, bool)) and not isinstance(x, np.generic):
            return x, None
        arr = np.asarray(x)
        if str(arr.dtype) == BFLOAT16:
            return arr.astype(np.float32), BFLOAT16
        return arr, arr.dtype.name


class CupyBackend(Backend):
    name = "cupy"

    def __init__(self) -> None:
        self.cp = _import("cupy")

    def to_backend(self, a: np.ndarray, dtname: str) -> Any:
        if dtname == BFLOAT16:
            raise BackendError("cupy has no bfloat16 dtype")
        return self.cp.asarray(a)

    def to_numpy(self, x: Any) -> tuple[Any, str | None]:
        if isinstance(x, self.cp.ndarray):
            arr = self.cp.asnumpy(x)
            return arr, arr.dtype.name
        return super().to_numpy(x)


_FACTORIES: dict[str, Callable[[], Backend]] = {
    "numpy": Backend,
    "torch": TorchBackend,
    "jax": JaxBackend,
    "cupy": CupyBackend,
}


def get_backend(name: str) -> Backend:
    """Instantiate a backend adapter by name (raises :class:`BackendError`)."""
    key = (name or "numpy").strip().lower()
    if key not in _FACTORIES:
        raise BackendError(f"unknown backend {name!r}; use one of {', '.join(BACKENDS)}")
    return _FACTORIES[key]()


class BackendCall:
    """Callable wrapper: converts inputs to the backend and outputs back to NumPy.

    ``set_case`` tells it the dtype name of every argument of the current case
    (needed for bfloat16, whose values are stored as float32). After each call
    :meth:`last_out_dtypes` lists the backend dtype of every output, in the
    order numguard normalizes outputs (tuples in order, dicts by sorted key).
    """

    def __init__(self, fn: Callable[..., Any], backend: Backend):
        self.fn = fn
        self.backend = backend
        self._dtnames: list[str] = []
        self._last: list[str] | None = None
        self.__name__ = getattr(fn, "__name__", "function")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__wrapped__ = fn

    def set_case(self, dtnames: Sequence[str]) -> None:
        self._dtnames = list(dtnames)

    def _dt(self, i: int, a: np.ndarray) -> str:
        if i < len(self._dtnames):
            return self._dtnames[i]
        return a.dtype.name

    def _snapshot(self, a: Any, b: Any) -> np.ndarray | None:
        if b is a or not isinstance(a, np.ndarray):
            return None
        return np.array(self.backend.to_numpy(b)[0], copy=True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bargs = [self.backend.to_backend(a, self._dt(i, a)) if isinstance(a, np.ndarray) else a
                 for i, a in enumerate(args)]
        before = [self._snapshot(a, b) for a, b in zip(args, bargs)]
        out = self.fn(*bargs, **kwargs)
        # numguard's mutation check compares the NumPy inputs before and after
        # the call: copy in-place changes of converted inputs back into them
        for a, b, snap in zip(args, bargs, before):
            if snap is None:
                continue
            after = np.asarray(self.backend.to_numpy(b)[0])
            if after.shape != snap.shape or after.tobytes() != snap.tobytes():
                with np.errstate(all="ignore"):
                    if after.shape == a.shape:
                        np.copyto(a, after, casting="unsafe")
                    else:  # pragma: no cover - a backend changed the shape in place
                        a[...] = np.nan if a.dtype.kind == "f" else 0
        names: list[str] = []
        converted = self._convert(out, names)
        self._last = names
        return converted

    def _convert(self, out: Any, names: list[str]) -> Any:
        if isinstance(out, Mapping):
            res = {}
            for k in sorted(out):
                res[k] = self._one(out[k], names)
            return res
        if isinstance(out, tuple):
            return tuple(self._one(o, names) for o in out)
        return self._one(out, names)

    def _one(self, x: Any, names: list[str]) -> Any:
        val, name = self.backend.to_numpy(x)
        names.append(name or np.asarray(val).dtype.name)
        return val

    def last_out_dtypes(self) -> list[str] | None:
        return self._last


def wrap(fn: Callable[..., Any], backend: str | Backend, dtypes: Sequence[str] = ()) -> Callable[..., Any]:
    """Return ``fn`` wrapped for ``backend`` (``fn`` itself for plain NumPy without bfloat16)."""
    be = backend if isinstance(backend, Backend) else get_backend(backend)
    bf16 = BFLOAT16 in {dtype_name(d) for d in dtypes}
    if be.name == "numpy":
        if not bf16:
            return fn
        if ml_dtypes_bfloat16() is None:
            raise BackendError("bfloat16 with the numpy backend needs ml_dtypes (pip install ml_dtypes)")
    if be.name == "cupy" and bf16:
        raise BackendError("cupy has no bfloat16 dtype")
    return BackendCall(fn, be)
