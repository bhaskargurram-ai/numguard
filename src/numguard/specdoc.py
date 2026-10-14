"""Spec documents: numguard specs as plain data (TOML tables / JSON objects).

A *spec document* describes how to check one function without Python code:
input domains, dtypes, properties, a reference written as a small expression,
tolerances and options. ``[[target]]`` tables in ``numguard.toml`` are spec
documents, and so is what the ``numguard suggest`` agent submits.

Validation is strict: unknown keys, wrong types and unknown enum values are
errors (so typos in a config are caught, and a model's submission is either
exactly understood or rejected).

Reference expressions
---------------------
``reference`` is a Python *expression* over the function's argument names,
e.g. ``"special.logsumexp(x)"`` or ``"np.sqrt(a * a + b * b)"``. It is parsed
and checked statically, then compiled into a restricted namespace that only
contains ``np``/``numpy``, ``special`` (``scipy.special``, if installed),
``math``, ``mpmath`` (if installed) and a few harmless builtins. Statements,
lambdas, private/dunder attributes and file/OS-related NumPy functions are
rejected. ``reference_precision`` chooses how it is evaluated:

* ``"float64"`` (default): on float64-upcast inputs;
* ``"longdouble"``: on ``np.longdouble`` inputs (see :func:`numguard.longdouble`);
* ``"mpmath"``: elementwise, each name bound to an ``mpmath.mpf`` scalar,
  e.g. ``"mpmath.log1p(x)"`` (see :func:`numguard.mp_scalar`).

The static check is a guard rail, not a sandbox: expressions that come from a
model are only ever *evaluated* in numguard's subprocess worker (see
:mod:`numguard.agent.sandbox`).
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .backends import BACKENDS
from .cases import ArraySpec, Domain
from .dtypes import ALL_FLOAT_DTYPES, validate_dtype
from .properties import parse_property

__all__ = [
    "SpecDocError",
    "TARGET_KEYS",
    "MODEL_KEYS",
    "PRECISIONS",
    "validate_spec_doc",
    "check_expression",
    "compile_reference",
    "reference_namespace",
    "spec_kwargs",
    "render_decorator",
    "render_python_spec",
    "render_toml_target",
    "toml_value",
]

PRECISIONS = ("float64", "longdouble", "mpmath")
_NAN = ("propagate", "raise", "ignore")
_FP = ("ignore", "fail")
_MONO = ("increasing", "decreasing")
MAX_EXPR_CHARS = 2000
MAX_EXPR_NODES = 400
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TARGET_RE = re.compile(r"^[A-Za-z0-9_./\\-]+(\.py)?:[A-Za-z_][A-Za-z0-9_.]*$|^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")


class SpecDocError(ValueError):
    """A spec document is invalid. ``problems`` lists every issue found."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------
# Restricted reference expressions
# --------------------------------------------------------------------------

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Call,
    ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Tuple, ast.List, ast.ListComp, ast.GeneratorExp, ast.comprehension,
    ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.MatMult, ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.LShift, ast.RShift, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is,
    ast.IsNot, ast.In, ast.NotIn, ast.Starred,
)
_MODULE_NAMES = ("np", "numpy", "math", "special", "scipy", "mpmath")
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "float": float,
    "int": int, "bool": bool, "tuple": tuple, "list": list, "zip": zip, "enumerate": enumerate,
    "round": round, "pow": pow, "all": all, "any": any, "sorted": sorted, "reversed": reversed,
    "complex": complex,
}
#: NumPy / mpmath attributes that touch files, the OS, global state or C internals.
DENIED_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "fromregex", "memmap", "DataSource", "lib", "ctypeslib", "ctypes", "f2py", "testing",
    "distutils", "show_config", "show_runtime", "info", "source", "seterr", "seterrcall",
    "setbufsize", "set_printoptions", "printoptions", "get_include", "dump", "dumps", "os", "sys",
    "system", "popen", "libmp", "pickle", "mp", "workdps", "workprec", "byref", "frombuffer",
    "add_newdoc", "deprecate", "who", "lookfor", "rec", "char", "polynomial", "random",
})


_DENIED_NAMES = frozenset({"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
                           "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
                           "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property"})


def check_expression(expr: Any, arg_names: Sequence[str] | None) -> list[str]:
    """Static checks of a reference expression; returns a list of problems (empty if fine).

    ``arg_names=None`` means the function's argument names are not known yet
    (a config file is validated before anything is imported): then any free
    identifier except a few dangerous builtins is accepted, and the names are
    checked again when the expression is compiled for the actual function.
    """
    if not isinstance(expr, str) or not expr.strip():
        return ["reference must be a non-empty string expression"]
    if len(expr) > MAX_EXPR_CHARS:
        return [f"reference is longer than {MAX_EXPR_CHARS} characters"]
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        return [f"reference is not a valid Python expression: {exc.msg}"]
    problems: list[str] = []
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_EXPR_NODES:
        problems.append(f"reference expression is too large ({len(nodes)} nodes)")
    bound = set(arg_names or ())
    for node in nodes:
        if isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    allowed_names = bound | set(_MODULE_NAMES) | set(SAFE_BUILTINS) | {"True", "False", "None"}
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            problems.append(f"{type(node).__name__} is not allowed in a reference expression")
        elif isinstance(node, ast.Name) and (node.id in _DENIED_NAMES or node.id.startswith("__")):
            problems.append(f"name {node.id!r} is not allowed in a reference expression")
        elif isinstance(node, ast.Name) and arg_names is not None and node.id not in allowed_names:
            problems.append(f"name {node.id!r} is not available (arguments: {', '.join(arg_names) or '-'}; "
                            f"modules: {', '.join(_MODULE_NAMES)})")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"private attribute {node.attr!r} is not allowed")
            elif node.attr in DENIED_ATTRIBUTES:
                problems.append(f"attribute {node.attr!r} is not allowed in a reference expression")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bytes,)):
            problems.append("bytes constants are not allowed")
    return list(dict.fromkeys(problems))


def reference_namespace() -> dict[str, Any]:
    """The globals a reference expression is evaluated in."""
    import numpy as np  # noqa: PLC0415

    ns: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "np": np, "numpy": np, "math": math}
    try:
        special = importlib.import_module("scipy.special")
    except ImportError:
        special = None
    if special is not None:
        ns["special"] = special

        class _Scipy:  # only scipy.special is exposed
            pass

        sc = _Scipy()
        sc.special = special  # type: ignore[attr-defined]
        ns["scipy"] = sc
    try:
        ns["mpmath"] = importlib.import_module("mpmath")
    except ImportError:
        pass
    return ns


def compile_reference(expr: str, arg_names: Sequence[str], precision: str = "float64") -> Callable[..., Any]:
    """Compile a checked reference expression into a callable (see module docs)."""
    problems = check_expression(expr, arg_names)
    if precision not in PRECISIONS:
        problems.append(f"reference_precision must be one of {', '.join(PRECISIONS)}")
    if problems:
        raise SpecDocError(problems)
    code = compile(expr.strip(), "<numguard reference>", "eval")
    base = reference_namespace()
    names = list(arg_names)
    if precision == "mpmath" and "mpmath" not in base:
        raise SpecDocError(["reference_precision 'mpmath' needs mpmath (pip install mpmath)"])

    def ref(*args: Any) -> Any:
        if len(args) != len(names):
            raise TypeError(f"reference expects {len(names)} argument(s), got {len(args)}")
        g = dict(base)
        g.update(zip(names, args))
        return eval(code, g)  # noqa: S307 - statically checked expression, restricted globals

    label = " ".join(expr.split())
    ref.__name__ = ref.__qualname__ = label if len(label) <= 60 else label[:59] + "…"
    ref.__doc__ = expr
    from .reference import longdouble, mp_scalar  # noqa: PLC0415

    if precision == "longdouble":
        return longdouble(ref)
    if precision == "mpmath":
        return mp_scalar(ref)
    return ref


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

#: Keys a model may submit (``numguard suggest``).
MODEL_KEYS = (
    "inputs", "dtypes", "properties", "reference", "reference_precision", "rtol", "atol", "nan",
    "finite", "allow_nonfinite", "may_raise", "exclude", "include", "ignore", "bounds", "monotonic",
    "monotonic_arg", "batched", "batch_axis", "fp_signals", "out_dtype", "check_mutation", "kwargs",
    "n", "large_n", "seed", "ref_upcast", "domain", "rationale",
)
#: Keys of a ``[[target]]`` table (and of ``[defaults]``, except name/function).
TARGET_KEYS = MODEL_KEYS + ("name", "function", "backend", "reference_function", "extra_cases",
                            "description")


class _V:
    def __init__(self, where: str):
        self.where = where
        self.problems: list[str] = []

    def err(self, key: str, msg: str) -> None:
        self.problems.append(f"{self.where}{key}: {msg}")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str_list(v: Any, key: str, val: _V, *, maxlen: int = 50, item_len: int = 100) -> list[str] | None:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(x, str) and 0 < len(x) <= item_len for x in v):
        val.err(key, "must be a list of strings")
        return None
    if len(v) > maxlen:
        val.err(key, f"at most {maxlen} entries")
        return None
    return list(v)


def _tol(v: Any, key: str, val: _V) -> Any:
    if _is_num(v):
        if not v >= 0:
            val.err(key, "must be >= 0")
        return float(v)
    if isinstance(v, Mapping):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or not _is_num(x) or not x >= 0:
                val.err(key, "must be a number or a table of dtype name -> number >= 0")
                return None
            out[k] = float(x)
        return out
    val.err(key, "must be a number or a table of dtype name -> number")
    return None


def _input_entry(item: Any, i: int, val: _V) -> dict[str, Any] | None:
    key = f"inputs[{i}]"
    if isinstance(item, str):
        item = {"domain": item}
    if not isinstance(item, Mapping):
        val.err(key, "must be a domain string or a table {name, domain, shape, dtype}")
        return None
    extra = set(item) - {"name", "domain", "shape", "dtype"}
    if extra:
        val.err(key, f"unknown key(s) {', '.join(sorted(map(str, extra)))}")
        return None
    out: dict[str, Any] = {}
    if "name" in item:
        if not isinstance(item["name"], str) or not _NAME_RE.match(item["name"]):
            val.err(key + ".name", "must be a Python identifier")
        else:
            out["name"] = item["name"]
    dom = item.get("domain", "any")
    try:
        Domain.parse(dom)
        out["domain"] = dom
    except (TypeError, ValueError) as exc:
        val.err(key + ".domain", str(exc))
    if "shape" in item:
        shape = item["shape"]
        if not isinstance(shape, list) or len(shape) > 8 or not all(
                (isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 4096)
                or (isinstance(d, str) and _NAME_RE.match(d)) for d in shape):
            val.err(key + ".shape", "must be a list of ints (0..4096) and symbolic names like \"n\"")
        else:
            out["shape"] = list(shape)
    if "dtype" in item:
        try:
            out["dtype"] = validate_dtype(item["dtype"])
        except (TypeError, ValueError):
            val.err(key + ".dtype", f"unknown dtype {item['dtype']!r}")
    return out


def validate_spec_doc(doc: Any, *, allowed: Sequence[str] = TARGET_KEYS, where: str = "",
                      arg_names: Sequence[str] | None = None, require_function: bool = False) -> dict[str, Any]:
    """Validate and normalize a spec document. Raises :class:`SpecDocError`.

    ``None`` values mean "not set" and are dropped. ``arg_names`` (the
    function's positional parameters, when known) is used to check the
    number of inputs and the names used in ``reference``.
    """
    val = _V(f"{where}." if where else "")
    if not isinstance(doc, Mapping):
        raise SpecDocError([f"{where or 'spec'}: must be a table/object"])
    doc = {k: v for k, v in doc.items() if v is not None}
    unknown = sorted(str(k) for k in doc if k not in allowed)
    if unknown:
        val.err("", f"unknown key(s): {', '.join(unknown)} (allowed: {', '.join(allowed)})")
    out: dict[str, Any] = {}
    for key in ("name", "description", "rationale"):
        if key in doc:
            limit = 80 if key == "name" else 2000
            if not isinstance(doc[key], str) or len(doc[key]) > limit:
                val.err(key, f"must be a string of at most {limit} characters")
            else:
                out[key] = doc[key]
    if "function" in doc:
        f = doc["function"]
        if not isinstance(f, str) or not _TARGET_RE.match(f):
            val.err("function", "must look like 'package.module:function' or 'path/to/file.py:function'")
        else:
            out["function"] = f
    elif require_function:
        val.err("function", "is required")
    if "reference_function" in doc:
        f = doc["reference_function"]
        if not isinstance(f, str) or not _TARGET_RE.match(f):
            val.err("reference_function", "must look like 'module:function'")
        else:
            out["reference_function"] = f
    if "backend" in doc:
        if doc["backend"] not in BACKENDS:
            val.err("backend", f"must be one of {', '.join(BACKENDS)}")
        else:
            out["backend"] = doc["backend"]
    if "domain" in doc:
        try:
            Domain.parse(doc["domain"])
            out["domain"] = doc["domain"]
        except (TypeError, ValueError) as exc:
            val.err("domain", str(exc))
    if "inputs" in doc:
        items = doc["inputs"]
        if isinstance(items, (str, Mapping)):
            items = [items]
        if not isinstance(items, list) or not 1 <= len(items) <= 16:
            val.err("inputs", "must be a list of 1 to 16 input entries")
        else:
            entries = [_input_entry(it, i, val) for i, it in enumerate(items)]
            if all(e is not None for e in entries):
                out["inputs"] = entries
                if arg_names is not None and len(entries) != len(arg_names):
                    val.err("inputs", f"has {len(entries)} entries but the function has "
                                      f"{len(arg_names)} required positional argument(s): {', '.join(arg_names)}")
    if "dtypes" in doc:
        lst = _str_list(doc["dtypes"], "dtypes", val, maxlen=4)
        if lst is not None:
            bad = [d for d in lst if d not in ALL_FLOAT_DTYPES]
            if bad or not lst:
                val.err("dtypes", f"must be a non-empty list of {', '.join(ALL_FLOAT_DTYPES)}")
            else:
                out["dtypes"] = list(dict.fromkeys(lst))
    if "properties" in doc:
        lst = _str_list(doc["properties"], "properties", val, maxlen=12)
        if lst is not None:
            for p in lst:
                try:
                    parse_property(p)
                except (TypeError, ValueError) as exc:
                    val.err("properties", str(exc))
            out["properties"] = lst
    for key in ("allow_nonfinite", "may_raise", "exclude", "include", "ignore"):
        if key in doc:
            lst = _str_list(doc[key], key, val)
            if lst is not None:
                out[key] = lst
    for key, choices in (("nan", _NAN), ("fp_signals", _FP), ("monotonic", _MONO),
                         ("reference_precision", PRECISIONS)):
        if key in doc:
            if doc[key] not in choices:
                val.err(key, f"must be one of {', '.join(choices)}")
            else:
                out[key] = doc[key]
    for key in ("finite", "batched", "check_mutation", "ref_upcast"):
        if key in doc:
            if not isinstance(doc[key], bool):
                val.err(key, "must be true or false")
            else:
                out[key] = doc[key]
    for key, lo, hi in (("n", 1, 4096), ("large_n", 2, 10_000_000), ("seed", 0, 2 ** 32 - 1),
                        ("batch_axis", -8, 8), ("monotonic_arg", 0, 15)):
        if key in doc:
            v = doc[key]
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                val.err(key, f"must be an integer in [{lo}, {hi}]")
            else:
                out[key] = v
    for key in ("rtol", "atol"):
        if key in doc:
            t = _tol(doc[key], key, val)
            if t is not None:
                out[key] = t
    if "bounds" in doc:
        b = doc["bounds"]
        if not (isinstance(b, list) and len(b) == 2 and all(_is_num(x) for x in b) and b[0] <= b[1]):
            val.err("bounds", "must be [lo, hi] with lo <= hi")
        else:
            out["bounds"] = [float(b[0]), float(b[1])]
    if "out_dtype" in doc:
        od = doc["out_dtype"]

        def one(x: Any) -> Any:
            if x in ("preserve", "none"):
                return x
            return validate_dtype(x)

        try:
            out["out_dtype"] = [one(x) for x in od] if isinstance(od, list) else one(od)
        except (TypeError, ValueError):
            val.err("out_dtype", "must be \"preserve\", \"none\", a dtype name or a list of those")
    if "kwargs" in doc:
        kw = doc["kwargs"]
        if not isinstance(kw, Mapping) or len(kw) > 16 or not all(
                isinstance(k, str) and _NAME_RE.match(k)
                and (isinstance(v, (str, bool, int, float)) or v is None) for k, v in kw.items()):
            val.err("kwargs", "must be a table of identifier -> scalar (string, number, boolean)")
        else:
            out["kwargs"] = dict(kw)
    if "extra_cases" in doc:
        ec = doc["extra_cases"]
        if not isinstance(ec, Mapping) or len(ec) > 50 or not all(
                isinstance(k, str) and isinstance(v, list) for k, v in ec.items()):
            val.err("extra_cases", "must be a table of name -> list of numbers (or list of lists)")
        else:
            out["extra_cases"] = {k: v for k, v in ec.items()}
    if "reference" in doc:
        names = _reference_arg_names(out, arg_names) if arg_names is not None or any(
            "name" in e for e in out.get("inputs", [])) else None
        problems = check_expression(doc["reference"], names)
        for p in problems:
            val.err("reference", p)
        if not problems:
            out["reference"] = doc["reference"].strip()
    if "reference" in out and "reference_function" in out:
        val.err("reference", "give either reference or reference_function, not both")
    if val.problems:
        raise SpecDocError(val.problems)
    return out


def _reference_arg_names(doc: Mapping[str, Any], arg_names: Sequence[str] | None) -> list[str]:
    inputs = doc.get("inputs") or []
    names = [e.get("name") for e in inputs]
    if arg_names:
        names = [n or (arg_names[i] if i < len(arg_names) else None) for i, n in enumerate(names)] or list(arg_names)
    return [n or f"arg{i}" for i, n in enumerate(names)] or (list(arg_names) if arg_names else ["x"])


# --------------------------------------------------------------------------
# Spec document -> numguard.check() keyword arguments
# --------------------------------------------------------------------------


def spec_kwargs(doc: Mapping[str, Any], fn: Callable[..., Any] | None = None, *,
                reference: Callable[..., Any] | None = None, compile_ref: bool = True) -> dict[str, Any]:
    """Keyword arguments for :func:`numguard.check` from a *validated* spec document.

    ``reference`` overrides the document's reference; otherwise, with
    ``compile_ref``, a ``reference`` expression is compiled in-process (only
    do that for trusted documents such as your own ``numguard.toml``).
    """
    from .api import _positional_params  # noqa: PLC0415

    arg_names = _positional_params(fn) if fn is not None else None
    kw: dict[str, Any] = {}
    if "inputs" in doc:
        specs = {}
        for i, e in enumerate(doc["inputs"]):
            name = e.get("name") or (arg_names[i] if arg_names and i < len(arg_names) else f"arg{i}")
            shape = tuple(e["shape"]) if "shape" in e else ("n",)
            specs[name] = ArraySpec(shape=shape, domain=Domain.parse(e.get("domain", "any")),
                                    dtype=e.get("dtype"), name=name)
        kw["inputs"] = specs
    if "domain" in doc:
        kw["domain"] = doc["domain"]
    for key in ("dtypes", "properties", "allow_nonfinite", "may_raise", "exclude", "include", "ignore",
                "nan", "fp_signals", "monotonic", "finite", "batched", "check_mutation", "ref_upcast",
                "n", "large_n", "seed", "batch_axis", "monotonic_arg", "rtol", "atol", "kwargs", "backend",
                "extra_cases"):
        if key in doc:
            kw[key] = doc[key]
    if "bounds" in doc:
        kw["bounds"] = tuple(doc["bounds"])
    if "out_dtype" in doc:
        od = doc["out_dtype"]
        conv = (lambda x: None if x == "none" else x)
        kw["out_dtype"] = [conv(x) for x in od] if isinstance(od, list) else conv(od)
    if reference is not None:
        kw["reference"] = reference
    elif compile_ref and "reference" in doc:
        names = _reference_arg_names(doc, arg_names)
        kw["reference"] = compile_reference(doc["reference"], names, doc.get("reference_precision", "float64"))
    return kw


# --------------------------------------------------------------------------
# Rendering (suggest output)
# --------------------------------------------------------------------------


def toml_value(v: Any) -> str:
    """Render a scalar / list / table as an inline TOML value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{ " + ", ".join(f"{_toml_key(k)} = {toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot render {v!r} as TOML")


def _toml_key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


_DOC_ORDER = ("inputs", "domain", "dtypes", "properties", "reference", "reference_precision",
              "reference_function", "rtol", "atol", "nan", "finite", "allow_nonfinite", "may_raise",
              "exclude", "include", "ignore", "bounds", "monotonic", "monotonic_arg", "batched",
              "batch_axis", "fp_signals", "out_dtype", "check_mutation", "kwargs", "n", "large_n",
              "seed", "ref_upcast", "backend", "extra_cases")


def render_toml_target(doc: Mapping[str, Any], function: str, name: str | None = None) -> str:
    """A ``[[target]]`` table for ``numguard.toml``."""
    lines = ["[[target]]"]
    if name:
        lines.append(f"name = {toml_value(name)}")
    lines.append(f"function = {toml_value(function)}")
    for key in _DOC_ORDER:
        if key in doc:
            lines.append(f"{key} = {toml_value(doc[key])}")
    return "\n".join(lines) + "\n"


def _py(v: Any) -> str:
    if isinstance(v, float) and math.isinf(v):
        return "float('inf')" if v > 0 else "float('-inf')"
    return repr(v)


def _python_reference(doc: Mapping[str, Any], arg_names: Sequence[str]) -> str | None:
    if "reference" not in doc:
        return None
    names = _reference_arg_names(doc, arg_names)
    lam = f"lambda {', '.join(names)}: {doc['reference']}"
    precision = doc.get("reference_precision", "float64")
    if precision == "longdouble":
        return f"numguard.longdouble({lam})"
    if precision == "mpmath":
        return f"numguard.highprec(mp_fn={lam})"
    return lam


def _python_kwargs(doc: Mapping[str, Any], arg_names: Sequence[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if "inputs" in doc:
        entries = doc["inputs"]
        simple = all(set(e) <= {"name", "domain"} for e in entries)
        names = _reference_arg_names(doc, arg_names)
        if simple:
            items.append(("inputs", "{" + ", ".join(f"{n!r}: {e.get('domain', 'any')!r}"
                                                    for n, e in zip(names, entries)) + "}"))
        else:
            parts = []
            for n, e in zip(names, entries):
                shape = tuple(e.get("shape", ["n"]))
                args = ", ".join(repr(d) for d in shape)
                extra = f", domain={e.get('domain', 'any')!r}"
                if e.get("dtype"):
                    extra += f", dtype={e['dtype']!r}"
                parts.append(f"{n!r}: numguard.Array({args}{extra})" if shape else
                             f"{n!r}: numguard.ArraySpec(shape=(){extra})")
            items.append(("inputs", "{" + ", ".join(parts) + "}"))
    ref = _python_reference(doc, arg_names)
    if ref:
        items.append(("reference", ref))
    for key in _DOC_ORDER:
        if key in ("inputs", "reference", "reference_precision", "reference_function", "backend") or key not in doc:
            continue
        v = doc[key]
        if key == "bounds":
            items.append((key, f"({_py(v[0])}, {_py(v[1])})"))
        elif key == "out_dtype":
            conv = (lambda x: None if x == "none" else x)
            items.append((key, repr([conv(x) for x in v] if isinstance(v, list) else conv(v))))
        else:
            items.append((key, repr(v)))
    if doc.get("backend") and doc["backend"] != "numpy":
        items.append(("backend", repr(doc["backend"])))
    return items


def _imports_for(doc: Mapping[str, Any]) -> list[str]:
    expr = doc.get("reference", "")
    out = ["import numguard"]
    if re.search(r"\bnp\.", expr):
        out.append("import numpy as np")
    if re.search(r"\bnumpy\.", expr):
        out.append("import numpy")
    if re.search(r"\bspecial\.", expr):
        out.append("import scipy.special as special")
    if re.search(r"\bscipy\.", expr):
        out.append("import scipy.special")
    if re.search(r"\bmath\.", expr):
        out.append("import math")
    if re.search(r"\bmpmath\.", expr):
        out.append("import mpmath")
    return out


def render_decorator(doc: Mapping[str, Any], arg_names: Sequence[str]) -> str:
    """A ready-to-paste ``@numguard.guard(...)`` decorator (with the imports it needs)."""
    items = _python_kwargs(doc, arg_names)
    body = "".join(f"    {k}={v},\n" for k, v in items)
    return "\n".join(_imports_for(doc)) + "\n\n" + f"@numguard.guard(\n{body})\n"


def render_python_spec(doc: Mapping[str, Any], function: str, arg_names: Sequence[str],
                       header: str = "") -> str:
    """A self-contained Python spec module with ``SPEC`` and a pytest/unittest test."""
    mod, _, attr = function.rpartition(":")
    items = _python_kwargs(doc, arg_names)
    fname = attr.split(".")[-1]
    lines = [f'"""numguard spec for {function}.', ""]
    if header:
        lines += [header, ""]
    lines += ['Review it before relying on it: run `python -m pytest <this file>`.', '"""', ""]
    lines += _imports_for(doc)
    importable = mod and not mod.endswith(".py") and "/" not in mod and "\\" not in mod
    if importable:
        lines.append(f"from {mod} import {attr.split('.')[0]}")
        call_target = attr
    else:
        lines.append("from numguard.resolve import resolve_target")
        call_target = f"resolve_target({function!r})"
    lines += ["", "SPEC = dict("]
    lines += [f"    {k}={v}," for k, v in items]
    lines += [")", "", "", f"def test_{re.sub(r'[^A-Za-z0-9_]', '_', fname)}_numguard():",
              f"    numguard.check({call_target}, **SPEC).raise_if_failed()", ""]
    return "\n".join(lines)
