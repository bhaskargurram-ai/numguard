"""pytest plugin for numguard (registered through the ``pytest11`` entry point).

Two ways to use it:

1. **Marker** on an ordinary test. The test body runs first; afterwards the
   plugin runs :func:`numguard.check` with the marker's arguments and fails
   the test with the report table if any check failed::

       @pytest.mark.numguard(fn=my_softmax, reference=scipy.special.softmax,
                             batched=True)
       def test_softmax_edge_cases():
           pass

   The function may also be given positionally: ``@pytest.mark.numguard(my_softmax)``.

2. **Auto-collection** of functions decorated with :func:`numguard.guard`
   from importable modules::

       pytest --numguard-modules mypkg.ops,mypkg.stats

   Each guarded function becomes a test item named ``numguard::<module>::<function>``.

Command-line options ``--numguard-dtypes float16,float32`` and
``--numguard-large-n 10000`` override the dtypes and large-``n`` size for
every numguard check in the session.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Iterator

import pytest

from .api import check, check_spec, get_spec
from .report import NumGuardError, Report

__all__ = [
    "NumguardItem",
    "pytest_addoption",
    "pytest_configure",
    "pytest_pyfunc_call",
    "pytest_collection_modifyitems",
]

MARKER = "numguard"
_MARKER_HELP = (
    "numguard(fn=None, **options): after the test body, run numguard.check(fn, **options) "
    "and fail with its report table if any edge-case check fails"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("numguard", "numerical edge-case checks")
    group.addoption(
        "--numguard-modules", action="append", default=[], metavar="MODULES",
        help="Comma-separated importable modules whose @numguard.guard functions are "
             "collected as tests (repeatable).",
    )
    group.addoption(
        "--numguard-dtypes", action="store", default=None, metavar="DTYPES",
        help="Comma-separated float dtypes for all numguard checks, e.g. float16,float32.",
    )
    group.addoption(
        "--numguard-large-n", action="store", type=int, default=None, metavar="N",
        help="Size of the large_n edge case for all numguard checks.",
    )
    parser.addini("numguard_modules", type="args", default=[],
                  help="Modules whose @numguard.guard functions are collected as tests.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", _MARKER_HELP)


def _overrides(config: pytest.Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dtypes = config.getoption("numguard_dtypes", default=None)
    if dtypes:
        out["dtypes"] = tuple(d.strip() for d in dtypes.split(",") if d.strip())
    large_n = config.getoption("numguard_large_n", default=None)
    if large_n is not None:
        out["large_n"] = int(large_n)
    return out


class _NumguardFailure(Exception):
    """Internal: carries a failed report out of an item's ``runtest``."""

    def __init__(self, report: Report):
        super().__init__(report.summary())
        self.report = report


# --------------------------------------------------------------------------
# Marker
# --------------------------------------------------------------------------


def _run_marker(item: pytest.Function, marker: pytest.Mark) -> None:
    kwargs = dict(marker.kwargs)
    fn: Callable[..., Any] | None = kwargs.pop("fn", None)
    args = list(marker.args)
    if fn is None and args:
        fn = args.pop(0)
    if fn is None or not callable(fn):
        raise pytest.UsageError(
            f"{item.nodeid}: @pytest.mark.numguard needs a function (fn=... or first argument)"
        )
    if args:
        raise pytest.UsageError(f"{item.nodeid}: unexpected positional marker args {args!r}")
    kwargs.update(_overrides(item.config))
    spec = get_spec(fn)
    name = getattr(fn, "__qualname__", repr(fn))
    if spec is not None and not {"inputs", "domain"} & set(kwargs):
        report = check(fn, spec=spec, name=name, **kwargs)
    else:
        report = check(fn, name=name, **kwargs)
    if not report.ok:
        pytest.fail(report.summary(), pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> Iterator[None]:
    result = yield
    marker = pyfuncitem.get_closest_marker(MARKER)
    if marker is not None:
        _run_marker(pyfuncitem, marker)
    return result


# --------------------------------------------------------------------------
# Auto-collection of @numguard.guard functions
# --------------------------------------------------------------------------


class NumguardItem(pytest.Item):
    """A test item that checks one ``@numguard.guard``-decorated function."""

    def __init__(self, *, fn: Callable[..., Any], module_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.fn = fn
        self.module_name = module_name
        self.add_marker(MARKER)

    def runtest(self) -> None:
        spec = get_spec(self.fn)
        assert spec is not None
        report = check_spec(self.fn, spec, name=f"{self.module_name}.{self.fn.__qualname__}",
                            **_overrides(self.config))
        if not report.ok:
            raise _NumguardFailure(report)

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException],
                     style: Any = None) -> Any:
        if isinstance(excinfo.value, (_NumguardFailure, NumGuardError)):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style=style)

    def reportinfo(self) -> tuple[Any, int | None, str]:
        try:
            path = inspect.getsourcefile(self.fn) or self.module_name
            line = inspect.getsourcelines(self.fn)[1]
        except (OSError, TypeError):
            path, line = self.module_name, None
        return path, line, f"numguard: {self.module_name}.{self.fn.__qualname__}"


class NumguardModule(pytest.Collector):
    """Collector grouping the guarded functions of one importable module."""

    def __init__(self, *, module_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.module_name = module_name

    def collect(self) -> list[pytest.Item]:
        module = importlib.import_module(self.module_name)
        items: list[pytest.Item] = []
        for attr, obj in vars(module).items():
            if callable(obj) and get_spec(obj) is not None:
                if getattr(obj, "__module__", self.module_name) != self.module_name:
                    continue  # imported from elsewhere; collected in its own module
                items.append(NumguardItem.from_parent(
                    self, name=attr, fn=obj, module_name=self.module_name))
        return items


def _module_names(config: pytest.Config) -> list[str]:
    names: list[str] = []
    for entry in list(config.getoption("numguard_modules", default=[]) or []) + list(
            config.getini("numguard_modules") or []):
        names.extend(m.strip() for m in str(entry).split(",") if m.strip())
    return list(dict.fromkeys(names))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    for module_name in _module_names(config):
        collector = NumguardModule.from_parent(session, name=f"numguard::{module_name}",
                                               module_name=module_name)
        try:
            new = collector.collect()
        except ImportError as exc:
            raise pytest.UsageError(
                f"--numguard-modules: cannot import {module_name!r}: {exc}") from exc
        items.extend(new)
