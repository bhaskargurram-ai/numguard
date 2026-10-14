"""``numguard.toml``: check functions from CI without decorators.

Example::

    [defaults]                        # optional, applied to every target
    dtypes = ["float32", "float64"]

    [[target]]
    function = "mypkg.ops:logsumexp"  # or "path/to/file.py:logsumexp"
    inputs = ["any"]                  # one entry per positional argument
    properties = ["translation_equivariant"]
    reference = "special.logsumexp(x)"

    [pytest]                          # optional: also run pytest (the numguard plugin is active)
    args = ["tests/test_numerics.py"]

    [agent]                           # optional: settings for `numguard fix` / the agent workflow
    enabled = false

Every ``[[target]]`` key is a spec-document key (see :mod:`numguard.specdoc`
and docs/CONFIG.md). Unknown keys are errors. API keys are never read from
this file.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from .specdoc import SpecDocError, TARGET_KEYS, validate_spec_doc

__all__ = ["CONFIG_FILENAME", "ConfigError", "Config", "PytestConfig", "AgentSettings", "load_config",
           "parse_config", "find_config"]

CONFIG_FILENAME = "numguard.toml"
MAX_TARGETS = 200
_KEYISH = re.compile(r"(api[_-]?key|token|secret|password)", re.I)


class ConfigError(ValueError):
    """``numguard.toml`` is invalid."""


@dataclass
class PytestConfig:
    args: list[str] = field(default_factory=list)
    timeout: int = 1800


@dataclass
class AgentSettings:
    enabled: bool = False
    rounds: int = 3
    max_tool_calls: int = 25
    max_targets: int = 3
    timeout: int = 120


@dataclass
class Config:
    path: Path | None
    targets: list[dict[str, Any]]
    pytest: PytestConfig | None = None
    agent: AgentSettings = field(default_factory=AgentSettings)
    #: Directory that target paths/modules are resolved against and pytest runs in
    #: (default: the configuration file's directory).
    root_override: Path | None = None

    @property
    def root(self) -> Path:
        if self.root_override is not None:
            return self.root_override
        return self.path.parent if self.path is not None else Path.cwd()

    def target(self, name: str) -> dict[str, Any]:
        for t in self.targets:
            if t["name"] == name:
                return t
        raise ConfigError(f"no target named {name!r}; targets: {', '.join(t['name'] for t in self.targets)}")


def _int(v: Any, key: str, lo: int, hi: int) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ConfigError(f"{key} must be an integer in [{lo}, {hi}]")
    return v


def _check_no_secrets(data: Any, where: str = "") -> None:
    if isinstance(data, Mapping):
        for k, v in data.items():
            if isinstance(k, str) and _KEYISH.search(k):
                raise ConfigError(f"{where}{k}: never put API keys or tokens in numguard.toml; "
                                  "use the NUMGUARD_LLM_API_KEY environment variable")
            _check_no_secrets(v, f"{where}{k}.")


def parse_config(data: Mapping[str, Any], path: Path | None = None) -> Config:
    """Validate a parsed TOML document."""
    _check_no_secrets(data)
    unknown = set(data) - {"defaults", "target", "pytest", "agent"}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown))} "
                          "(allowed: [defaults], [[target]], [pytest], [agent])")
    defaults_raw = data.get("defaults", {})
    try:
        defaults = validate_spec_doc(defaults_raw, allowed=tuple(k for k in TARGET_KEYS if k not in ("name", "function")),
                                     where="defaults")
    except SpecDocError as exc:
        raise ConfigError(str(exc)) from exc
    raw_targets = data.get("target", [])
    if not isinstance(raw_targets, list):
        raise ConfigError("target must be an array of tables: use [[target]]")
    if len(raw_targets) > MAX_TARGETS:
        raise ConfigError(f"at most {MAX_TARGETS} targets")
    targets: list[dict[str, Any]] = []
    problems: list[str] = []
    names: set[str] = set()
    for i, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            problems.append(f"target[{i}]: must be a table")
            continue
        merged = {**defaults_raw, **raw}
        try:
            doc = validate_spec_doc(merged, where=f"target[{i}]", require_function=True)
        except SpecDocError as exc:
            problems.extend(exc.problems)
            continue
        doc.setdefault("name", doc["function"].rpartition(":")[2] or doc["function"])
        if doc["name"] in names:
            problems.append(f"target[{i}].name: duplicate target name {doc['name']!r} (set name = \"...\")")
            continue
        names.add(doc["name"])
        targets.append(doc)
    if problems:
        raise ConfigError("; ".join(problems))
    py = None
    if "pytest" in data:
        raw = data["pytest"]
        if not isinstance(raw, Mapping) or set(raw) - {"args", "timeout", "enabled"}:
            raise ConfigError("[pytest] accepts args (list of strings), timeout (seconds) and enabled")
        if raw.get("enabled", True):
            args = raw.get("args", [])
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ConfigError("pytest.args must be a list of strings")
            py = PytestConfig(args=list(args), timeout=_int(raw.get("timeout", 1800), "pytest.timeout", 1, 86400))
    agent = AgentSettings()
    if "agent" in data:
        raw = data["agent"]
        allowed = {"enabled", "rounds", "max_tool_calls", "max_targets", "timeout"}
        if not isinstance(raw, Mapping) or set(raw) - allowed:
            raise ConfigError(f"[agent] accepts {', '.join(sorted(allowed))}")
        if not isinstance(raw.get("enabled", False), bool):
            raise ConfigError("agent.enabled must be true or false")
        agent = AgentSettings(
            enabled=raw.get("enabled", False),
            rounds=_int(raw.get("rounds", 3), "agent.rounds", 1, 10),
            max_tool_calls=_int(raw.get("max_tool_calls", 25), "agent.max_tool_calls", 1, 200),
            max_targets=_int(raw.get("max_targets", 3), "agent.max_targets", 1, 50),
            timeout=_int(raw.get("timeout", 120), "agent.timeout", 5, 3600),
        )
    return Config(path=path, targets=targets, pytest=py, agent=agent)


def load_config(path: str | os.PathLike[str]) -> Config:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {p}: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: invalid TOML: {exc}") from exc
    return parse_config(data, p.resolve())


def find_config(start: str | os.PathLike[str] | None = None) -> Path | None:
    """``numguard.toml`` in ``start`` (default: the current directory), if present."""
    p = Path(start or Path.cwd()) / CONFIG_FILENAME
    return p if p.is_file() else None
