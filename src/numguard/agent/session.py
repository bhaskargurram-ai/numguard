"""A tool-use conversation with one schema-validated submit tool.

The model may call read-only tools any number of times (up to a budget) and
ends by calling the submit tool. A submission is passed to a validator; if
it is rejected, the reasons go back to the model as the tool result and it
may try again. Tool outputs are wrapped in per-session random delimiters and
the system prompt tells the model that delimited text is data, not
instructions.
"""

from __future__ import annotations

import json
import secrets as _secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLMToolError, ToolLLM, ToolResult, ToolSpec, Turn, Usage

__all__ = ["Tool", "ToolError", "SessionResult", "Session", "wrap_untrusted", "new_nonce", "MAX_TOOL_OUTPUT"]

MAX_TOOL_OUTPUT = 12_000

Validator = Callable[[dict[str, Any]], "tuple[Any, list[str]]"]


class ToolError(Exception):
    """A tool call failed in an expected way (bad path, bad pattern...); shown to the model."""


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], str]
    #: Wrap the output as untrusted data (everything read from the repository or a report).
    untrusted: bool = True


@dataclass
class SessionResult:
    status: str  # submitted | no_submission | error
    value: Any = None
    error: str = ""
    rejections: list[list[str]] = field(default_factory=list)


def new_nonce() -> str:
    return _secrets.token_hex(6)


def wrap_untrusted(text: str, nonce: str, label: str = "data") -> str:
    """Delimit untrusted text; occurrences of the nonce inside it are neutralized."""
    body = str(text).replace(nonce, "[nonce]")
    if len(body) > MAX_TOOL_OUTPUT:
        body = body[:MAX_TOOL_OUTPUT] + f"\n... ({len(body) - MAX_TOOL_OUTPUT} more characters truncated)"
    return f"<untrusted-{label} id={nonce}>\n{body}\n</untrusted-{label} id={nonce}>"


class Session:
    """One conversation. Call :meth:`run` with the first user message, then
    :meth:`followup` for refinement rounds (the conversation continues)."""

    def __init__(self, llm: ToolLLM, system: str, tools: Sequence[Tool], submit: ToolSpec, validate: Validator,
                 *, max_tool_calls: int = 25, max_requests: int = 40, nonce: str | None = None,
                 log: Callable[[str], None] = lambda s: None):
        self.llm = llm
        self.system = system
        self.tools = {t.spec.name: t for t in tools}
        self.submit = submit
        self.validate = validate
        self.max_tool_calls = max_tool_calls
        self.max_requests = max_requests
        self.nonce = nonce or new_nonce()
        self.log = log
        self.turns: list[Turn] = []
        self.usage = Usage()
        self.tool_calls = 0
        self.requests = 0
        self.transcript: list[dict[str, Any]] = []

    def _specs(self) -> list[ToolSpec]:
        return [t.spec for t in self.tools.values()] + [self.submit]

    def run(self, user_text: str) -> SessionResult:
        self.turns.append(Turn("user", text=user_text))
        return self._loop()

    def followup(self, user_text: str) -> SessionResult:
        return self.run(user_text)

    def _call_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        tool = self.tools[name]
        try:
            out = tool.handler(args)
        except ToolError as exc:
            return f"error: {exc}", True
        except (TypeError, ValueError, KeyError) as exc:
            return f"error: invalid arguments ({type(exc).__name__}: {exc})", True
        out = str(out)
        return (wrap_untrusted(out, self.nonce) if tool.untrusted else out), False

    def _loop(self) -> SessionResult:
        nudges = 0
        rejections: list[list[str]] = []
        start_requests = self.requests
        while self.requests - start_requests < self.max_requests:
            force = self.submit.name if self.tool_calls >= self.max_tool_calls else None
            try:
                reply = self.llm.chat(self.system, self.turns, self._specs(), force_tool=force)
            except LLMToolError as exc:
                return SessionResult("error", error=str(exc), rejections=rejections)
            self.requests += 1
            self.usage.add(reply.usage)
            self.turns.append(Turn("assistant", text=reply.text, tool_calls=list(reply.tool_calls)))
            if not reply.tool_calls:
                nudges += 1
                self.transcript.append({"assistant_text": reply.text[:2000]})
                if nudges > 2:
                    return SessionResult("no_submission", error="the model stopped without calling "
                                         f"{self.submit.name}", rejections=rejections)
                self.turns.append(Turn("user", text=f"Use the tools if you need more information, then call "
                                                    f"{self.submit.name}. Do not answer in plain text."))
                continue
            results: list[ToolResult] = []
            accepted: Any = None
            got_submission = False
            for call in reply.tool_calls:
                self.tool_calls += 1
                entry: dict[str, Any] = {"tool": call.name, "arguments": _short_json(call.arguments)}
                if call.parse_error:
                    results.append(ToolResult(call.id, call.name, f"error: {call.parse_error}", True))
                    entry["error"] = call.parse_error
                elif call.name == self.submit.name:
                    if got_submission:
                        results.append(ToolResult(call.id, call.name, "ignored: only one submission per turn", True))
                        continue
                    got_submission = True
                    value, problems = self.validate(call.arguments)
                    if problems:
                        rejections.append(problems)
                        entry["rejected"] = problems
                        text = "REJECTED. Fix these problems and call {} again:\n- {}".format(
                            self.submit.name, "\n- ".join(problems))
                        results.append(ToolResult(call.id, call.name, text, True))
                    else:
                        accepted = value
                        entry["accepted"] = True
                        results.append(ToolResult(call.id, call.name, "accepted"))
                elif call.name in self.tools:
                    out, is_err = self._call_tool(call.name, call.arguments)
                    results.append(ToolResult(call.id, call.name, out, is_err))
                else:
                    results.append(ToolResult(call.id, call.name, f"error: unknown tool {call.name!r}", True))
                    entry["error"] = "unknown tool"
                self.transcript.append(entry)
            self.turns.append(Turn("tool", results=results))
            if accepted is not None:
                return SessionResult("submitted", value=accepted, rejections=rejections)
        return SessionResult("no_submission", error="request budget exhausted", rejections=rejections)


def _short_json(v: Any, limit: int = 400) -> str:
    s = json.dumps(v, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"
