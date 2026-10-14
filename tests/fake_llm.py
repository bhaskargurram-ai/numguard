"""A scripted LLM transport (no network): each request is answered with the next scripted step."""

from __future__ import annotations

import itertools
import json

from numguard.httpclient import HTTPResponse

_ids = itertools.count(1)


def anthropic_reply(calls, text="", usage=(120, 30), stop=None):
    content = [{"type": "text", "text": text}] if text else []
    for name, args in calls:
        content.append({"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": name, "input": args})
    return {"id": "msg", "type": "message", "role": "assistant", "content": content,
            "stop_reason": stop or ("tool_use" if calls else "end_turn"),
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "cache_read_input_tokens": 10}}


def openai_reply(calls, text="", usage=(120, 30), stop=None):
    msg = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{next(_ids)}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": stop or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


class ScriptedTransport:
    """Answers each request with the next scripted step.

    A step is a list of ``(tool_name, arguments)`` pairs (one assistant turn
    with those tool calls), a string (a text-only turn), an ``HTTPResponse``
    (returned as is, e.g. a 429), or a callable ``step(payload) -> step``
    that can look at the request (e.g. to answer with what the model was shown).
    """

    def __init__(self, provider, steps):
        self.provider = provider
        self.steps = list(steps)
        self.requests = []

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body)
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "payload": payload})
        if not self.steps:
            raise AssertionError("the scripted LLM ran out of steps")
        step = self.steps.pop(0)
        if callable(step) and not isinstance(step, HTTPResponse):
            step = step(payload)
        if isinstance(step, HTTPResponse):
            return step
        if isinstance(step, str):
            calls, text = [], step
        else:
            calls, text = step, ""
        reply = anthropic_reply(calls, text) if self.provider == "anthropic" else openai_reply(calls, text)
        return HTTPResponse(200, json.dumps(reply).encode())

    def request_text(self, i):
        """All text of request ``i`` (for asserting what the model was shown)."""
        return json.dumps(self.requests[i]["payload"])

    def last_tool_result(self, i):
        """Text of the tool results sent with request ``i`` (Anthropic format)."""
        msgs = self.requests[i]["payload"]["messages"]
        return "\n".join(str(b.get("content", "")) for b in msgs[-1]["content"] if b.get("type") == "tool_result")
