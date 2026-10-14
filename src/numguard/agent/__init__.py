"""The optional numguard agent (experimental; bring your own LLM key).

An LLM *proposes* -- a spec (``numguard suggest``), a rewrite of a failing
function (``numguard fix``), benchmark solutions (``numguard bench``) -- and
numguard *verifies by executing*: model-written code only runs in a
subprocess worker with a timeout and a scrubbed environment after a static
denylist check, and only the executed, re-checked outcome counts. The model
never decides a verdict. See docs/AGENT.md and docs/SECURITY_MODEL.md.
"""
