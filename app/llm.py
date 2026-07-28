"""LLM abstraction with two implementations:

- MockLLM   — deterministic, scripted. No network, no key. Powers the tests and a
              keyless demo, and makes the agent loop reproducible.
- GroqLLM   — real tool-calling via Groq's OpenAI-compatible API (set GROQ_API_KEY).

Both return the same `Assistant` shape, so the agent loop is identical either way.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from .config import cfg


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Assistant:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM:
    def chat(self, messages: list[dict], tools: list[dict]) -> Assistant:  # pragma: no cover
        raise NotImplementedError


# ── Mock ────────────────────────────────────────────────────────────────────
class MockLLM(LLM):
    """`plan` is either a list of step dicts returned in order, or a callable
    (messages) -> step dict for reactive behavior. A step is one of:
        {"tool": "name", "args": {...}}   → propose a tool call
        {"content": "final answer"}        → finish
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> Assistant:
        step = self.plan(messages) if callable(self.plan) else (
            self.plan[self.i] if self.i < len(self.plan) else {"content": "Done."})
        self.i += 1
        if "tool" in step:
            return Assistant(tool_calls=[ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}", name=step["tool"], arguments=step.get("args", {}))])
        return Assistant(content=step.get("content", "Done."))


# ── Groq (real) ─────────────────────────────────────────────────────────────
class GroqLLM(LLM):
    def __init__(self):
        self.client = httpx.Client(
            base_url=cfg.groq_base_url,
            headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
            timeout=60,
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> Assistant:
        resp = self.client.post("/chat/completions", json={
            "model": cfg.groq_model, "messages": messages,
            "tools": tools, "tool_choice": "auto", "temperature": 0.2,
        })
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        return Assistant(content=msg.get("content"), tool_calls=calls)


def default_llm() -> LLM:
    """Real Groq if a key is set, otherwise a mock that does a small canned task —
    so `python -m app run "..."` works with zero configuration."""
    if cfg.groq_api_key:
        return GroqLLM()
    from .mock_plans import default_demo_plan
    return MockLLM(default_demo_plan)
