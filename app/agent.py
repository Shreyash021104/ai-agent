"""The agent loop — orchestration written by hand, not by a framework.

Each turn: build context (task + history + retrieved memories) → ask the LLM,
offering the tools → if it proposes a tool call, execute it (sandboxed,
timeout-bounded) and feed the result back → repeat until the model gives a final
answer, or a guardrail stops it:
  • max_iterations   — don't loop forever
  • circuit breaker  — if the SAME tool fails N times in a row, halt and report
                       partial progress instead of thrashing
At the end the outcome is written to long-term memory for future runs to recall.
"""
from __future__ import annotations

import json
from typing import Iterator

from .config import cfg
from .db import Store
from .llm import LLM, Assistant
from .tools import TOOLS, ToolError, tool_specs

SYSTEM_PROMPT = (
    "You are a capable task-completing agent. Use the provided tools to accomplish "
    "the user's task. Think step by step, call one tool at a time, and when the task "
    "is complete, reply with a final answer and no tool call."
)


def _assistant_msg(a: Assistant) -> dict:
    return {
        "role": "assistant",
        "content": a.content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in a.tool_calls
        ] or None,
    }


class Agent:
    def __init__(self, store: Store, llm: LLM):
        self.store = store
        self.llm = llm

    def run(self, task: str, session: str | None = None) -> Iterator[dict]:
        """Run the agent, yielding events as it works (for streaming UIs)."""
        yield {"type": "start", "task": task}

        # 1) Long-term memory retrieval
        memories = self.store.recall(task, top_k=cfg.memory_top_k)
        if memories:
            yield {"type": "memory", "recalled": memories}

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memories:
            mem_text = "\n".join(f"- {m['text']}" for m in memories)
            messages.append({"role": "system", "content": f"Relevant memories from past runs:\n{mem_text}"})
        messages.append({"role": "user", "content": task})

        last_failed_tool: str | None = None
        consecutive_failures = 0
        answer = ""
        status = "halted"

        for iteration in range(1, cfg.max_iterations + 1):
            assistant = self.llm.chat(messages, tool_specs())

            if not assistant.tool_calls:
                answer = assistant.content or ""
                status = "done"
                yield {"type": "final", "answer": answer, "iterations": iteration}
                break

            messages.append(_assistant_msg(assistant))

            halt = False
            for tc in assistant.tool_calls:
                yield {"type": "tool_call", "name": tc.name, "args": tc.arguments, "iteration": iteration}
                tool = TOOLS.get(tc.name)
                if tool is None:
                    result, ok = f"error: unknown tool {tc.name!r}", False
                else:
                    try:
                        result, ok = tool.fn(tc.arguments), True
                    except ToolError as e:
                        result, ok = f"error: {e}", False
                    except Exception as e:  # noqa: BLE001 - never let a tool crash the loop
                        result, ok = f"error: {type(e).__name__}: {e}", False

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

                if ok:
                    yield {"type": "tool_result", "name": tc.name, "result": str(result)[:500]}
                    consecutive_failures, last_failed_tool = 0, None
                else:
                    yield {"type": "tool_error", "name": tc.name, "error": str(result)[:500]}
                    if tc.name == last_failed_tool:
                        consecutive_failures += 1
                    else:
                        last_failed_tool, consecutive_failures = tc.name, 1
                    if consecutive_failures >= cfg.circuit_breaker_threshold:
                        answer = (f"Halted: tool {tc.name!r} failed "
                                  f"{consecutive_failures} times in a row (circuit breaker).")
                        status = "halted"
                        yield {"type": "halted", "reason": answer}
                        halt = True
                        break
            if halt:
                break
        else:
            answer = f"Halted: reached the {cfg.max_iterations}-iteration limit."
            status = "halted"
            yield {"type": "halted", "reason": answer}

        # 2) Persist the outcome for future recall.
        self.store.remember(f"Task: {task}\nOutcome: {answer}",
                            kind="task_outcome", session=session)
        yield {"type": "done", "status": status}

    def run_sync(self, task: str, session: str | None = None) -> dict:
        """Collect the whole run into a trace + final answer (for tests/API)."""
        trace, answer, status = [], "", "halted"
        for ev in self.run(task, session):
            trace.append(ev)
            if ev["type"] == "final":
                answer, status = ev["answer"], "done"
            elif ev["type"] == "halted":
                answer, status = ev["reason"], "halted"
        return {"trace": trace, "answer": answer, "status": status,
                "iterations": sum(1 for e in trace if e["type"] == "tool_call")}
