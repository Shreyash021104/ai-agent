"""Reactive scripts for the MockLLM so the agent is fully demo-able without an
LLM key. `default_demo_plan` drives a small multi-tool task that also shows a
deliberate tool failure and recovery — exactly what the demo needs to prove."""
from __future__ import annotations


def default_demo_plan(messages: list[dict]) -> dict:
    # Drive off how many tool results we've produced so far.
    n = len([m for m in messages if m.get("role") == "tool"])
    if n == 0:
        # 1) use the sandboxed code tool
        return {"tool": "run_code", "args": {"code": "print(sum(i for i in range(1, 101)))"}}
    if n == 1:
        # 2) deliberately read a file that doesn't exist yet → induces a failure
        return {"tool": "read_file", "args": {"path": "summary.txt"}}
    if n == 2:
        # 3) recover from the failure by creating the file
        return {"tool": "write_file",
                "args": {"path": "summary.txt", "content": "The sum of 1..100 is 5050."}}
    # 4) finish, referencing recalled memory if any was injected into context
    recalled = any("Relevant memories" in (m.get("content") or "")
                   for m in messages if m.get("role") == "system")
    extra = " I also recalled context from an earlier session." if recalled else ""
    return {"content": "Done — I computed 1+…+100 = 5050 (via the code tool), "
                       "recovered from the missing-file error, and saved the result to "
                       f"summary.txt.{extra}"}


def circuit_breaker_plan(tool_name: str = "web_search") -> list[dict]:
    """Keeps calling the same tool, which the harness will make fail — used to
    prove the circuit breaker halts the agent."""
    return [{"tool": tool_name, "args": {"query": "x"}} for _ in range(6)]
