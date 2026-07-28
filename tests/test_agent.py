"""The agent loop: completes a multi-tool task (including recovering from a
deliberate tool failure), and the circuit breaker halts repeated failures."""
from app.agent import Agent
from app.llm import MockLLM
from app.mock_plans import default_demo_plan


def test_agent_completes_task_and_recovers_from_failure(store):
    agent = Agent(store, MockLLM(default_demo_plan))
    result = agent.run_sync("compute 1..100 and save it", session="s1")

    assert result["status"] == "done"
    types = [e["type"] for e in result["trace"]]
    # It used the code tool, hit a tool error, then recovered — all present.
    assert "tool_result" in types
    assert "tool_error" in types      # the deliberate missing-file failure
    assert types[-1] == "done"
    assert "5050" in result["answer"]

    # The outcome was written to long-term memory.
    recalled = store.recall("what did I compute", top_k=5)
    assert any("5050" in m["text"] or "compute" in m["text"].lower() for m in recalled)


def test_circuit_breaker_halts_repeated_failures(store):
    # Keep calling a tool that always fails (read a missing file).
    plan = [{"tool": "read_file", "args": {"path": "nope-missing.txt"}} for _ in range(6)]
    agent = Agent(store, MockLLM(plan))
    result = agent.run_sync("loop on a failing tool")

    assert result["status"] == "halted"
    assert "circuit breaker" in result["answer"].lower()
    # It stopped at the threshold (3), not after all 6 scripted calls.
    assert sum(1 for e in result["trace"] if e["type"] == "tool_call") == 3


def test_unknown_tool_is_handled_not_crashing(store):
    agent = Agent(store, MockLLM([{"tool": "no_such_tool", "args": {}}, {"content": "done"}]))
    result = agent.run_sync("call a bogus tool")
    # A bogus tool is an error fed back, not a crash; the agent still finishes.
    assert result["status"] == "done"
    assert any(e["type"] == "tool_error" for e in result["trace"])
