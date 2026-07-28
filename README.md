# Loop — AI Agent with Tool-Calling & Memory

A task-completing AI agent with a **hand-written agent loop** (no LangChain): it calls real
tools — sandboxed code execution, scoped file I/O, web search — maintains **long-term memory**
across sessions, and **recovers from tool failures** with a circuit breaker and iteration
limits. Runs **keyless** on a deterministic mock LLM, or on Groq with a free API key.

<p align="center">
  <img src="docs/demo.gif" alt="The agent planning a task, calling tools one at a time, recovering from a failed tool call, and recalling a memory from a previous run" width="100%" />
  <br>
  <em>The agent working a task tool-by-tool, recovering from a deliberate tool failure, then recalling a memory from an earlier run. (<a href="docs/demo.mp4">full-quality video</a>)</em>
</p>

## The problem this is built around

The gap between "I've called an LLM API" and "I understand agents" is the gap that gets you
past the résumé screen for AI roles. This project closes it by building the agent loop *by
hand*, so the interesting, production-shaping problems are visible instead of hidden inside a
framework:

1. **The agent loop itself.** The model doesn't "do" anything — it proposes a **tool call**;
   *you* execute it and feed the result back; repeat until it produces a final answer. That
   propose → execute → observe cycle is the entire ballgame, and it's ~40 lines here.
2. **Tool-calling protocol.** Tools are described to the model as JSON-schema functions; the
   model returns a structured call; you dispatch it. Getting the message format right
   (assistant tool-call messages, `tool` result messages) is what makes multi-turn tool use
   actually work.
3. **Failure handling — the part that separates a toy from a real agent.** What happens when a
   tool errors, or the model loops calling the same failing tool forever? Here: the error is
   fed back so the model can adapt, a **circuit breaker** halts the agent if one tool fails N
   times in a row, and a hard **iteration cap** bounds every run. It reports partial progress
   instead of thrashing or burning tokens.
4. **Memory.** Short-term memory is the running message history; **long-term memory** is a
   vector store — each run's outcome is embedded and saved, and future runs retrieve the most
   relevant past memories into their context.

## Runs with zero setup: keyless mock mode

With no `GROQ_API_KEY`, the agent runs on a **deterministic mock LLM** that drives a small
multi-tool task (compute with the code tool → hit a missing-file error → recover → save the
result). This makes the whole system — loop, tools, guardrails, memory — testable in CI and
demo-able on any machine with **no API key and no network**. Set a free Groq key and the exact
same loop runs a real tool-calling model.

## Architecture

```
  User task ─► Agent loop
                 │  1. build context: task + running history + retrieved memories
                 │◄──────────────────────────────────── long-term memory (Postgres, cosine top-k)
                 ▼
             2. LLM.chat(messages, tools)   ── mock or Groq, same interface
                 │
          proposes a tool call? ──► 3. execute tool (sandboxed, timeout-bounded)
                 │                         success → feed result back, loop
                 │                         failure → feed error back
                 │                                    same tool fails N× → circuit breaker halts
                 ▼
            final answer (or halted: iteration cap / breaker)
                 │
                 └─► 4. store the outcome in long-term memory for next time
```

## The tools

| Tool | What it does | Safety |
|---|---|---|
| `run_code` | Executes a Python snippet, returns stdout/stderr | Subprocess in isolated mode, wall-clock **timeout**, CPU rlimit |
| `write_file` / `read_file` | Read/write text in the agent's workspace | Paths resolved and **confined to the workspace** (traversal blocked) |
| `web_search` | Looks up info via DuckDuckGo (no key) | Timeout-bounded; graceful failure |

## The hardest decisions

### 1. Write the loop, don't import it
The loop is `app/agent.py` — build context, call the model with tool specs, dispatch any tool
call, append the result, repeat. Doing it by hand means the failure paths are *mine* to design
(below) rather than a framework's to hide. When a framework breaks in production, this is the
understanding that lets you fix it.

### 2. The circuit breaker (the "senior" bit)
An agent that calls a failing tool, sees the error, and calls it again — forever — is the most
common way agents burn money. The breaker tracks **consecutive failures of the same tool**;
at the threshold (default 3) it halts and returns partial progress. Combined with the
iteration cap, no run can loop or thrash without bound. `test_agent.py` proves it stops at 3,
not at the 6 calls the script requested.

### 3. Tool failures are data, not exceptions
A tool raising `ToolError` isn't a crash — the error string is fed back to the model as the
tool result, so the model can *adapt* (try a different path, a different tool). The demo shows
this: the agent reads a file that doesn't exist, gets the error, and recovers by writing it.

### 4. Memory as retrieval, not a giant prompt
You can't stuff every past interaction into the context window. So outcomes are **embedded and
stored**, and each new task retrieves only the top-k most similar memories (cosine) above a
relevance floor. The embedder is dependency-free and deterministic by default (hashed
character n-grams); swapping in a neural embedder (fastembed/OpenAI) or pgvector's `<=>` +
HNSW index is a localized change — the architecture is the point.

## Verifying it yourself

Tests run on the **mock LLM** (deterministic) against a real Postgres — no key, no network:

```bash
python scripts/migrate.py
python -m pytest -q
```

They cover: the agent **completing a multi-tool task and recovering** from a deliberate tool
failure; the **circuit breaker** halting repeated same-tool failures at the threshold; an
unknown tool being handled as an error (not a crash); **sandboxed code execution** with a
timeout; **path-traversal** being blocked in the file tools; and memory **store → similarity
retrieval** (relevant memory found, irrelevant query returns nothing).

## Running locally

Requires Python 3.11+ and PostgreSQL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/migrate.py
python -m app api                       # chat UI at http://localhost:8095
# or run once in the terminal:
python -m app run "Compute the sum of 1..100 and save it to a file"
```

For a real model, set a free key from [console.groq.com](https://console.groq.com):
```bash
export GROQ_API_KEY=gsk_...
```

## What I'd change at 10× scale

- **A neural embedder + pgvector** (`<=>` operator, HNSW index) for semantic memory at scale;
  the embedder is already abstracted behind one function.
- **Stronger sandboxing** for `run_code` — a container or gVisor/Firecracker microVM rather
  than a subprocess, with no network and a read-only FS.
- **Parallel tool calls** — the loop currently executes tool calls sequentially; models can
  request several at once.
- **Streaming tokens** from the model (not just events) for a snappier UI.
- **Memory hygiene** — dedupe/summarize memories and add recency/importance weighting so the
  store doesn't grow unbounded or drown relevant facts.

## License

MIT
