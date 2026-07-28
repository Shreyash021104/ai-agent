# Contributing

Thanks for your interest in improving Loop!

## Getting set up

1. Install Python 3.11+ and PostgreSQL.
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt pytest`
4. `python scripts/migrate.py`
5. `python -m app api` and open http://localhost:8095 (runs keyless on the mock LLM)

## Before opening a pull request

Run the full test suite — this is exactly what CI runs (it uses the mock LLM, so no key is
needed, but it does need a real Postgres):

```bash
python -m pytest -q
```

If you change the agent loop (`app/agent.py`) or add a tool, add a test. The failure-handling
guarantees — circuit breaker, tool errors fed back as data, sandbox limits — are the point of
this project, so they must stay covered.

## Guidelines

- New tools go in `app/tools.py` via `@tool(name, description, json_schema)` and must fail by
  raising `ToolError` (never by crashing the loop). File/exec tools must stay confined to the
  workspace.
- Keep the `LLM` interface (`chat(messages, tools) -> Assistant`) stable so the mock and Groq
  implementations remain interchangeable.
- Don't reach for a framework for the core loop — the hand-written loop is the whole point.

By contributing, you agree that your contributions will be licensed under the MIT License.
