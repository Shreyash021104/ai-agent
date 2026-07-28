"""Entrypoint:

    python -m app api                 # HTTP API + chat UI
    python -m app run "your task"     # run the agent once in the terminal
    python -m app migrate             # create schema
"""
from __future__ import annotations

import sys

from .config import cfg


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "api":
        import uvicorn
        uvicorn.run("app.api:app", host=cfg.api_host, port=cfg.api_port, log_level="warning")
    elif cmd == "migrate":
        from .db import Store
        s = Store(cfg.database_url); s.migrate(); s.close()
        print("schema applied")
    elif cmd == "run":
        from .agent import Agent
        from .db import Store
        from .llm import default_llm
        task = " ".join(sys.argv[2:]) or "compute 1+...+100 and save it to a file"
        store = Store(cfg.database_url); store.migrate()
        agent = Agent(store, default_llm())
        for ev in agent.run(task):
            t = ev["type"]
            if t == "tool_call":
                print(f"  🔧 {ev['name']}({ev['args']})", flush=True)
            elif t == "tool_result":
                print(f"     → {ev['result']}", flush=True)
            elif t == "tool_error":
                print(f"     ✗ {ev['error']}", flush=True)
            elif t == "memory":
                print(f"  🧠 recalled {len(ev['recalled'])} memory(ies)", flush=True)
            elif t == "final":
                print(f"\n✅ {ev['answer']}", flush=True)
            elif t == "halted":
                print(f"\n⛔ {ev['reason']}", flush=True)
        store.close()
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
