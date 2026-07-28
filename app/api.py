"""HTTP API + chat UI. POST /run streams the agent's work as Server-Sent Events
so the UI shows each tool call and result live, not just the final answer."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .config import cfg
from .db import Store
from .llm import default_llm

PAGE = (Path(__file__).parent / "page.html").read_text()
_store: Store | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    _store = Store(cfg.database_url)
    _store.migrate()
    yield
    _store.close()


app = FastAPI(title="ai-agent", lifespan=lifespan)


class RunBody(BaseModel):
    task: str
    session: str = "default"


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.get("/api/config")
async def config():
    return {"llm": "groq" if cfg.groq_api_key else "mock",
            "max_iterations": cfg.max_iterations,
            "circuit_breaker_threshold": cfg.circuit_breaker_threshold}


@app.post("/run")
async def run(body: RunBody):
    agent = Agent(_store, default_llm())

    def stream():
        for ev in agent.run(body.task, session=body.session):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/memories")
async def memories(q: str = ""):
    if q:
        return _store.recall(q, top_k=10)
    # recent memories
    with _store.pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, kind, text, created_at FROM memories ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
    return jsonable_encoder(rows)
