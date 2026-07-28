"""Postgres-backed long-term memory + run log.

Memories are stored with their embedding as a float array; retrieval ranks by
cosine similarity. At this scale a Python-side cosine over the candidate set is
plenty; production would use pgvector's `<=>` operator + an HNSW index, which is a
drop-in change (store the embedding in a `vector` column, rank in SQL)."""
from __future__ import annotations

import json
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .embed import cosine, embed

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS memories (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session    text,
  kind       text NOT NULL DEFAULT 'fact',   -- fact | task_outcome | preference
  text       text NOT NULL,
  embedding  jsonb NOT NULL,
  metadata   jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session    text,
  task       text NOT NULL,
  status     text NOT NULL DEFAULT 'running', -- running | done | halted
  iterations int NOT NULL DEFAULT 0,
  trace      jsonb NOT NULL DEFAULT '[]',
  answer     text,
  created_at timestamptz NOT NULL DEFAULT now()
);
"""


class Store:
    def __init__(self, database_url: str):
        conninfo = database_url.replace("postgres://", "postgresql://", 1)
        self.pool = ConnectionPool(conninfo, min_size=1, max_size=8, open=True,
                                   kwargs={"row_factory": dict_row})

    def close(self) -> None:
        self.pool.close()

    def migrate(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)

    # ── Memory ─────────────────────────────────────────────────────────────
    def remember(self, text: str, *, kind: str = "fact", session: str | None = None,
                 metadata: dict | None = None) -> dict:
        with self.pool.connection() as conn:
            return conn.execute(
                """INSERT INTO memories (session, kind, text, embedding, metadata)
                   VALUES (%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING *""",
                (session, kind, text, json.dumps(embed(text)), json.dumps(metadata or {})),
            ).fetchone()

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.15) -> list[dict]:
        """Return the top_k most similar memories to `query` (cosine), above a
        floor so irrelevant memories aren't surfaced."""
        q = embed(query)
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT id, kind, text, embedding, created_at FROM memories").fetchall()
        scored = []
        for r in rows:
            score = cosine(q, r["embedding"])
            if score >= min_score:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, r in scored[:top_k]:
            out.append({"id": str(r["id"]), "kind": r["kind"], "text": r["text"],
                        "score": round(score, 3), "created_at": r["created_at"].isoformat()})
        return out

    # ── Run log ────────────────────────────────────────────────────────────
    def start_run(self, task: str, session: str | None) -> str:
        with self.pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO runs (session, task) VALUES (%s,%s) RETURNING id",
                (session, task)).fetchone()
            return str(row["id"])

    def finish_run(self, run_id: str, *, status: str, iterations: int, trace: list, answer: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE runs SET status=%s, iterations=%s, trace=%s::jsonb, answer=%s
                   WHERE id=%s""",
                (status, iterations, json.dumps(trace), answer, run_id))

    def get_run(self, run_id: str) -> Optional[dict]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM runs WHERE id=%s", (run_id,)).fetchone()
