"""Create the agentdb database (if missing) and apply the schema. Idempotent."""
from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import cfg  # noqa: E402
from app.db import Store  # noqa: E402


def main() -> None:
    url = urlparse(cfg.database_url.replace("postgres://", "postgresql://", 1))
    dbname = url.path.lstrip("/") or "agentdb"
    admin = url._replace(path="/postgres").geturl()
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,)).fetchone():
            conn.execute(f'CREATE DATABASE "{dbname}"')
            print(f"created database {dbname}")
    store = Store(cfg.database_url)
    store.migrate()
    store.close()
    print("schema applied")


if __name__ == "__main__":
    main()
