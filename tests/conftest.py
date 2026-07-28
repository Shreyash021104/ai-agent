import os

# Shorten the tool timeout for the run_code timeout test. Must be set before
# app.config is imported (which reads it once).
os.environ.setdefault("TOOL_TIMEOUT_SECONDS", "3")

import pytest  # noqa: E402

from app.config import cfg  # noqa: E402
from app.db import Store  # noqa: E402


@pytest.fixture(scope="session")
def store():
    s = Store(cfg.database_url)
    s.migrate()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _clean(store):
    # Keep memory tests independent.
    with store.pool.connection() as conn:
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM runs")
    yield
