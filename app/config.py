"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str = os.getenv("DATABASE_URL", "postgres://localhost:5432/agentdb")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8095"))

    # LLM: with no GROQ_API_KEY the agent runs on the deterministic MockLLM, so the
    # whole thing is demo-able and testable with zero external setup.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Agent guardrails
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "12"))
    circuit_breaker_threshold: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
    tool_timeout_seconds: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))

    # Scoped workspace for file tools + code execution.
    workspace_dir: str = os.getenv("WORKSPACE_DIR", os.path.join(os.getcwd(), "workspace"))

    # Memory retrieval
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "3"))


cfg = Config()
