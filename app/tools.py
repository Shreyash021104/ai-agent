"""The agent's tools. Each tool has a name, a description, and a JSON-schema for
its arguments (this is exactly what a tool-calling LLM needs), plus an `execute`
that returns a string result. Tools fail by raising ToolError; the agent decides
whether to retry, recover, or trip the circuit breaker."""
from __future__ import annotations

import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import httpx

from .config import cfg


class ToolError(Exception):
    """A tool failed. The message is fed back to the model as the tool result."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON schema
    fn: Callable[[dict], str]

    def spec(self) -> dict:
        # OpenAI/Groq tool-calling format.
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict):
    def deco(fn: Callable[[dict], str]) -> Callable[[dict], str]:
        TOOLS[name] = Tool(name, description, parameters, fn)
        return fn
    return deco


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]


# ── Workspace helpers (path-traversal safe) ────────────────────────────────
def _safe_path(rel: str) -> str:
    root = os.path.realpath(cfg.workspace_dir)
    os.makedirs(root, exist_ok=True)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise ToolError(f"path {rel!r} escapes the workspace")
    return full


# ── Tools ──────────────────────────────────────────────────────────────────
@tool("web_search", "Search the web for up-to-date information.",
      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def web_search(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        raise ToolError("query is required")
    try:
        r = httpx.get("https://api.duckduckgo.com/",
                      params={"q": query, "format": "json", "no_html": 1},
                      timeout=cfg.tool_timeout_seconds)
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"search failed: {e}")
    if data.get("AbstractText"):
        return f"{data['AbstractText']} (source: {data.get('AbstractSource','')})"
    topics = [t.get("Text", "") for t in data.get("RelatedTopics", []) if t.get("Text")]
    if topics:
        return " | ".join(topics[:3])
    return "No instant answer found."


@tool("run_code", "Execute a short Python snippet and return its stdout/stderr. Sandboxed with a timeout and memory cap.",
      {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})
def run_code(args: dict) -> str:
    code = args.get("code", "")
    if not code.strip():
        raise ToolError("code is required")

    def limits():  # child process CPU limit (best-effort sandbox; wall-clock timeout is the hard stop)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        except Exception:  # noqa: BLE001
            pass

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],  # -I: isolated mode (no env/user site)
            capture_output=True, text=True, timeout=cfg.tool_timeout_seconds,
            cwd=_safe_path("."), preexec_fn=limits if os.name != "nt" else None,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"code timed out after {cfg.tool_timeout_seconds}s")
    out = (proc.stdout or "") + (("\n[stderr] " + proc.stderr) if proc.stderr else "")
    out = out.strip()
    if proc.returncode != 0 and not out:
        raise ToolError(f"code exited with status {proc.returncode}")
    return out[:4000] or "(no output)"


@tool("write_file", "Write text to a file in the agent's workspace.",
      {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
       "required": ["path", "content"]})
def write_file(args: dict) -> str:
    path = _safe_path(args["path"])
    with open(path, "w") as f:
        f.write(args.get("content", ""))
    return f"wrote {len(args.get('content',''))} bytes to {args['path']}"


@tool("read_file", "Read a file from the agent's workspace.",
      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
def read_file(args: dict) -> str:
    path = _safe_path(args["path"])
    if not os.path.exists(path):
        raise ToolError(f"file {args['path']!r} not found")
    with open(path) as f:
        return f.read()[:4000]
