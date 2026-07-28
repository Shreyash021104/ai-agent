"""Text embeddings for the memory store.

The default embedder is dependency-free and deterministic: it hashes character
3-grams into a fixed-dimension vector and L2-normalizes it. That keeps the whole
project offline, fast, and CI-friendly, and it's enough to demonstrate the vector
memory architecture (store → cosine top-k retrieval). Swapping in a neural
embedder (fastembed / OpenAI) is a one-line change: implement `embed()` and keep
the same dimension. Semantic quality goes up; nothing else changes.
"""
from __future__ import annotations

import math
import zlib

DIM = 256


def _stable_hash(s: str) -> int:
    # crc32 is deterministic across processes (unlike Python's str hash, which is
    # salted per-process) — essential so an embedding stored by the API matches one
    # computed in a test or another worker.
    return zlib.crc32(s.encode("utf-8"))


def _grams(text: str, n: int = 3):
    t = f"  {text.lower().strip()}  "
    for i in range(len(t) - n + 1):
        yield t[i : i + n]


def embed(text: str) -> list[float]:
    """Hash character 3-grams into a DIM-dimensional, L2-normalized vector."""
    vec = [0.0] * DIM
    for g in _grams(text):
        h = _stable_hash(g) % DIM
        vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both are unit vectors → dot = cosine
