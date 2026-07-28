"""Long-term memory: store, then retrieve by semantic-ish similarity (cosine)."""
from app.embed import cosine, embed


def test_embedding_is_deterministic_and_normalized():
    a, b = embed("the capital of France is Paris"), embed("the capital of France is Paris")
    assert a == b                      # deterministic across calls
    assert abs(cosine(a, a) - 1.0) < 1e-9  # unit vector


def test_similar_text_scores_higher_than_unrelated():
    q = embed("how do I reset my password")
    close = embed("resetting your password")
    far = embed("bananas are yellow")
    assert cosine(q, close) > cosine(q, far)


def test_recall_returns_relevant_memory(store):
    store.remember("The deploy runbook lives in docs/deploy.md", kind="fact")
    store.remember("The user prefers dark mode", kind="preference")
    hits = store.recall("where is the deployment runbook", top_k=1)
    assert hits
    assert "deploy" in hits[0]["text"].lower()


def test_unrelated_query_recalls_nothing(store):
    store.remember("The deploy runbook lives in docs/deploy.md", kind="fact")
    assert store.recall("quantum chromodynamics lecture notes", top_k=3) == []
