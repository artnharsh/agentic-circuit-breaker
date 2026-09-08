"""
Middleware Interceptor — embeddings.py (Day 2 implementation).

Wraps sentence-transformers to produce a vector for each reasoning attempt.
These vectors are what the heuristic engine uses for cosine similarity.

Stub for Day 1 — full implementation in Day 2.
"""

# TODO (Day 2): Implement EmbeddingService class that:
#   1. Loads all-MiniLM-L6-v2 via sentence-transformers
#   2. Exposes embed(text: str) -> list[float]
#   3. Caches the model as a singleton (slow to load)
