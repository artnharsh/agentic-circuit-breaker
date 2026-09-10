"""
Heuristic Engine — similarity.py

Computes cosine similarity between two embedding vectors.

Formula (from the B.Tech synopsis):
    Sim(N, N−2) = (e_N · e_(N−2)) / (||e_N|| × ||e_(N−2)||)

Since sentence-transformers returns unit-normalized vectors (L2 norm = 1),
the cosine similarity reduces to a simple dot product — fast and exact.

Why N vs N-2 (not N vs N-1)?
  Comparing consecutive iterations (N vs N-1) may miss slow-onset thrashing
  where the agent gradually converges on the same failure. Comparing across
  two steps (N vs N-2) gives the breaker a "window" to catch gradual drift,
  while still reacting quickly enough to interrupt before hitting recursion_limit.
"""

from __future__ import annotations

import numpy as np


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Both vectors should be unit-normalized (which sentence-transformers
    guarantees when normalize_embeddings=True). In that case this is
    equivalent to a dot product and runs in O(dim) time.

    Args:
        vec_a: First embedding vector (list of floats, dim=384).
        vec_b: Second embedding vector (list of floats, dim=384).

    Returns:
        Cosine similarity in [−1.0, 1.0]. Values close to 1.0 indicate
        the agent is producing nearly identical reasoning (thrashing).

    Raises:
        ValueError: If either vector is zero-length or has mismatched dim.
    """
    if not vec_a or not vec_b:
        raise ValueError("Cannot compute similarity of empty vectors.")
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}."
        )

    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        # Zero vector → undefined similarity; return 0 (treat as divergent)
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def get_recent_similarity(embeddings: list[list[float]], window: int = 2) -> float | None:
    """
    Compute similarity between the most recent embedding and the one
    `window` steps back. Returns None if not enough embeddings yet.

    Args:
        embeddings: Ordered list of embedding vectors (oldest first).
        window: How many steps back to compare against (default: 2 = N vs N-2).

    Returns:
        Cosine similarity, or None if fewer than window+1 embeddings exist.
    """
    if len(embeddings) < window + 1:
        return None

    latest = embeddings[-1]
    comparison = embeddings[-(window + 1)]
    return compute_cosine_similarity(latest, comparison)
