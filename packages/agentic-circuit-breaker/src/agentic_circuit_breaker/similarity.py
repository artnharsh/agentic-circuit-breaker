"""
Cosine similarity computation for the Agentic Circuit Breaker.

Compares embedding vectors to detect semantic repetition (thrashing)
in agent outputs. Since sentence-transformers returns unit-normalized
vectors, cosine similarity reduces to a dot product.
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
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Values close to 1.0 indicate
        near-identical text (thrashing).

    Raises:
        ValueError: If either vector is empty or dimensions mismatch.
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


def get_recent_similarity(
    embeddings: list[list[float]], window: int = 2
) -> float | None:
    """
    Compute similarity between the most recent embedding and the one
    `window` steps back.

    This is the core comparison: Sim(N, N-window).
    With window=2 (default), compares the latest output to the one two steps
    back, catching both immediate repetition and gradual convergence.

    Args:
        embeddings: Ordered list of embedding vectors (oldest first).
        window: How many steps back to compare against.
                window=1 → N vs N-1 (immediate repetition)
                window=2 → N vs N-2 (default, catches gradual drift)
                window=3 → N vs N-3 (very slow-onset thrashing)

    Returns:
        Cosine similarity, or None if fewer than (window + 1) embeddings exist.
    """
    if len(embeddings) < window + 1:
        return None

    latest = embeddings[-1]
    comparison = embeddings[-(window + 1)]
    return compute_cosine_similarity(latest, comparison)
