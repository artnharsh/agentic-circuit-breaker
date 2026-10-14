"""
Heuristic Engine — similarity.py.

Delegates to the standalone agentic_circuit_breaker library.
"""

from __future__ import annotations

from agentic_circuit_breaker.similarity import (
    compute_cosine_similarity,
    get_recent_similarity,
)

__all__ = ["compute_cosine_similarity", "get_recent_similarity"]
