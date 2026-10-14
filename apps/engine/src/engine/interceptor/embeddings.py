"""
Middleware Interceptor — embeddings.py.

Delegates to the standalone agentic_circuit_breaker library.
"""

from __future__ import annotations

from agentic_circuit_breaker.embeddings import EmbeddingService

embedding_service = EmbeddingService()

__all__ = ["EmbeddingService", "embedding_service"]
