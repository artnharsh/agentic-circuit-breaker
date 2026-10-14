"""
Agentic Circuit Breaker.

A real-time middleware for LangGraph multi-agent pipelines that detects and
interrupts semantic thrashing loops with per-node observability.
"""

from __future__ import annotations

from .breaker import CircuitBreaker
from .config import BreakerConfig
from .embeddings import EmbeddingService
from .interceptor import NodeInterceptor, NodeTracker
from .intervention import CircuitBreakerTripped
from .similarity import compute_cosine_similarity, get_recent_similarity
from .state_machine import CircuitBreakerStateMachine
from .types import (
    BreakerState,
    NodeReport,
    RunReport,
    SimilarityEvent,
    TokenUsage,
)
from .verifier import HalfOpenVerifier, VerifierResult

__all__ = [
    "BreakerConfig",
    "BreakerState",
    "CircuitBreaker",
    "CircuitBreakerStateMachine",
    "CircuitBreakerTripped",
    "EmbeddingService",
    "HalfOpenVerifier",
    "NodeInterceptor",
    "NodeReport",
    "NodeTracker",
    "RunReport",
    "SimilarityEvent",
    "TokenUsage",
    "VerifierResult",
    "compute_cosine_similarity",
    "get_recent_similarity",
]
