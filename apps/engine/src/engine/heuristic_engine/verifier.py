"""
Heuristic Engine — verifier.py.

Wraps the standalone agentic_circuit_breaker verifier with engine settings.
"""

from __future__ import annotations

from agentic_circuit_breaker.verifier import (
    HalfOpenVerifier as BaseVerifier,
)
from agentic_circuit_breaker.verifier import (
    VerifierResult,
)

from engine.config import get_settings


class HalfOpenVerifier(BaseVerifier):
    def __init__(
        self,
        half_open_threshold: float | None = None,
        open_threshold: float | None = None,
        max_iterations: int | None = None,
    ) -> None:
        settings = get_settings()
        ho = (
            half_open_threshold
            if half_open_threshold is not None
            else settings.similarity_half_open
        )
        op = open_threshold if open_threshold is not None else settings.similarity_open
        mi = max_iterations if max_iterations is not None else settings.recursion_limit
        super().__init__(half_open_threshold=ho, open_threshold=op, max_iterations=mi)


__all__ = ["HalfOpenVerifier", "VerifierResult"]
