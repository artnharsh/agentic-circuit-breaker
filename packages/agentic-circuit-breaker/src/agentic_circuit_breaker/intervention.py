"""
Intervention strategies — what happens when the circuit breaker fires.

Built-in strategies:
  - "flag"  → inject circuit_breaker_triggered=True into the node result
  - "abort" → raise CircuitBreakerTripped exception

Custom:
  - Any callable(state, result, node_name) → dict
"""

from __future__ import annotations


class CircuitBreakerTripped(Exception):
    """
    Raised when on_open="abort" and the circuit breaker fires.

    Attributes:
        node_name: The node that triggered the breaker.
        similarity: The cosine similarity that caused OPEN.
        iteration: The call number when it fired.
    """

    def __init__(
        self,
        node_name: str,
        similarity: float,
        iteration: int,
    ) -> None:
        self.node_name = node_name
        self.similarity = similarity
        self.iteration = iteration
        super().__init__(
            f"Circuit breaker fired on node '{node_name}' at iteration {iteration} "
            f"(similarity={similarity:.4f}). The agent was producing repeated outputs."
        )
