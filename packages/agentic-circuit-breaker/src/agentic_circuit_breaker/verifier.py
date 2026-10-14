"""
HALF_OPEN secondary verification.

When a node's FSM enters HALF_OPEN, the verifier runs a lightweight
secondary check to add confidence about whether thrashing is real.

The verifier does NOT override the FSM — it provides a supplementary
signal for diagnostics and logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerifierResult:
    """Result of the HALF_OPEN secondary verification."""

    is_thrashing_confirmed: bool
    """True if secondary signal corroborates the similarity signal."""

    confidence: float
    """Combined confidence score in [0, 1]."""

    reason: str
    """Human-readable explanation."""

    similarity: float
    """The similarity value that triggered this check."""

    iteration: int
    """The iteration number when this check ran."""


class HalfOpenVerifier:
    """
    Lightweight secondary check run when a node's FSM is in HALF_OPEN.

    Combines cosine similarity with iteration progress and token budget
    consumption rate to assess thrashing confidence.

    Args:
        half_open_threshold: The FSM's HALF_OPEN threshold.
        open_threshold: The FSM's OPEN threshold.
        max_iterations: Expected maximum iterations (for progress calculation).
    """

    def __init__(
        self,
        half_open_threshold: float = 0.85,
        open_threshold: float = 0.95,
        max_iterations: int = 15,
    ) -> None:
        self._half_open = half_open_threshold
        self._open = open_threshold
        self._max_iterations = max_iterations

    def verify(
        self,
        *,
        similarity: float,
        iteration: int,
        total_tokens_used: int = 0,
        estimated_token_budget: int = 10_000,
    ) -> VerifierResult:
        """
        Run the secondary verification check.

        Args:
            similarity: Cosine similarity value from the state machine.
            iteration: Current call number for the node (1-indexed).
            total_tokens_used: Total tokens consumed so far in this run.
            estimated_token_budget: Estimated token limit for this run.

        Returns:
            VerifierResult with confidence score and explanation.
        """
        # Signal 1: How far through the iteration budget are we?
        recursion_progress = min(iteration / max(self._max_iterations, 1), 1.0)

        # Signal 2: Token budget consumption rate
        token_progress = min(total_tokens_used / max(estimated_token_budget, 1), 1.0)

        # Signal 3: How high is similarity above the HALF_OPEN threshold?
        threshold_range = max(self._open - self._half_open, 0.01)
        sim_excess = (similarity - self._half_open) / threshold_range
        sim_excess = min(max(sim_excess, 0.0), 1.0)

        # Weighted combination
        # Similarity is the primary signal; recursion progress is secondary
        confidence = 0.5 * sim_excess + 0.3 * recursion_progress + 0.2 * token_progress

        is_confirmed = confidence >= 0.4  # conservative threshold

        if is_confirmed:
            reason = (
                f"Thrashing likely: similarity={similarity:.3f} "
                f"(>{self._half_open}), "
                f"iteration progress={recursion_progress:.0%}, "
                f"confidence={confidence:.2f}"
            )
        else:
            reason = (
                f"Thrashing uncertain: similarity={similarity:.3f} — "
                f"giving agent another iteration "
                f"(confidence={confidence:.2f} < 0.40)"
            )

        logger.info(f"[HalfOpenVerifier] {reason}")

        return VerifierResult(
            is_thrashing_confirmed=is_confirmed,
            confidence=confidence,
            reason=reason,
            similarity=similarity,
            iteration=iteration,
        )
