"""
Heuristic Engine — verifier.py

Secondary verification step used in the HALF_OPEN state.

When the state machine enters HALF_OPEN (similarity ≥ 0.85 but < 0.95),
the verifier performs a quick secondary check to provide additional signal
about whether the agent is truly stuck.

Secondary signal: token budget consumption rate.
  If the agent has already spent >60% of its token budget (estimated as
  iterations_completed / recursion_limit) and similarity is still high,
  we increase confidence that it is thrashing.

This verifier does NOT override the state machine — it only adds a logging
annotation. The state machine itself drives all OPEN/CLOSED transitions.
From Day 4 onward, the verifier's output can influence intervention strategy
(e.g., partial answer vs. full abort).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from engine.config import get_settings

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
    Lightweight secondary check run when the state machine is in HALF_OPEN.

    Combines the cosine similarity with a token budget consumption rate
    to decide whether to confirm thrashing or give the agent another chance.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

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
            iteration: Current iteration number (zero-indexed).
            total_tokens_used: Total tokens consumed so far in this run.
            estimated_token_budget: Estimated token limit for this run.

        Returns:
            VerifierResult with confidence score and human-readable reason.
        """
        settings = get_settings()
        recursion_limit = settings.recursion_limit

        # Signal 1: How far through the recursion budget are we?
        recursion_progress = min(iteration / max(recursion_limit, 1), 1.0)

        # Signal 2: Token budget consumption rate
        token_progress = min(
            total_tokens_used / max(estimated_token_budget, 1), 1.0
        )

        # Signal 3: How high is the similarity above the HALF_OPEN threshold?
        sim_excess = (similarity - settings.similarity_half_open) / max(
            settings.similarity_open - settings.similarity_half_open, 0.01
        )
        sim_excess = min(max(sim_excess, 0.0), 1.0)

        # Weighted combination
        # Similarity is the primary signal; recursion progress is secondary
        confidence = 0.5 * sim_excess + 0.3 * recursion_progress + 0.2 * token_progress

        is_confirmed = confidence >= 0.4  # conservative threshold

        if is_confirmed:
            reason = (
                f"Thrashing likely: similarity={similarity:.3f} "
                f"(>{settings.similarity_half_open}), "
                f"recursion progress={recursion_progress:.0%}, "
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
