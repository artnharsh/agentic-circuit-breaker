"""
CLOSED → HALF_OPEN → OPEN finite state machine.

Decoupled from engine.config — thresholds are passed as constructor arguments,
making this usable in any context without pydantic-settings.

State transitions:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Sim(N, N−W) < half_open_threshold     → stay / return CLOSED       │
  │  half_open_threshold ≤ Sim < open       → HALF_OPEN                 │
  │  Sim ≥ open_threshold                   → OPEN (breaker fires)      │
  └──────────────────────────────────────────────────────────────────────┘

HALF_OPEN exists to prevent false positives: a single high-similarity step
might be a coincidence. HALF_OPEN requires consistent high similarity before
triggering OPEN.
"""

from __future__ import annotations

import logging

from .types import BreakerState

logger = logging.getLogger(__name__)


class CircuitBreakerStateMachine:
    """
    Per-node finite state machine that transitions between
    CLOSED, HALF_OPEN, and OPEN based on cosine similarity.

    Each watched node in a pipeline gets its own FSM instance.

    Args:
        half_open_threshold: Similarity >= this enters HALF_OPEN.
        open_threshold: Similarity >= this triggers OPEN.
    """

    def __init__(
        self,
        half_open_threshold: float = 0.85,
        open_threshold: float = 0.95,
    ) -> None:
        self._half_open_threshold = half_open_threshold
        self._open_threshold = open_threshold
        self._state: BreakerState = BreakerState.CLOSED
        self._history: list[tuple[float, BreakerState]] = []

    @property
    def current_state(self) -> BreakerState:
        """Current FSM state."""
        return self._state

    @property
    def history(self) -> list[tuple[float, BreakerState]]:
        """List of (similarity, resulting_state) pairs."""
        return list(self._history)

    @property
    def half_open_threshold(self) -> float:
        return self._half_open_threshold

    @property
    def open_threshold(self) -> float:
        return self._open_threshold

    def update(self, similarity: float) -> BreakerState:
        """
        Process a new similarity value and transition the state machine.

        Args:
            similarity: Cosine similarity between current and N-window-back
                        embeddings (range [-1, 1]).

        Returns:
            The new BreakerState after processing.
        """
        prev_state = self._state

        if similarity >= self._open_threshold:
            # High similarity → OPEN regardless of previous state
            self._state = BreakerState.OPEN

        elif similarity >= self._half_open_threshold:
            if self._state == BreakerState.CLOSED:
                # First sign of thrashing → HALF_OPEN (one warning)
                self._state = BreakerState.HALF_OPEN
            elif self._state == BreakerState.HALF_OPEN:
                # Consistent high similarity → escalate to OPEN
                self._state = BreakerState.OPEN
            # If already OPEN, stay OPEN

        else:
            # Low similarity → agent is making progress
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.CLOSED

        self._history.append((similarity, self._state))

        if self._state != prev_state:
            logger.info(
                f"[StateMachine] {prev_state.value} → {self._state.value} "
                f"| similarity={similarity:.4f}"
            )
        else:
            logger.debug(
                f"[StateMachine] {self._state.value} (unchanged) "
                f"| similarity={similarity:.4f}"
            )

        return self._state

    def reset(self) -> None:
        """Reset to CLOSED state and clear history."""
        self._state = BreakerState.CLOSED
        self._history.clear()
        logger.debug("[StateMachine] Reset to CLOSED.")

    def __repr__(self) -> str:
        return (
            f"CircuitBreakerStateMachine("
            f"state={self._state.value}, "
            f"half_open={self._half_open_threshold}, "
            f"open={self._open_threshold}, "
            f"steps={len(self._history)})"
        )
