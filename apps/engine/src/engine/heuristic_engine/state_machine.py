"""
Heuristic Engine — state_machine.py

The CLOSED → HALF-OPEN → OPEN finite state machine.

State transitions (thresholds from config/.env):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Sim(N, N−2) < HALF_OPEN_THRESHOLD (0.85)  → stay / return CLOSED   │
  │  HALF_OPEN_THRESHOLD ≤ Sim < OPEN_THRESHOLD → HALF_OPEN             │
  │  Sim ≥ OPEN_THRESHOLD (0.95)               → OPEN (breaker fires)  │
  └──────────────────────────────────────────────────────────────────────┘

CLOSED:
  Normal operation. Agent is making progress.
  On each Researcher output: compute Sim, check thresholds.

HALF_OPEN:
  Suspected thrashing. Similarity is high but below the OPEN threshold.
  The breaker does NOT fire yet — it gives the agent one more chance.
  If the next similarity is also high → escalate to OPEN.
  If the next similarity drops below HALF_OPEN threshold → return to CLOSED.

OPEN:
  Thrashing confirmed. Breaker fires — pipeline is redirected to Writer
  with a forced-summarization prompt instead of another Researcher loop.
  After firing, the state machine resets to CLOSED (ready for the next run).

Why HALF_OPEN?
  A single high-similarity step might be a coincidence (the agent found the
  same relevant paragraph on two sequential searches). HALF_OPEN prevents
  false positives by requiring consistent high similarity before triggering.
"""

from __future__ import annotations

import logging
from enum import Enum

from engine.config import get_settings

logger = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    HALF_OPEN = "HALF_OPEN"
    OPEN = "OPEN"


class CircuitBreakerStateMachine:
    """
    Finite state machine that transitions between CLOSED, HALF_OPEN, and OPEN
    based on the cosine similarity between consecutive Researcher embeddings.

    One instance per pipeline run. Created by InterceptorContext.

    Usage:
        sm = CircuitBreakerStateMachine()
        state = sm.update(similarity=0.97)  # → BreakerState.OPEN
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._half_open_threshold: float = settings.similarity_half_open  # 0.85
        self._open_threshold: float = settings.similarity_open             # 0.95
        self._state: BreakerState = BreakerState.CLOSED
        self._history: list[tuple[float, BreakerState]] = []
        logger.debug(
            f"[StateMachine] Initialized | "
            f"HALF_OPEN≥{self._half_open_threshold} | OPEN≥{self._open_threshold}"
        )

    @property
    def current_state(self) -> BreakerState:
        return self._state

    @property
    def history(self) -> list[tuple[float, BreakerState]]:
        """List of (similarity, resulting_state) pairs for this run."""
        return list(self._history)

    def update(self, similarity: float) -> BreakerState:
        """
        Process a new similarity value and transition the state machine.

        Args:
            similarity: Cosine similarity between the current Researcher
                        embedding and the one two steps back (range [−1, 1]).

        Returns:
            The new BreakerState after processing this similarity value.
        """
        prev_state = self._state

        if similarity >= self._open_threshold:
            # High similarity → OPEN regardless of previous state
            self._state = BreakerState.OPEN

        elif similarity >= self._half_open_threshold:
            if self._state == BreakerState.CLOSED:
                # First sign of thrashing → enter HALF_OPEN (one warning)
                self._state = BreakerState.HALF_OPEN
            elif self._state == BreakerState.HALF_OPEN:
                # Still high → escalate to OPEN
                self._state = BreakerState.OPEN
            # If already OPEN, stay OPEN (handled by caller)

        else:
            # Low similarity → agent is making progress → return to CLOSED
            if self._state in (BreakerState.HALF_OPEN,):
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
        """Reset to CLOSED state (called after the breaker fires)."""
        self._state = BreakerState.CLOSED
        logger.debug("[StateMachine] Reset to CLOSED.")

    def __repr__(self) -> str:
        return (
            f"CircuitBreakerStateMachine("
            f"state={self._state.value}, "
            f"steps={len(self._history)})"
        )
