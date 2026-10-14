"""
Heuristic Engine — state_machine.py.

Wraps the standalone agentic_circuit_breaker state machine with engine-level configuration.
"""

from __future__ import annotations

from agentic_circuit_breaker.state_machine import (
    CircuitBreakerStateMachine as BaseStateMachine,
)
from agentic_circuit_breaker.types import BreakerState

from engine.config import get_settings


class CircuitBreakerStateMachine(BaseStateMachine):
    """
    FSM with thresholds populated from engine settings by default.
    """

    def __init__(
        self,
        half_open_threshold: float | None = None,
        open_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        ho = (
            half_open_threshold
            if half_open_threshold is not None
            else settings.similarity_half_open
        )
        op = open_threshold if open_threshold is not None else settings.similarity_open
        super().__init__(half_open_threshold=ho, open_threshold=op)


__all__ = ["BreakerState", "CircuitBreakerStateMachine"]
