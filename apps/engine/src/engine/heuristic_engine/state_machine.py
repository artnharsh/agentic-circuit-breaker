"""
Heuristic Engine — state_machine.py (Day 3 implementation).

The CLOSED → HALF-OPEN → OPEN finite state machine.

Thresholds (from config, defaults per synopsis):
  - CLOSED     : Sim < 0.85   (agent is making progress)
  - HALF-OPEN  : 0.85 ≤ Sim < 0.95  (suspected thrashing)
  - OPEN       : Sim ≥ 0.95   (thrashing confirmed → fire intervention)

Stub for Day 1 — full implementation in Day 3.
"""

from enum import Enum


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    HALF_OPEN = "HALF_OPEN"
    OPEN = "OPEN"


# TODO (Day 3): Implement CircuitBreakerStateMachine class with:
#   - current_state: BreakerState
#   - update(similarity: float) -> BreakerState
#   - reset() -> None
#   - Thresholds read from config
