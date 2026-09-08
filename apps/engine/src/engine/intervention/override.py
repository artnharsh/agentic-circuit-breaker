"""
Intervention — override.py (Day 4 implementation).

When the circuit breaker fires (OPEN state), this module injects a
forced-summarization prompt into the running pipeline, causing the Writer
node to produce a usable partial answer from whatever research was gathered,
instead of the graph crashing.

Stub for Day 1 — full implementation in Day 4.
"""

# TODO (Day 4): Implement inject_summarization_override(state: AgentState) -> AgentState
#   that modifies the graph state to:
#   1. Set circuit_breaker_triggered = True
#   2. Inject a forced-summarization system message
#   3. Call the Writer directly, bypassing further Researcher/Critic cycles
