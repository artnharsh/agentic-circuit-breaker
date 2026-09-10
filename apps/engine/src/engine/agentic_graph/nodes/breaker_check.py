"""
Circuit Breaker Check Node — breaker_check.py

This node sits between the Critic (RETRY verdict) and the Researcher.
It runs the Heuristic Engine to decide whether to:
  - Let the pipeline continue (CLOSED/HALF_OPEN) → route to Researcher
  - Fire the circuit breaker (OPEN) → set circuit_breaker_triggered=True
    and route to Writer for forced summarization

Graph position:
    critic ──RETRY──→ breaker_check ──CLOSED/HALF_OPEN──→ researcher
                           │
                          OPEN
                           │
                           └──→ writer (forced summarization)

This node is only registered when build_graph() receives an InterceptorContext.
In baseline mode (no ctx), the graph routes directly critic → researcher,
preserving Day 1 behavior.

The node uses a closure over InterceptorContext to access:
  - ctx.pending_logs  → recent Researcher embeddings
  - ctx.state_machine → the CLOSED/HALF_OPEN/OPEN FSM
  - ctx.token_accumulator → token budget progress for the verifier
"""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING

from engine.agentic_graph.state import AgentState
from engine.heuristic_engine.similarity import get_recent_similarity
from engine.heuristic_engine.state_machine import BreakerState, CircuitBreakerStateMachine
from engine.heuristic_engine.verifier import HalfOpenVerifier

if TYPE_CHECKING:
    from engine.interceptor.hooks import InterceptorContext

logger = logging.getLogger(__name__)

_verifier = HalfOpenVerifier()


def make_breaker_check_node(ctx: "InterceptorContext") -> Callable[[AgentState], dict]:
    """
    Factory that creates the breaker_check node, closed over InterceptorContext.

    Using a factory (not a class) keeps the LangGraph node signature clean:
    the returned function takes only AgentState and returns dict — exactly
    what LangGraph expects.

    Args:
        ctx: The shared InterceptorContext for this run. Contains:
             - pending_logs: accumulated Researcher embeddings
             - state_machine: the per-run CLOSED/HALF_OPEN/OPEN FSM
             - token_accumulator: running token totals

    Returns:
        A LangGraph-compatible node function.
    """

    def breaker_check_node(state: AgentState) -> dict:
        """
        Evaluate the heuristic engine and decide whether to fire the breaker.

        Returns a state patch with:
          - circuit_breaker_triggered=True  → Writer will do forced summarization
          - circuit_breaker_triggered=False → Researcher runs next as normal
        """
        iteration = state.get("iteration_count", 0)

        # ── Extract Researcher embeddings from in-memory pending_logs ──────
        researcher_embeddings: list[list[float]] = [
            log.embedding
            for log in ctx.pending_logs
            if log.node == "researcher" and log.embedding is not None
        ]

        if len(researcher_embeddings) < 2:
            # Not enough data for comparison yet → stay CLOSED, let pipeline continue
            logger.debug(
                f"[BreakerCheck] Iteration {iteration} | "
                f"Only {len(researcher_embeddings)} embedding(s) — need ≥2, staying CLOSED"
            )
            return {
                "circuit_breaker_triggered": False,
                "iteration_count": iteration,
            }

        # ── Compute cosine similarity (N vs N-2) ───────────────────────────
        similarity = get_recent_similarity(researcher_embeddings, window=1)
        # window=1 means latest vs second-latest (effectively N vs N-1 here
        # since we accumulate per loop iteration, not per node call)

        if similarity is None:
            return {"circuit_breaker_triggered": False, "iteration_count": iteration}

        logger.info(
            f"[BreakerCheck] Iteration {iteration} | "
            f"Similarity: {similarity:.4f} | "
            f"Embeddings compared: {len(researcher_embeddings)}"
        )

        # ── Run state machine ──────────────────────────────────────────────
        new_breaker_state = ctx.state_machine.update(similarity)

        # ── HALF_OPEN: run secondary verifier ──────────────────────────────
        if new_breaker_state == BreakerState.HALF_OPEN:
            verifier_result = _verifier.verify(
                similarity=similarity,
                iteration=iteration,
                total_tokens_used=ctx.token_accumulator.total_tokens,
            )
            # The verifier result is logged above; it doesn't override the FSM
            # in Day 3 (the FSM decides on the next similarity reading).
            # From Day 4, verifier_result can trigger early escalation.

        # ── OPEN: fire the circuit breaker ────────────────────────────────
        if new_breaker_state == BreakerState.OPEN:
            logger.warning(
                f"[BreakerCheck] 🔥 CIRCUIT BREAKER OPEN | "
                f"Iteration {iteration} | Similarity {similarity:.4f} ≥ "
                f"{ctx.state_machine._open_threshold} | "
                f"Redirecting to forced summarization."
            )
            # Update the breaker_state in the most recent pending_log
            if ctx.pending_logs:
                ctx.pending_logs[-1].breaker_state = BreakerState.OPEN.value

            return {
                "circuit_breaker_triggered": True,
                "iteration_count": iteration,
            }

        # ── CLOSED / HALF_OPEN: continue normally ─────────────────────────
        if ctx.pending_logs:
            ctx.pending_logs[-1].breaker_state = new_breaker_state.value

        return {
            "circuit_breaker_triggered": False,
            "iteration_count": iteration,
        }

    breaker_check_node.__name__ = "breaker_check"
    return breaker_check_node


def breaker_routing(state: AgentState) -> str:
    """
    LangGraph conditional edge: routes after breaker_check_node.

    Returns:
        "writer"     → circuit breaker fired; forced summarization
        "researcher" → breaker did not fire; pipeline continues normally
    """
    if state.get("circuit_breaker_triggered", False):
        return "writer"
    return "researcher"
