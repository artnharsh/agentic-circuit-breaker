"""
Day 3 tests — Heuristic Engine (state machine + circuit breaker check)

Tests:
  1. Cosine similarity: correct values, edge cases
  2. State machine: correct CLOSED/HALF_OPEN/OPEN transitions
  3. Full breaker check: adversarial run triggers breaker_triggered=True
  4. Normal run: breaker_triggered stays False
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import numpy as np
import pytest

from engine.heuristic_engine.similarity import (
    compute_cosine_similarity,
    get_recent_similarity,
)
from engine.heuristic_engine.state_machine import BreakerState, CircuitBreakerStateMachine


# ── Cosine similarity tests ────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_give_similarity_1(self):
        v = [1.0, 0.0, 0.0, 0.5]
        sim = compute_cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors_give_similarity_0(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = compute_cosine_similarity(v1, v2)
        assert abs(sim - 0.0) < 1e-6

    def test_opposite_vectors_give_similarity_minus_1(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        sim = compute_cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_0(self):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        sim = compute_cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_raises_on_empty_vector(self):
        with pytest.raises(ValueError):
            compute_cosine_similarity([], [1.0])

    def test_raises_on_dimension_mismatch(self):
        with pytest.raises(ValueError):
            compute_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_get_recent_similarity_returns_none_if_not_enough(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]  # only 2, need 3 for window=1
        result = get_recent_similarity(embeddings, window=2)
        assert result is None

    def test_get_recent_similarity_correct_window(self):
        """Window=1: compare embeddings[-1] vs embeddings[-2]."""
        v1 = [1.0, 0.0]
        v2 = [0.9, 0.1]  # close to v1
        v3 = [0.0, 1.0]  # orthogonal
        embeddings = [v1, v2, v3]
        # get_recent_similarity with window=1: v3 vs v2
        sim = get_recent_similarity(embeddings, window=1)
        assert sim is not None
        # v3=[0,1] vs v2=[0.9, 0.1]: should be low
        expected = float(np.dot(
            np.array(v3) / np.linalg.norm(v3),
            np.array(v2) / np.linalg.norm(v2),
        ))
        assert abs(sim - expected) < 1e-5


# ── State machine tests ────────────────────────────────────────────────────────

class TestCircuitBreakerStateMachine:
    def test_starts_closed(self):
        sm = CircuitBreakerStateMachine()
        assert sm.current_state == BreakerState.CLOSED

    def test_low_similarity_stays_closed(self):
        sm = CircuitBreakerStateMachine()
        state = sm.update(0.3)
        assert state == BreakerState.CLOSED

    def test_medium_similarity_enters_half_open(self):
        sm = CircuitBreakerStateMachine()
        state = sm.update(0.87)  # ≥ 0.85, < 0.95
        assert state == BreakerState.HALF_OPEN

    def test_high_similarity_opens_directly(self):
        sm = CircuitBreakerStateMachine()
        state = sm.update(0.97)  # ≥ 0.95
        assert state == BreakerState.OPEN

    def test_half_open_then_low_returns_to_closed(self):
        sm = CircuitBreakerStateMachine()
        sm.update(0.87)  # → HALF_OPEN
        state = sm.update(0.3)   # drops back
        assert state == BreakerState.CLOSED

    def test_half_open_then_medium_escalates_to_open(self):
        sm = CircuitBreakerStateMachine()
        sm.update(0.87)  # → HALF_OPEN
        state = sm.update(0.88)  # still ≥ 0.85 → OPEN
        assert state == BreakerState.OPEN

    def test_very_high_similarity_opens_from_closed(self):
        sm = CircuitBreakerStateMachine()
        state = sm.update(0.9913)  # like the mock
        assert state == BreakerState.OPEN

    def test_history_tracks_transitions(self):
        sm = CircuitBreakerStateMachine()
        sm.update(0.87)
        sm.update(0.88)
        assert len(sm.history) == 2
        assert sm.history[0][1] == BreakerState.HALF_OPEN
        assert sm.history[1][1] == BreakerState.OPEN

    def test_reset_returns_to_closed(self):
        sm = CircuitBreakerStateMachine()
        sm.update(0.97)  # OPEN
        sm.reset()
        assert sm.current_state == BreakerState.CLOSED


# ── End-to-end breaker integration test ──────────────────────────────────────

@pytest.mark.asyncio
async def test_adversarial_query_triggers_circuit_breaker():
    """
    The CORE research contribution test:
    Adversarial query → circuit breaker fires → graceful partial answer
    (NOT a GraphRecursionError crash).
    """
    from engine.storage.db import init_db, get_session_factory
    from engine.interceptor.hooks import InterceptorContext, flush_pending_logs
    from engine.interceptor.log_store import LogStore
    from engine.agentic_graph.graph import build_graph
    from engine.agentic_graph.state import AgentState

    await init_db()

    corpus_dir = Path(__file__).parents[3] / "data" / "corpus"
    corpus_docs: list[str] = []
    if corpus_dir.exists():
        corpus_docs = [f.read_text() for f in sorted(corpus_dir.glob("*.md"))]

    run_id = str(uuid.uuid4())
    factory = get_session_factory()

    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query="adversarial")
        ctx = InterceptorContext(run_id=run_id)
        graph = build_graph(interceptor_ctx=ctx)

        state: AgentState = {
            "query": (
                "Provide the exact GDP impact coefficient for every country "
                "in the world for every agricultural subsidy at monthly granularity."
            ),
            "corpus_docs": corpus_docs or ["No relevant data available."],
            "research_notes": "", "critique": "", "draft": "", "final_output": "",
            "iteration_count": 0, "circuit_breaker_triggered": False, "messages": [],
        }

        result = graph.invoke(state, config={"recursion_limit": 10})
        await flush_pending_logs(ctx, db)
        await db.commit()

    # The circuit breaker MUST have fired
    assert result["circuit_breaker_triggered"] is True, (
        "Expected circuit_breaker_triggered=True for an adversarial query. "
        f"FSM history: {ctx.state_machine.history}"
    )

    # A partial answer MUST be produced (no silent crash)
    assert result.get("final_output"), "Expected a partial answer even when breaker fires"

    # Must have stopped well before recursion_limit (saves iterations)
    assert result.get("iteration_count", 10) < 6, (
        "Breaker should fire before hitting recursion_limit=6"
    )


@pytest.mark.asyncio
async def test_normal_query_does_not_trigger_breaker():
    """Normal queries must complete without triggering the circuit breaker."""
    from engine.storage.db import init_db, get_session_factory
    from engine.interceptor.hooks import InterceptorContext, flush_pending_logs
    from engine.interceptor.log_store import LogStore
    from engine.agentic_graph.graph import build_graph
    from engine.agentic_graph.state import AgentState

    await init_db()

    corpus_dir = Path(__file__).parents[3] / "data" / "corpus"
    corpus_docs: list[str] = []
    if corpus_dir.exists():
        corpus_docs = [f.read_text() for f in sorted(corpus_dir.glob("*.md"))]

    run_id = str(uuid.uuid4())
    factory = get_session_factory()

    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query="normal")
        ctx = InterceptorContext(run_id=run_id)
        graph = build_graph(interceptor_ctx=ctx)

        state: AgentState = {
            "query": "What is solar energy and how do photovoltaic panels work?",
            "corpus_docs": corpus_docs or [
                "Solar energy is generated by photovoltaic panels that convert sunlight."
            ],
            "research_notes": "", "critique": "", "draft": "", "final_output": "",
            "iteration_count": 0, "circuit_breaker_triggered": False, "messages": [],
        }

        result = graph.invoke(state, config={"recursion_limit": 10})
        await flush_pending_logs(ctx, db)
        await db.commit()

    assert result["circuit_breaker_triggered"] is False, (
        "Normal query must NOT trigger the circuit breaker"
    )
    assert result.get("final_output"), "Normal query must produce a final answer"
