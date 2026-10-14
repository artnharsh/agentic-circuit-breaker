"""
Unit and integration tests for the agentic-circuit-breaker library.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from agentic_circuit_breaker import (
    BreakerConfig,
    BreakerState,
    CircuitBreaker,
    CircuitBreakerStateMachine,
    CircuitBreakerTripped,
    compute_cosine_similarity,
    get_recent_similarity,
)

# ── Cosine Similarity Tests ───────────────────────────────────────────────────


class TestSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        sim = compute_cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = compute_cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_empty_vectors_raise(self):
        with pytest.raises(ValueError, match="empty vectors"):
            compute_cosine_similarity([], [1.0])

    def test_mismatched_dimensions_raise(self):
        with pytest.raises(ValueError, match="mismatch"):
            compute_cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_vector_returns_zero(self):
        assert compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_get_recent_similarity_window(self):
        embeddings = [
            [1.0, 0.0],  # 0
            [0.0, 1.0],  # 1
            [1.0, 0.0],  # 2 (same as 0)
        ]
        # window = 1: compares idx 2 and idx 1 -> [1, 0] vs [0, 1] -> 0.0
        assert abs(get_recent_similarity(embeddings, window=1) - 0.0) < 1e-6

        # window = 2: compares idx 2 and idx 0 -> [1, 0] vs [1, 0] -> 1.0
        assert abs(get_recent_similarity(embeddings, window=2) - 1.0) < 1e-6

        # window = 3: not enough embeddings
        assert get_recent_similarity(embeddings, window=3) is None


# ── State Machine Tests ───────────────────────────────────────────────────────


class TestStateMachine:
    def test_state_machine_transitions(self):
        fsm = CircuitBreakerStateMachine(half_open_threshold=0.85, open_threshold=0.95)
        assert fsm.current_state == BreakerState.CLOSED

        # Low similarity stays CLOSED
        assert fsm.update(0.50) == BreakerState.CLOSED

        # Moderate-high similarity enters HALF_OPEN
        assert fsm.update(0.88) == BreakerState.HALF_OPEN

        # High similarity escalates from HALF_OPEN to OPEN
        assert fsm.update(0.89) == BreakerState.OPEN

        # Direct jump to OPEN on very high similarity
        fsm.reset()
        assert fsm.current_state == BreakerState.CLOSED
        assert fsm.update(0.97) == BreakerState.OPEN

    def test_half_open_recovery(self):
        fsm = CircuitBreakerStateMachine(half_open_threshold=0.85, open_threshold=0.95)
        fsm.update(0.88)
        assert fsm.current_state == BreakerState.HALF_OPEN

        # Agent diversifies output -> recovers to CLOSED
        fsm.update(0.30)
        assert fsm.current_state == BreakerState.CLOSED


# ── Config Validation Tests ───────────────────────────────────────────────────


class TestBreakerConfig:
    def test_valid_config(self):
        cfg = BreakerConfig(half_open_threshold=0.80, open_threshold=0.90)
        cfg.validate()

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            BreakerConfig(half_open_threshold=0.95, open_threshold=0.85).validate()

        with pytest.raises(ValueError):
            BreakerConfig(half_open_threshold=-0.1).validate()

    def test_invalid_intervention(self):
        with pytest.raises(ValueError):
            BreakerConfig(on_open="invalid_strategy").validate()


# ── CircuitBreaker Execution & Per-Node Observability Tests ────────────────────


class TestCircuitBreakerObservability:
    def test_per_node_thrashing_localization(self):
        """
        Simulate a multi-agent system where:
          - researcher produces repeated outputs (thrashing)
          - critic produces varied outputs (not thrashing)
        Verify that the breaker identifies 'researcher' as the culprit.
        """

        # Custom deterministic embedder for fast, reproducible testing
        def mock_embed(text: str) -> list[float]:
            if "repetitive research" in text:
                return [1.0, 0.0]  # identical vector every time
            elif "critique_1" in text:
                return [0.0, 1.0]
            elif "critique_2" in text:
                return [0.0, -1.0]
            elif "critique_3" in text:
                return [0.707, 0.707]
            return [0.5, 0.5]

        cb = CircuitBreaker(
            half_open_threshold=0.85,
            open_threshold=0.95,
            similarity_window=1,  # immediate N vs N-1 comparison
            embed_fn=mock_embed,
            on_open="flag",
        )

        researcher_calls = 0
        critic_calls = 0

        def researcher_node(state: dict) -> dict:
            nonlocal researcher_calls
            researcher_calls += 1
            return {
                "research_notes": f"repetitive research {1}",  # same text
                "call_count": researcher_calls,
            }

        def critic_node(state: dict) -> dict:
            nonlocal critic_calls
            critic_calls += 1
            return {
                "critique": f"critique_{critic_calls}",  # varying text
            }

        watched_researcher = cb.watch(researcher_node, name="researcher")
        watched_critic = cb.watch(critic_node, name="critic")

        state = {}
        # Iteration 1
        res1 = watched_researcher(state)
        crit1 = watched_critic(res1)
        assert not cb.is_triggered

        # Iteration 2
        res2 = watched_researcher(crit1)
        _ = watched_critic(res2)
        # researcher produced identical output twice -> sim=1.0 -> OPEN
        assert cb.is_triggered
        assert cb.triggered_by == "researcher"

        # Report checks
        report = cb.report()
        assert report.breaker_triggered is True
        assert report.triggered_by == "researcher"
        assert "researcher" in report.thrashing_nodes
        assert "critic" not in report.thrashing_nodes
        assert report.node_reports["researcher"].is_thrashing is True
        assert report.node_reports["critic"].is_thrashing is False

        # Summary output checks
        summary_text = report.summary()
        assert "TRIGGERED at iteration" in summary_text
        assert "researcher" in summary_text
        assert "THRASHING" in summary_text


# ── LangGraph Integration Tests ───────────────────────────────────────────────


class AgentState(TypedDict):
    notes: str
    feedback: str
    circuit_breaker_triggered: bool
    circuit_breaker_source: str


class TestLangGraphIntegration:
    def test_wrap_langgraph_workflow(self):
        """Test wrapping a LangGraph StateGraph and intercepting thrashing."""

        def mock_embed(text: str) -> list[float]:
            if "stuck" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

        cb = CircuitBreaker(
            half_open_threshold=0.85,
            open_threshold=0.95,
            similarity_window=1,
            embed_fn=mock_embed,
            on_open="flag",
        )

        iter_counter = {"researcher": 0, "critic": 0}

        def researcher(state: AgentState) -> dict:
            iter_counter["researcher"] += 1
            return {"notes": "stuck research"}

        def critic(state: AgentState) -> dict:
            iter_counter["critic"] += 1
            return {"feedback": f"feedback_{iter_counter['critic']}"}

        workflow = StateGraph(AgentState)
        workflow.add_node("researcher", researcher)
        workflow.add_node("critic", critic)
        workflow.add_edge(START, "researcher")
        workflow.add_edge("researcher", "critic")

        def route_critic(state: AgentState) -> str:
            if state.get("circuit_breaker_triggered"):
                return END
            if iter_counter["researcher"] >= 3:
                return END
            return "researcher"

        workflow.add_conditional_edges(
            "critic", route_critic, {"researcher": "researcher", END: END}
        )

        compiled_graph = workflow.compile()
        protected_graph = cb.wrap(compiled_graph)

        result = protected_graph.invoke(
            {
                "notes": "",
                "feedback": "",
                "circuit_breaker_triggered": False,
                "circuit_breaker_source": "",
            }
        )

        assert result.get("circuit_breaker_triggered") is True
        assert result.get("circuit_breaker_source") == "researcher"
        assert cb.triggered_by == "researcher"

    def test_abort_strategy(self):
        """Test on_open='abort' raises CircuitBreakerTripped."""
        cb = CircuitBreaker(
            open_threshold=0.90,
            similarity_window=1,
            embed_fn=lambda t: [1.0, 0.0],
            on_open="abort",
        )

        def node(s: dict) -> dict:
            return {"output": "loop"}

        watched = cb.watch(node, name="agent_x")
        watched({})  # 1st call
        with pytest.raises(CircuitBreakerTripped) as exc_info:
            watched({})  # 2nd call -> sim=1.0 >= 0.90 -> raises

        assert exc_info.value.node_name == "agent_x"
