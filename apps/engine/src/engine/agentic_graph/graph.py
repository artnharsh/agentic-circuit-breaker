"""
LangGraph StateGraph definition — the Agentic Circuit Breaker pipeline.

Graph topology (Day 3+, with circuit breaker):
                         ┌─────────────────────┐
    START → researcher → critic                 │
                           │                    │
                      SATISFIED             RETRY
                           │                    │
                         writer          breaker_check
                           │              ↙       ↘
                          END      CLOSED/HO    OPEN
                                    ↙              ↘
                               researcher         writer
                                                   (forced summarization)
                                                    │
                                                   END

Baseline mode (no interceptor_ctx):
  - breaker_check is registered as a passthrough node → always routes to researcher
  - Behavior identical to Day 1: Critic RETRY → Researcher loop until recursion_limit crash

Day 3 mode (with interceptor_ctx):
  - breaker_check runs the full Heuristic Engine (cosine sim + FSM)
  - When OPEN: circuit_breaker_triggered=True → Writer does forced summarization
  - No crash, no wasted tokens, graceful partial answer
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from engine.agentic_graph.nodes.critic import critic_node, should_continue
from engine.agentic_graph.nodes.researcher import researcher_node
from engine.agentic_graph.nodes.writer import writer_node
from engine.agentic_graph.state import AgentState
from engine.config import get_settings

if TYPE_CHECKING:
    from engine.interceptor.hooks import InterceptorContext

logger = logging.getLogger(__name__)


def _passthrough_breaker_check(state: AgentState) -> dict:
    """
    Baseline (no interceptor) breaker check — always passes through.
    Returns no state changes, letting should_continue route to researcher.
    """
    return {}


def _passthrough_breaker_routing(state: AgentState) -> str:
    """Baseline routing: always continue to researcher (no breaker logic)."""
    return "researcher"


def build_graph(interceptor_ctx: Optional["InterceptorContext"] = None, use_circuit_breaker: bool = False):
    """
    Build and compile the pipeline StateGraph.

    Args:
        interceptor_ctx: Optional InterceptorContext. When provided:
          - All nodes are wrapped with Middleware Interceptor hooks (Day 2)
        use_circuit_breaker: When True, a real circuit breaker check node is registered (Day 3+).
          When False, a passthrough node is used (Day 1 baseline behavior).

    Returns:
        A compiled LangGraph that can be invoked with an AgentState dict.
    """
    settings = get_settings()
    recursion_limit = settings.recursion_limit

    mode = "intercepted+breaker" if interceptor_ctx else "baseline"
    logger.info(f"Building graph | mode={mode} | recursion_limit={recursion_limit}")

    # ── Optionally wrap nodes with interceptor hooks ──────────────────────────
    if interceptor_ctx is not None:
        from engine.interceptor.hooks import wrap_node
        from engine.agentic_graph.nodes.breaker_check import (
            make_breaker_check_node, breaker_routing
        )

        _researcher = wrap_node(researcher_node, "researcher", interceptor_ctx)
        _critic = wrap_node(critic_node, "critic", interceptor_ctx)
        _writer = wrap_node(writer_node, "writer", interceptor_ctx)
        
        if use_circuit_breaker:
            _breaker_check = make_breaker_check_node(interceptor_ctx)
            _breaker_routing = breaker_routing
        else:
            _breaker_check = _passthrough_breaker_check
            _breaker_routing = _passthrough_breaker_routing

    else:
        _researcher = researcher_node
        _critic = critic_node
        _writer = writer_node
        _breaker_check = _passthrough_breaker_check
        _breaker_routing = _passthrough_breaker_routing

    # ── Define the graph ──────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", _researcher)
    workflow.add_node("critic", _critic)
    workflow.add_node("breaker_check", _breaker_check)
    workflow.add_node("writer", _writer)

    # Edges
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "breaker_check": "breaker_check",
            "writer": "writer",
        },
    )

    workflow.add_conditional_edges(
        "breaker_check",
        _breaker_routing,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    workflow.add_edge("writer", END)

    compiled = workflow.compile()
    logger.info(f"Graph compiled | nodes: {list(compiled.nodes)}")
    return compiled


def get_baseline_graph():
    """Return a baseline (non-intercepted) compiled graph singleton."""
    global _baseline_graph
    if _baseline_graph is None:
        _baseline_graph = build_graph()
    return _baseline_graph


_baseline_graph = None
