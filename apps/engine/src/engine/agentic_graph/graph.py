"""
LangGraph StateGraph definition — the Agentic Circuit Breaker pipeline.

Graph topology:
    START → researcher → critic ──SATISFIED──→ writer → END
                           │
                         RETRY
                           │
                           └──────────────→ researcher (loop)

The graph is compiled with `recursion_limit` from config (default: 6).
When an adversarial query causes the Researcher→Critic loop to exceed this
limit, LangGraph raises a `GraphRecursionError` — this is the baseline
crash behavior that the circuit breaker (Day 3+) will prevent.

Day 2 addition: `build_graph(interceptor_ctx)` accepts an optional
InterceptorContext. When provided, all nodes are wrapped with the
Middleware Interceptor hooks before being registered in the graph.

Usage:
    # Without interceptor (Day 1 baseline)
    graph = build_graph()

    # With interceptor (Day 2+)
    from engine.interceptor.hooks import InterceptorContext
    ctx = InterceptorContext(log_store=log_store)
    graph = build_graph(interceptor_ctx=ctx)
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


def build_graph(interceptor_ctx: Optional["InterceptorContext"] = None):
    """
    Build and compile the Researcher → Critic → Writer LangGraph pipeline.

    Args:
        interceptor_ctx: Optional InterceptorContext. When provided, all nodes
            are wrapped with the Middleware Interceptor hooks (embedding, token
            counting, SQLite logging). When None, nodes run unwrapped (baseline).

    Returns:
        A compiled LangGraph that can be invoked with an AgentState dict.
    """
    settings = get_settings()
    recursion_limit = settings.recursion_limit

    mode = "intercepted" if interceptor_ctx else "baseline"
    logger.info(f"Building graph | mode={mode} | recursion_limit={recursion_limit}")

    # ── Optionally wrap nodes with interceptor hooks ───────────────────────
    if interceptor_ctx is not None:
        from engine.interceptor.hooks import wrap_node
        _researcher = wrap_node(researcher_node, "researcher", interceptor_ctx)
        _critic = wrap_node(critic_node, "critic", interceptor_ctx)
        _writer = wrap_node(writer_node, "writer", interceptor_ctx)
    else:
        _researcher = researcher_node
        _critic = critic_node
        _writer = writer_node

    # ── Define the graph ───────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", _researcher)
    workflow.add_node("critic", _critic)
    workflow.add_node("writer", _writer)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    workflow.add_edge("writer", END)

    compiled = workflow.compile()
    logger.info(f"Graph compiled | nodes: {list(compiled.nodes)}")
    return compiled


def get_graph():
    """Return a baseline (non-intercepted) compiled graph singleton."""
    global _baseline_graph
    if _baseline_graph is None:
        _baseline_graph = build_graph()
    return _baseline_graph


_baseline_graph = None
