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

Usage:
    from engine.agentic_graph.graph import build_graph
    graph = build_graph()
    result = graph.invoke(initial_state)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from engine.agentic_graph.nodes.critic import critic_node, should_continue
from engine.agentic_graph.nodes.researcher import researcher_node
from engine.agentic_graph.nodes.writer import writer_node
from engine.agentic_graph.state import AgentState
from engine.config import get_settings

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build and compile the Researcher → Critic → Writer LangGraph pipeline.

    Returns a compiled LangGraph that can be invoked with an AgentState dict.
    The recursion_limit is read from config (RECURSION_LIMIT env var, default 6).
    """
    settings = get_settings()
    recursion_limit = settings.recursion_limit

    logger.info(f"Building graph with recursion_limit={recursion_limit}")

    # ── Define the graph ───────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    # Add edges
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "critic")

    # Conditional edge from critic: RETRY → researcher, SATISFIED → writer
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher": "researcher",  # loop back
            "writer": "writer",          # proceed to writing
        },
    )

    workflow.add_edge("writer", END)

    # ── Compile with recursion limit ───────────────────────────────────────
    # LangGraph raises GraphRecursionError when this limit is hit.
    # This is intentional — it's the baseline crash behavior we're studying.
    compiled = workflow.compile()

    logger.info("Graph compiled successfully.")
    return compiled


# Module-level singleton (lazy-initialized on first import)
_graph = None


def get_graph():
    """Return the module-level compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
