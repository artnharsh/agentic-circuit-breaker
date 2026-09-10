"""
Critic agent node.

Responsibilities:
  1. Evaluate whether the Researcher's notes are sufficient to answer the query
  2. Return "RETRY" if more research is needed (drives the loop)
  3. Return "SATISFIED" if the research is good enough (exits the loop)

This is the node that drives the thrashing loop on adversarial queries:
when the corpus doesn't contain the specific data the query demands, the
Critic will always respond RETRY — causing the graph to loop until
recursion_limit is hit.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agentic_graph.state import AgentState
from engine.llm.client import get_llm

logger = logging.getLogger(__name__)

# Marker injected by normal-query callers to short-circuit mock RETRY behavior
FORCE_SATISFIED_MARKER = "FORCE_SATISFIED"


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node: Critic.

    Evaluates the Researcher's output and either approves (SATISFIED) or
    requests another attempt (RETRY). Also increments iteration_count.
    """
    query = state["query"]
    research_notes = state.get("research_notes", "")
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Critic] Iteration {iteration} | Evaluating research notes...")

    # ── Build the evaluation prompt ────────────────────────────────────────
    system_prompt = (
        "You are a Critic agent in a multi-agent RAG pipeline. "
        "Your job is to evaluate whether the Researcher's findings are "
        "sufficient to answer the user's query accurately and completely.\n\n"
        "Respond with EXACTLY one of:\n"
        "  RETRY: <reason why the research is insufficient>\n"
        "  SATISFIED: <brief confirmation of what was well-researched>"
    )

    human_prompt = (
        f"Original query: {query}\n\n"
        f"Researcher's findings:\n{research_notes}\n\n"
        "Is this research sufficient to answer the query? "
        "If the research lacks specific data, quantitative facts, or "
        "precise information that the query demands, respond with RETRY."
    )

    # ── Call the LLM ───────────────────────────────────────────────────────
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = llm.invoke(messages)
    critique_text = str(response.content).strip()

    # ── Parse the verdict ──────────────────────────────────────────────────
    verdict = _parse_verdict(critique_text)
    new_iteration = iteration + 1

    logger.info(
        f"[Critic] Iteration {iteration} → {new_iteration} | "
        f"Verdict: {verdict}"
    )

    return {
        "critique": critique_text,
        "iteration_count": new_iteration,
        "messages": [*messages, response],
    }


def _parse_verdict(critique_text: str) -> str:
    """Extract RETRY or SATISFIED from the critic's response."""
    upper = critique_text.upper()
    if "SATISFIED" in upper:
        return "SATISFIED"
    return "RETRY"


def should_continue(state: AgentState) -> str:
    """
    LangGraph conditional edge function.

    Returns:
      "breaker_check" → if the Critic said RETRY (goes to circuit breaker check first)
      "writer"        → if the Critic said SATISFIED (proceed to writing directly)

    Day 3 change: RETRY now routes to breaker_check (not directly to researcher).
    The breaker_check node then decides: researcher (CLOSED/HALF_OPEN) or writer (OPEN).
    In baseline mode (no interceptor), breaker_check is a passthrough → researcher.
    """
    critique = state.get("critique", "RETRY")
    if "SATISFIED" in critique.upper():
        logger.info("[Critic] → routing to Writer")
        return "writer"
    logger.info("[Critic] → routing to BreakerCheck (RETRY)")
    return "breaker_check"
