"""
Writer agent node.

Responsibilities:
  1. Receive the approved research notes from the Researcher (after Critic says SATISFIED)
  2. Produce a final, coherent answer for the user
  3. Set `final_output` in the graph state

This node is only reached when the Critic is satisfied — on adversarial queries
that trigger the thrashing loop, this node will never be reached (the graph will
crash with GraphRecursionError before the Critic ever says SATISFIED).
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agentic_graph.state import AgentState
from engine.llm.client import get_llm

logger = logging.getLogger(__name__)


def writer_node(state: AgentState) -> dict:
    """
    LangGraph node: Writer.

    Synthesizes the final answer from approved research notes.
    Returns updated `draft` and `final_output` fields.
    """
    query = state["query"]
    research_notes = state.get("research_notes", "")
    iteration_count = state.get("iteration_count", 0)

    logger.info(
        f"[Writer] Composing final answer after {iteration_count} research iterations..."
    )

    # ── Build the writing prompt ───────────────────────────────────────────
    system_prompt = (
        "You are a Writer agent in a multi-agent RAG pipeline. "
        "Your job is to produce a clear, accurate, and well-structured "
        "final answer based on the research provided by the Researcher. "
        "Write for a knowledgeable audience. Be concise and factual."
    )

    human_prompt = (
        f"Query: {query}\n\n"
        f"Research notes (approved by Critic):\n{research_notes}\n\n"
        "Please write a comprehensive final answer to the query based on "
        "the provided research. Structure your response clearly."
    )

    # ── Call the LLM ───────────────────────────────────────────────────────
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = llm.invoke(messages)
    final_output = str(response.content)

    logger.info(
        f"[Writer] Final answer composed | Length: {len(final_output)} chars"
    )

    return {
        "draft": final_output,
        "final_output": final_output,
        "messages": [*messages, response],
    }
