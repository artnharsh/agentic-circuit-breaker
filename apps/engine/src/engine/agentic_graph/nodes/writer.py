"""
Writer agent node.

Responsibilities:
  1. Receive the approved research notes from the Researcher
     (after Critic says SATISFIED — normal path)
  2. OR: Receive a forced-summarization signal from the Circuit Breaker
     (after circuit_breaker_triggered=True — Day 3+ path)
  3. Produce a final, coherent answer (or best-effort partial answer)
  4. Set `final_output` in the graph state

Day 3 addition: Forced-summarization mode
  When `circuit_breaker_triggered=True`, the Writer knows the Researcher
  exhausted the corpus without finding the needed data. It uses a different
  system prompt that tells it to:
    - Acknowledge the limitation honestly
    - Summarize what WAS found from the corpus
    - Explicitly state what information was not available
  This produces a graceful degradation instead of a crash.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agentic_graph.state import AgentState
from engine.llm.client import get_llm

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_NORMAL_SYSTEM_PROMPT = (
    "You are a Writer agent in a multi-agent RAG pipeline. "
    "Your job is to produce a clear, accurate, and well-structured "
    "final answer based on the research provided by the Researcher. "
    "Write for a knowledgeable audience. Be concise and factual."
)

_FORCED_SUMMARY_SYSTEM_PROMPT = (
    "You are a Writer agent in a multi-agent RAG pipeline. "
    "The Researcher was unable to find complete information to answer the query — "
    "it tried multiple times but the corpus does not contain the specific data requested. "
    "Your job is to produce a PARTIAL ANSWER that:\n"
    "  1. Clearly states that complete information was not available in the knowledge base\n"
    "  2. Summarizes the RELATED information that WAS found\n"
    "  3. Explicitly lists what specific information was missing\n"
    "  4. Suggests where the user might find the missing information\n"
    "Be honest about limitations. Do not hallucinate missing data."
)


def writer_node(state: AgentState) -> dict:
    """
    LangGraph node: Writer.

    Synthesizes the final answer (normal path) or a forced partial summary
    (circuit breaker path). Returns updated `draft` and `final_output` fields.
    """
    query = state["query"]
    research_notes = state.get("research_notes", "")
    iteration_count = state.get("iteration_count", 0)
    circuit_breaker_triggered = state.get("circuit_breaker_triggered", False)

    mode = "forced-summarization" if circuit_breaker_triggered else "normal"
    logger.info(
        f"[Writer] Composing {'partial answer' if circuit_breaker_triggered else 'final answer'} "
        f"| mode={mode} | iterations={iteration_count}"
    )

    # ── Build the writing prompt ───────────────────────────────────────────────
    if circuit_breaker_triggered:
        system_prompt = _FORCED_SUMMARY_SYSTEM_PROMPT
        human_prompt = (
            f"Query: {query}\n\n"
            f"Research notes gathered (partial — corpus could not fully answer):\n"
            f"{research_notes or 'No relevant information was found in the corpus.'}\n\n"
            "Please write a partial answer following your instructions. "
            "Be transparent about what information was and was not available."
        )
    else:
        system_prompt = _NORMAL_SYSTEM_PROMPT
        human_prompt = (
            f"Query: {query}\n\n"
            f"Research notes (approved by Critic):\n{research_notes}\n\n"
            "Please write a comprehensive final answer to the query based on "
            "the provided research. Structure your response clearly."
        )

    # ── Call the LLM ──────────────────────────────────────────────────────────
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = llm.invoke(messages)
    final_output = str(response.content)

    logger.info(
        f"[Writer] Answer composed | "
        f"length={len(final_output)} chars | "
        f"circuit_breaker={'triggered' if circuit_breaker_triggered else 'not triggered'}"
    )

    return {
        "draft": final_output,
        "final_output": final_output,
        "messages": [*messages, response],
    }
