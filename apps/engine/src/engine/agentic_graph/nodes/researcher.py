"""
Researcher agent node.

Responsibilities:
  1. Retrieve relevant passages from the pre-loaded corpus_docs
  2. Call the LLM to synthesize research notes from those passages
  3. Return updated `research_notes` to the graph state

The Researcher is the node that gets stuck in the loop on adversarial queries —
it keeps trying (and failing) to find specific data the corpus doesn't contain.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agentic_graph.state import AgentState
from engine.llm.client import get_llm

logger = logging.getLogger(__name__)


def researcher_node(state: AgentState) -> dict:
    """
    LangGraph node: Researcher.

    Retrieves relevant corpus passages and synthesizes research notes.
    Returns a partial state dict with only the fields this node updates.
    """
    query = state["query"]
    corpus_docs = state.get("corpus_docs", [])
    iteration = state.get("iteration_count", 0)
    previous_critique = state.get("critique", "")

    logger.info(f"[Researcher] Iteration {iteration} | Query: {query[:80]}...")

    # ── Build retrieval context from corpus ────────────────────────────────
    # Simple keyword-based retrieval for Day 1 (no vector store yet)
    relevant_docs = _retrieve_relevant_docs(query, corpus_docs)

    if relevant_docs:
        context = "\n\n---\n\n".join(relevant_docs[:3])  # cap at 3 chunks
    else:
        context = "No directly relevant documents found in the corpus."

    # ── Build the prompt ───────────────────────────────────────────────────
    system_prompt = (
        "You are a Researcher agent in a multi-agent RAG pipeline. "
        "Your job is to extract and synthesize relevant information from "
        "the provided document corpus to help answer the user's query. "
        "Be thorough but concise. Only report what the documents actually say."
    )

    critique_context = ""
    if previous_critique and "RETRY" in previous_critique:
        critique_context = (
            f"\n\nPrevious critique feedback (iteration {iteration}):\n"
            f"{previous_critique}\n"
            "Please address this feedback in your new research attempt."
        )

    human_prompt = (
        f"Query: {query}\n\n"
        f"Available corpus documents:\n{context}"
        f"{critique_context}\n\n"
        "Please synthesize the relevant information from these documents."
    )

    # ── Call the LLM ───────────────────────────────────────────────────────
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = llm.invoke(messages)
    research_notes = str(response.content)

    logger.info(
        f"[Researcher] Iteration {iteration} | "
        f"Notes length: {len(research_notes)} chars"
    )

    return {
        "research_notes": research_notes,
        "messages": [*messages, response],
    }


def _retrieve_relevant_docs(query: str, corpus_docs: list[str]) -> list[str]:
    """
    Simple keyword-based retrieval over corpus_docs.

    Scores each document by the number of query words it contains.
    Day 2+ will replace this with sentence-transformer vector similarity.
    """
    if not corpus_docs:
        return []

    query_words = set(query.lower().split())
    # Remove common stop words for better signal
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "what", "how", "why", "when", "where", "which", "who", "and",
        "or", "but", "in", "on", "at", "to", "for", "of", "with",
    }
    query_words -= stop_words

    scores: list[tuple[float, str]] = []
    for doc in corpus_docs:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        score = overlap / max(len(query_words), 1)
        scores.append((score, doc))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores if _ > 0]
