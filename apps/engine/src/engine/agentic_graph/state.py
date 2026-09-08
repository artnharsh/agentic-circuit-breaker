"""
Shared agent state schema for the LangGraph pipeline.

All nodes read from and write to AgentState. LangGraph passes this TypedDict
through the graph, merging updates from each node.

Fields are additive — each node only writes its own output fields.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The shared state object passed between all graph nodes.

    Managed by LangGraph — each node returns a dict with only the keys it updates.
    The `messages` field uses LangGraph's built-in add_messages reducer so messages
    accumulate (are appended) rather than replaced.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    """The original user query, set once at graph invocation and never changed."""

    corpus_docs: list[str]
    """
    Pre-loaded document chunks from the corpus.
    Loaded by the API layer before graph invocation, passed in as initial state.
    """

    # ── Intermediate results ──────────────────────────────────────────────────
    research_notes: str
    """
    Summary produced by the Researcher node.
    Updated on every iteration — contains the latest retrieval attempt.
    """

    critique: str
    """
    Feedback produced by the Critic node.
    "RETRY" means the loop continues; "SATISFIED" means the writer runs.
    """

    # ── Output ────────────────────────────────────────────────────────────────
    draft: str
    """Intermediate draft produced by the Writer (may be empty if not reached)."""

    final_output: str
    """The final answer returned to the caller."""

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    iteration_count: int
    """
    Number of Researcher→Critic cycles completed.
    Incremented by the Critic node on each pass.
    Used by the interceptor (Day 2+) to index log rows.
    """

    circuit_breaker_triggered: bool
    """
    Set to True by the intervention layer (Day 4+) when the breaker fires.
    False by default.
    """

    # ── LangGraph messages channel ────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    """
    Accumulated message history.
    Uses LangGraph's add_messages reducer — new messages are appended, not replaced.
    """
