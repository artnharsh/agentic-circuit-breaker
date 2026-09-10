"""
Middleware Interceptor — hooks.py

Wraps each LangGraph node transparently to:
  1. Embed the Researcher's research_notes using sentence-transformers
  2. Extract real billed token counts from the LLM response
  3. Accumulate structured log entries in memory during the synchronous graph run
  4. Expose flush_pending_logs() for the API layer to batch-write to SQLite
     AFTER graph.invoke() returns (avoiding SQLite write-lock conflicts)

Design principle: the hooks must NOT modify agent prompts or the pipeline's
logic. They are pure observers — they read state before/after each node and
log the delta, without the agents knowing they're being watched.

Architecture (Day 2):
  - PendingLog: in-memory snapshot of one node call
  - InterceptorContext: holds the per-run PendingLog list + token accumulator
  - wrap_node(fn, ctx): returns a wrapped node that appends to ctx.pending_logs
  - flush_pending_logs(ctx, db): async — writes all pending logs to SQLite in one
    batch after graph.invoke() returns (called from runs.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from engine.agentic_graph.state import AgentState
from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
from engine.interceptor.embeddings import embedding_service
from engine.interceptor.token_counter import TokenUsage, extract_token_usage

logger = logging.getLogger(__name__)

# Type alias for LangGraph node functions
NodeFn = Callable[[AgentState], dict]


@dataclass
class PendingLog:
    """
    In-memory snapshot of a single node call, accumulated synchronously
    during graph.invoke() and written to SQLite in one async batch afterward.
    """
    run_id: str
    iteration: int
    node: str
    embedding: list[float] | None
    prompt_tokens: int
    completion_tokens: int
    breaker_state: str | None
    input_text: str
    output_text: str
    timestamp: datetime


@dataclass
class InterceptorContext:
    """
    Per-run context shared across all node hooks.

    During graph.invoke() (synchronous), hooks append PendingLog entries here.
    After graph.invoke() returns, the API layer calls flush_pending_logs() to
    write all entries to SQLite in one async batch.
    """

    run_id: str
    """The UUID string for this run — must match the Run row in the DB."""

    token_accumulator: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    """Running total of tokens consumed across all node calls."""

    pending_logs: list[PendingLog] = field(default_factory=list)
    """
    In-memory log entries accumulated synchronously during graph.invoke().
    Written to SQLite by flush_pending_logs() after graph.invoke() returns.
    """

    state_machine: CircuitBreakerStateMachine = field(
        default_factory=CircuitBreakerStateMachine
    )
    """The CLOSED/HALF_OPEN/OPEN finite state machine for this run."""

    embed_nodes: set[str] = field(default_factory=lambda: {"researcher"})
    """Which nodes get their output embedded (default: researcher only)."""


def wrap_node(node_fn: NodeFn, node_name: str, ctx: InterceptorContext) -> NodeFn:
    """
    Return a wrapped version of a LangGraph node function.

    The wrapped function:
      1. Calls the original node function
      2. Extracts token usage from the LLM response (if present in messages)
      3. Embeds the output text if this node is in ctx.embed_nodes
      4. Appends a PendingLog entry to ctx.pending_logs (in-memory, no DB I/O)
      5. Returns the original node's output unchanged

    DB writes happen later via flush_pending_logs() — never from sync code.

    Args:
        node_fn: The original node function (researcher_node, critic_node, etc.)
        node_name: 'researcher' | 'critic' | 'writer'
        ctx: The shared InterceptorContext for this run.

    Returns:
        A new function with the same signature — safe to use as a LangGraph node.
    """

    def wrapped(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0)

        # ── Call the original node ───────────────────────────────────────────
        result = node_fn(state)

        # ── Extract token usage from the last LLM response ──────────────────
        usage = _extract_usage_from_result(result)
        ctx.token_accumulator = ctx.token_accumulator + usage

        # ── Embed output text (researcher only by default) ───────────────────
        embedding: list[float] | None = None
        output_text = _get_output_text(node_name, result, state)

        if node_name in ctx.embed_nodes and output_text:
            try:
                embedding = embedding_service.embed(output_text)
                logger.debug(
                    f"[Interceptor] {node_name} | iter {iteration} | "
                    f"embedding dim={len(embedding)}"
                )
            except Exception as e:
                logger.warning(f"[Interceptor] Embedding failed for {node_name}: {e}")

        # ── Accumulate in-memory log entry (no DB I/O here) ─────────────────
        input_text = _get_input_text(node_name, state)
        ctx.pending_logs.append(PendingLog(
            run_id=ctx.run_id,
            iteration=iteration,
            node=node_name,
            embedding=embedding,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            breaker_state=None,  # Populated from Day 3
            input_text=input_text[:500],
            output_text=output_text[:500],
            timestamp=datetime.now(timezone.utc),
        ))

        logger.info(
            f"[Interceptor] {node_name} | iter {iteration} | "
            f"tokens: {usage.prompt_tokens}+{usage.completion_tokens} | "
            f"total so far: {ctx.token_accumulator.total_tokens}"
        )

        return result

    wrapped.__name__ = f"intercepted_{node_fn.__name__}"
    return wrapped


async def flush_pending_logs(ctx: InterceptorContext, db) -> None:
    """
    Write all accumulated PendingLog entries to SQLite in one async batch.

    Called by the API layer AFTER graph.invoke() returns, within the same
    async DB session used for the Run row — no locking conflicts.

    Args:
        ctx: The InterceptorContext containing pending_logs.
        db: An open AsyncSession to write to.
    """
    from engine.storage.models import IterationLog

    if not ctx.pending_logs:
        logger.debug("[Interceptor] No pending logs to flush.")
        return

    for entry in ctx.pending_logs:
        log = IterationLog(
            run_id=entry.run_id,
            iteration=entry.iteration,
            node=entry.node,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            breaker_state=entry.breaker_state,
            input_text=entry.input_text,
            output_text=entry.output_text,
            timestamp=entry.timestamp,
        )
        if entry.embedding is not None:
            log.set_embedding(entry.embedding)
        db.add(log)

    await db.flush()
    logger.info(
        f"[Interceptor] Flushed {len(ctx.pending_logs)} log entries to DB "
        f"for run {ctx.run_id[:8]}..."
    )
    ctx.pending_logs.clear()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_usage_from_result(result: dict) -> TokenUsage:
    """Extract token usage from the last AIMessage in the node's result."""
    messages = result.get("messages", [])
    if not messages:
        return TokenUsage(0, 0)
    try:
        return extract_token_usage(messages[-1])
    except Exception:
        return TokenUsage(0, 0)


def _get_output_text(node_name: str, result: dict, state: AgentState) -> str:
    """Extract the main text output from a node result."""
    if node_name == "researcher":
        return result.get("research_notes", "")
    elif node_name == "critic":
        return result.get("critique", "")
    elif node_name == "writer":
        return result.get("final_output", result.get("draft", ""))
    return ""


def _get_input_text(node_name: str, state: AgentState) -> str:
    """Extract the relevant input text for a node."""
    if node_name == "researcher":
        return state.get("query", "")
    elif node_name == "critic":
        return state.get("research_notes", "")[:500]
    elif node_name == "writer":
        return state.get("research_notes", "")[:500]
    return ""
