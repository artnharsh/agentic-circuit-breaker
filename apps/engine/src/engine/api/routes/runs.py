"""
POST /runs — trigger a pipeline run.

Day 2 update:
  - Initializes DB on first call (tables auto-created)
  - Creates a LogStore for each run
  - Builds an InterceptorContext and wraps the graph nodes with hooks
  - Logs every node call (embedding + tokens + state) to SQLite
  - Returns total token usage in the response
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter
from langgraph.errors import GraphRecursionError

from engine.agentic_graph.graph import build_graph
from engine.agentic_graph.state import AgentState
from engine.api.schemas import RunRequest, RunResponse, RunStatus
from engine.config import get_settings
from engine.interceptor.hooks import InterceptorContext, flush_pending_logs
from engine.interceptor.log_store import LogStore
from engine.storage.db import get_session_factory, init_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["Runs"])

# Track whether the DB has been initialized this session
_db_initialized = False


async def _ensure_db() -> None:
    """Initialize DB tables on first request (idempotent)."""
    global _db_initialized
    if not _db_initialized:
        await init_db()
        _db_initialized = True


def _load_corpus_docs() -> list[str]:
    """Load document chunks from the corpus directory."""
    settings = get_settings()

    # routes/ → api/ → engine(pkg) → src/ → engine(app) → apps/ → repo root = parents[6]
    _repo_root = Path(__file__).parents[6]

    raw = settings.corpus_dir.strip()
    if not raw:
        corpus_path = _repo_root / "data" / "corpus"
    elif Path(raw).is_absolute():
        corpus_path = Path(raw)
    else:
        candidate = _repo_root / raw
        corpus_path = candidate if candidate.exists() else Path(raw)

    if not corpus_path.exists():
        logger.warning(f"Corpus directory not found: {corpus_path}. Using empty corpus.")
        return []

    docs: list[str] = []
    for file in sorted(corpus_path.glob("*.md")):
        try:
            docs.append(file.read_text(encoding="utf-8").strip())
        except Exception as e:
            logger.warning(f"Could not read corpus file {file}: {e}")

    for file in sorted(corpus_path.glob("*.txt")):
        try:
            docs.append(file.read_text(encoding="utf-8").strip())
        except Exception as e:
            logger.warning(f"Could not read corpus file {file}: {e}")

    logger.info(f"Loaded {len(docs)} corpus documents from {corpus_path}")
    return docs


@router.post("", response_model=RunResponse, summary="Trigger a pipeline run")
async def create_run(request: RunRequest) -> RunResponse:
    """
    Run the Researcher → Critic → Writer pipeline for the given query.

    **Day 2 behavior:**
    Every node call is now intercepted — embeddings computed, tokens counted,
    and rows written to SQLite. The total token count is returned in the response.

    **Baseline behavior (use_circuit_breaker=false):**
    Adversarial queries crash with GraphRecursionError (status="crashed").

    **Circuit breaker behavior (use_circuit_breaker=true, Day 4+):**
    Middleware intercepts thrashing and gracefully summarizes.
    """
    await _ensure_db()

    run_id = str(uuid.uuid4())
    logger.info(f"[Run {run_id[:8]}] Starting | Query: {request.query[:80]}...")

    corpus_docs = _load_corpus_docs()

    initial_state: AgentState = {
        "query": request.query,
        "corpus_docs": corpus_docs,
        "research_notes": "",
        "critique": "",
        "draft": "",
        "final_output": "",
        "iteration_count": 0,
        "circuit_breaker_triggered": False,
        "messages": [],
    }

    settings = get_settings()

    # ── Set up the Middleware Interceptor ──────────────────────────────────
    factory = get_session_factory()
    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query=request.query)

        ctx = InterceptorContext(run_id=run_id)
        graph = build_graph(interceptor_ctx=ctx, use_circuit_breaker=request.use_circuit_breaker)

        # ── Run the graph ──────────────────────────────────────────────────
        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": settings.recursion_limit},
            )

            # Write all accumulated interceptor log entries to SQLite
            await flush_pending_logs(ctx, db)

            iterations = final_state.get("iteration_count", 0)
            output = final_state.get("final_output", "")
            breaker_triggered = final_state.get("circuit_breaker_triggered", False)

            status = (
                RunStatus.BREAKER_TRIGGERED if breaker_triggered
                else RunStatus.COMPLETED
            )

            await log_store.finalize_run(
                status=status.value,
                iterations=iterations,
                circuit_breaker_triggered=breaker_triggered,
                final_output=output,
            )
            await db.commit()

            logger.info(
                f"[Run {run_id[:8]}] Completed | "
                f"iterations={iterations} | "
                f"tokens={ctx.token_accumulator.total_tokens}"
            )

            return RunResponse(
                run_id=run_id,
                status=status,
                query=request.query,
                output=output or None,
                iterations=iterations,
                circuit_breaker_triggered=breaker_triggered,
                tokens_used=ctx.token_accumulator.total_tokens or None,
            )

        except GraphRecursionError:
            # Flush whatever was logged before the crash
            await flush_pending_logs(ctx, db)

            logger.error(
                f"[Run {run_id[:8]}] CRASHED — GraphRecursionError "
                f"after recursion_limit={settings.recursion_limit} steps | "
                f"tokens consumed: {ctx.token_accumulator.total_tokens}"
            )

            await log_store.finalize_run(
                status="crashed",
                iterations=settings.recursion_limit,
                circuit_breaker_triggered=False,
                error=(
                    f"GraphRecursionError: The agent pipeline exceeded the maximum "
                    f"recursion limit ({settings.recursion_limit} steps) without "
                    f"converging on an answer."
                ),
            )
            await db.commit()

            return RunResponse(
                run_id=run_id,
                status=RunStatus.CRASHED,
                query=request.query,
                output=None,
                error=(
                    f"GraphRecursionError: The agent pipeline exceeded the maximum "
                    f"recursion limit ({settings.recursion_limit} steps) without "
                    f"converging on an answer. The corpus may not contain sufficient "
                    f"information to answer this query."
                ),
                iterations=settings.recursion_limit,
                circuit_breaker_triggered=False,
                tokens_used=ctx.token_accumulator.total_tokens or None,
            )

        except Exception as exc:
            logger.exception(f"[Run {run_id[:8]}] Unexpected error: {exc}")
            try:
                await log_store.finalize_run(
                    status="crashed",
                    iterations=0,
                    error=str(exc),
                )
                await db.commit()
            except Exception:
                pass
            raise
