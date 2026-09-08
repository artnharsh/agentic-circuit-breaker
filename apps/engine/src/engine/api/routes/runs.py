"""
POST /runs — trigger a pipeline run.
GET  /runs/{run_id} — (Day 2+, once DB persistence is in place)

Day 1 implementation:
  - Runs the graph in-memory (no DB persistence yet)
  - Catches GraphRecursionError and returns status="crashed"
  - Returns status="completed" on successful runs
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from langgraph.errors import GraphRecursionError

from engine.agentic_graph.graph import get_graph
from engine.agentic_graph.state import AgentState
from engine.api.schemas import RunRequest, RunResponse, RunStatus
from engine.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["Runs"])


def _load_corpus_docs() -> list[str]:
    """
    Load document chunks from the corpus directory.
    Each .md or .txt file is treated as one document chunk.

    Resolves the corpus path relative to the repo root (4 levels up from this file),
    so the server works regardless of the working directory it's launched from.
    """
    settings = get_settings()

    # routes/ → api/ → engine(pkg) → src/ → engine(app) → apps/ → repo root = parents[6]
    _repo_root = Path(__file__).parents[6]

    raw = settings.corpus_dir.strip()
    if not raw:
        corpus_path = _repo_root / "data" / "corpus"
    elif Path(raw).is_absolute():
        corpus_path = Path(raw)
    else:
        # Try relative to repo root first, then CWD
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

    **Baseline behavior (use_circuit_breaker=false):**
    - On adversarial queries (unanswerable from corpus), the Researcher→Critic
      loop will hit `recursion_limit` and the response will have status="crashed"
      with error="GraphRecursionError".
    - This is the intentional baseline behavior that proves the research problem.

    **Circuit breaker behavior (use_circuit_breaker=true, Day 4+):**
    - The middleware intercepts each node call, detects semantic thrashing,
      and gracefully summarizes instead of crashing.
    """
    run_id = str(uuid.uuid4())
    logger.info(f"[Run {run_id}] Starting | Query: {request.query[:80]}...")

    # ── Load corpus ────────────────────────────────────────────────────────
    corpus_docs = _load_corpus_docs()

    # ── Build initial state ────────────────────────────────────────────────
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

    # ── Run the graph ──────────────────────────────────────────────────────
    settings = get_settings()
    graph = get_graph()

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": settings.recursion_limit},
        )

        iterations = final_state.get("iteration_count", 0)
        output = final_state.get("final_output", "")
        breaker_triggered = final_state.get("circuit_breaker_triggered", False)

        status = (
            RunStatus.BREAKER_TRIGGERED if breaker_triggered
            else RunStatus.COMPLETED
        )

        logger.info(
            f"[Run {run_id}] Completed | "
            f"Iterations: {iterations} | Status: {status}"
        )

        return RunResponse(
            run_id=run_id,
            status=status,
            query=request.query,
            output=output or None,
            iterations=iterations,
            circuit_breaker_triggered=breaker_triggered,
        )

    except GraphRecursionError as exc:
        # This is the baseline crash behavior — the graph hit recursion_limit
        # without the Critic ever saying SATISFIED.
        logger.error(
            f"[Run {run_id}] CRASHED — GraphRecursionError after "
            f"recursion_limit={settings.recursion_limit} steps"
        )
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
        )

    except Exception as exc:
        logger.exception(f"[Run {run_id}] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
