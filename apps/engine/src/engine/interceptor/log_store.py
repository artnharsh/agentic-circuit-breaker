"""
Middleware Interceptor — log_store.py

Writes structured trace rows to the SQLite database after each node call.
Also provides the read path for the Heuristic Engine (Day 3) to fetch
recent embeddings for cosine similarity computation.

This module is the bridge between:
  - The Interceptor (writes rows on every node call)
  - The Heuristic Engine (reads rows to detect thrashing)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from engine.storage.models import IterationLog, Run

logger = logging.getLogger(__name__)


class LogStore:
    """
    Async log store backed by SQLite via SQLAlchemy.

    One LogStore instance is created per run and shared across all node
    hooks within that run.
    """

    def __init__(self, db: AsyncSession, run_id: str) -> None:
        self._db = db
        self._run_id = run_id
        self._token_totals: dict[str, int] = {"prompt": 0, "completion": 0}

    # ── Write path ─────────────────────────────────────────────────────────────

    async def create_run(
        self,
        query: str,
        status: str = "running",
    ) -> Run:
        """Insert the initial Run row when a pipeline invocation starts."""
        run = Run(
            id=self._run_id,
            query=query,
            status=status,
        )
        self._db.add(run)
        await self._db.flush()  # get the row into the DB without committing yet
        logger.debug(f"[LogStore] Created Run row: {self._run_id}")
        return run

    async def log_iteration(
        self,
        *,
        iteration: int,
        node: str,
        embedding: list[float] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        breaker_state: str | None = None,
        input_text: str = "",
        output_text: str = "",
    ) -> IterationLog:
        """
        Insert one IterationLog row.

        Args:
            iteration: Zero-indexed iteration number within the run.
            node: 'researcher' | 'critic' | 'writer'
            embedding: 384-dim float vector (from EmbeddingService), or None.
            prompt_tokens: Billed input tokens for this node call.
            completion_tokens: Billed output tokens for this node call.
            breaker_state: Current state machine state (populated from Day 3).
            input_text: Truncated input content for inspection.
            output_text: Truncated output content for inspection.
        """
        log = IterationLog(
            run_id=self._run_id,
            iteration=iteration,
            node=node,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            breaker_state=breaker_state,
            input_text=input_text[:500] if input_text else None,
            output_text=output_text[:500] if output_text else None,
            timestamp=datetime.now(timezone.utc),
        )
        if embedding is not None:
            log.set_embedding(embedding)

        self._db.add(log)
        await self._db.flush()

        # Track running token totals
        self._token_totals["prompt"] += prompt_tokens
        self._token_totals["completion"] += completion_tokens

        logger.debug(
            f"[LogStore] Iteration {iteration} | {node} | "
            f"tokens: {prompt_tokens}+{completion_tokens} | "
            f"embedding: {'yes' if embedding else 'no'}"
        )
        return log

    async def finalize_run(
        self,
        *,
        status: str,
        iterations: int,
        circuit_breaker_triggered: bool = False,
        final_output: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update the Run row when the pipeline completes or crashes."""
        result = await self._db.execute(
            select(Run).where(Run.id == self._run_id)
        )
        run = result.scalar_one_or_none()
        if run is None:
            logger.warning(f"[LogStore] Run {self._run_id} not found for finalize.")
            return

        run.status = status
        run.iterations = iterations
        run.circuit_breaker_triggered = circuit_breaker_triggered
        run.total_prompt_tokens = self._token_totals["prompt"]
        run.total_completion_tokens = self._token_totals["completion"]
        run.final_output = final_output
        run.error = error

        await self._db.flush()
        logger.info(
            f"[LogStore] Run {self._run_id} finalized | "
            f"status: {status} | iterations: {iterations} | "
            f"tokens: {self._token_totals['prompt']}+{self._token_totals['completion']}"
        )

    # ── Read path (used by Heuristic Engine, Day 3) ────────────────────────────

    async def get_researcher_embeddings(self, last_n: int = 5) -> list[list[float]]:
        """
        Fetch the most recent N embeddings from Researcher node calls for this run.

        Returns them in chronological order (oldest first).
        The Heuristic Engine uses this to compute cosine(N, N-2).

        Args:
            last_n: How many recent embeddings to return.

        Returns:
            List of float vectors, oldest first. May be shorter than last_n
            if fewer iterations have completed.
        """
        result = await self._db.execute(
            select(IterationLog)
            .where(
                IterationLog.run_id == self._run_id,
                IterationLog.node == "researcher",
                IterationLog.embedding_json.is_not(None),
            )
            .order_by(IterationLog.iteration.desc())
            .limit(last_n)
        )
        logs = result.scalars().all()

        # Reverse to get chronological order
        embeddings = []
        for log in reversed(logs):
            vec = log.get_embedding()
            if vec is not None:
                embeddings.append(vec)

        return embeddings

    @property
    def total_tokens(self) -> int:
        return self._token_totals["prompt"] + self._token_totals["completion"]
