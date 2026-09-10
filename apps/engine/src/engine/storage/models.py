"""
Storage — models.py

SQLAlchemy ORM models for the structured log store.

Tables:
  - Run           : one row per pipeline invocation
  - IterationLog  : one row per node call (researcher / critic / writer)
                    stores the embedding vector, token counts, and breaker state

These models back the Middleware Interceptor (Day 2) and are read by
the Heuristic Engine (Day 3) to compute cosine similarity over the sliding window.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One row per POST /runs invocation."""

    __tablename__ = "runs"

    id = Column(String(36), primary_key=True)
    """UUID string — matches RunResponse.run_id."""

    query = Column(Text, nullable=False)
    """The original user query."""

    status = Column(String(32), nullable=False, default="running")
    """One of: running / completed / crashed / breaker_triggered."""

    iterations = Column(Integer, nullable=False, default=0)
    """Number of Researcher→Critic cycles completed."""

    circuit_breaker_triggered = Column(Boolean, nullable=False, default=False)
    """True if the breaker fired during this run."""

    total_prompt_tokens = Column(Integer, nullable=True)
    """Sum of prompt tokens across all node calls."""

    total_completion_tokens = Column(Integer, nullable=True)
    """Sum of completion tokens across all node calls."""

    final_output = Column(Text, nullable=True)
    """The final answer (null if crashed)."""

    error = Column(Text, nullable=True)
    """Error message (null unless crashed)."""

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )


class IterationLog(Base):
    """
    One row per node invocation within a run.

    This is the core data structure for the Heuristic Engine:
      - The `embedding` field stores the vector for the Researcher's research_notes
      - cosine(embedding[N], embedding[N-2]) drives the CLOSED/HALF-OPEN/OPEN state machine
    """

    __tablename__ = "iteration_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(String(36), nullable=False, index=True)
    """Foreign key to Run.id (not enforced at DB level for SQLite simplicity)."""

    iteration = Column(Integer, nullable=False)
    """Zero-indexed iteration number within the run."""

    node = Column(String(32), nullable=False)
    """Which node produced this log: 'researcher' | 'critic' | 'writer'."""

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_json = Column(Text, nullable=True)
    """
    JSON-serialized list[float] — the sentence-transformer vector for this step.
    Stored as text (SQLite has no native array type).
    Only populated for 'researcher' nodes (critic/writer embeddings are optional).
    """

    embedding_dim = Column(Integer, nullable=True)
    """Dimensionality of the embedding (384 for all-MiniLM-L6-v2)."""

    # ── Token counts (billing numbers) ───────────────────────────────────────
    prompt_tokens = Column(Integer, nullable=True)
    """Actual billed prompt tokens from the LLM API response usage field."""

    completion_tokens = Column(Integer, nullable=True)
    """Actual billed completion tokens from the LLM API response usage field."""

    # ── Circuit breaker state at the time of this call ────────────────────────
    breaker_state = Column(String(16), nullable=True)
    """CLOSED | HALF_OPEN | OPEN — populated from Day 3 onward."""

    # ── Content snapshots (truncated for storage efficiency) ──────────────────
    input_text = Column(Text, nullable=True)
    """First 500 chars of the node's input (research_notes or critique)."""

    output_text = Column(Text, nullable=True)
    """First 500 chars of the node's output."""

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def get_embedding(self) -> list[float] | None:
        """Deserialize the stored embedding from JSON."""
        if self.embedding_json is None:
            return None
        return json.loads(self.embedding_json)

    def set_embedding(self, vec: list[float]) -> None:
        """Serialize a float vector to JSON for storage."""
        self.embedding_json = json.dumps(vec)
        self.embedding_dim = len(vec)
