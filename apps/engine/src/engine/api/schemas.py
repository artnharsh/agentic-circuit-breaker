"""Pydantic request/response schemas for the Engine API."""

from __future__ import annotations

from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Possible outcomes of a pipeline run."""
    COMPLETED = "completed"
    CRASHED = "crashed"
    BREAKER_TRIGGERED = "breaker_triggered"  # Day 4+
    RUNNING = "running"


class RunRequest(BaseModel):
    """POST /runs — request body."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The query to run through the Researcher→Critic→Writer pipeline.",
        examples=["What is solar energy and how does it work?"],
    )
    use_circuit_breaker: bool = Field(
        default=False,
        description=(
            "Enable the circuit breaker middleware. "
            "False = baseline (will crash on adversarial queries). "
            "True = protected (Day 4+, breaker intercepts). "
        ),
    )


class RunResponse(BaseModel):
    """POST /runs — response body."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: RunStatus
    query: str
    output: Optional[str] = Field(
        default=None,
        description="The final answer (only present when status=completed or breaker_triggered).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (only present when status=crashed).",
    )
    iterations: int = Field(
        default=0,
        description="Number of Researcher→Critic cycles completed.",
    )
    circuit_breaker_triggered: bool = Field(
        default=False,
        description="True if the circuit breaker fired during this run.",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens consumed (prompt + completion). Populated from Day 2.",
    )
