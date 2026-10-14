"""
Core data types for the Agentic Circuit Breaker library.

All public types used in the API surface are defined here to avoid
circular imports between modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class BreakerState(str, Enum):
    """State of a per-node circuit breaker FSM."""

    CLOSED = "CLOSED"
    """Normal operation — agent is making progress."""

    HALF_OPEN = "HALF_OPEN"
    """Suspected thrashing — similarity is high but below OPEN threshold.
    Gives the agent one more chance before escalating."""

    OPEN = "OPEN"
    """Thrashing confirmed — breaker fires, pipeline is redirected."""


@dataclass
class TokenUsage:
    """Token counts extracted from LLM API responses."""

    prompt_tokens: int = 0
    """Tokens in the input/prompt (billed by the LLM provider)."""

    completion_tokens: int = 0
    """Tokens in the output/completion (billed by the LLM provider)."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __repr__(self) -> str:
        return f"TokenUsage(prompt={self.prompt_tokens}, completion={self.completion_tokens}, total={self.total_tokens})"


ZERO_USAGE = TokenUsage(0, 0)


@dataclass
class SimilarityEvent:
    """One similarity measurement for a specific node at a specific iteration."""

    iteration: int
    """The call number for this node (1-indexed)."""

    similarity: float
    """Cosine similarity value in [-1.0, 1.0]."""

    breaker_state: BreakerState
    """FSM state after processing this similarity value."""

    node_name: str
    """Which node produced this measurement."""

    timestamp: datetime
    """When this measurement was taken."""

    def __repr__(self) -> str:
        return (
            f"SimilarityEvent(iter={self.iteration}, sim={self.similarity:.4f}, "
            f"state={self.breaker_state.value}, node={self.node_name!r})"
        )


@dataclass
class NodeReport:
    """Per-node summary included in a RunReport."""

    node_name: str
    """Name of the graph node."""

    call_count: int
    """How many times this node was invoked during the run."""

    similarity_history: list[SimilarityEvent] = field(default_factory=list)
    """Chronological similarity measurements for this node."""

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    """Total tokens consumed by this node across all calls."""

    final_state: BreakerState = BreakerState.CLOSED
    """The FSM state for this node at end of run."""

    @property
    def max_similarity(self) -> float:
        """Highest similarity score observed for this node."""
        if not self.similarity_history:
            return 0.0
        return max(e.similarity for e in self.similarity_history)

    @property
    def is_thrashing(self) -> bool:
        """True if this node's FSM reached OPEN at any point."""
        return self.final_state == BreakerState.OPEN or any(
            e.breaker_state == BreakerState.OPEN for e in self.similarity_history
        )


@dataclass
class RunReport:
    """
    Complete diagnostic report for one pipeline run.

    This is the primary output a developer inspects after running their
    pipeline through the circuit breaker.
    """

    breaker_triggered: bool
    """Whether the circuit breaker fired during this run."""

    triggered_by: str | None = None
    """Name of the node whose FSM first reached OPEN (None if not triggered)."""

    trigger_iteration: int | None = None
    """Iteration number when the breaker fired (None if not triggered)."""

    node_reports: dict[str, NodeReport] = field(default_factory=dict)
    """Per-node breakdown — keyed by node name."""

    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    """Aggregate token usage across all watched nodes."""

    duration_ms: float = 0.0
    """Total run duration in milliseconds."""

    @property
    def thrashing_nodes(self) -> list[str]:
        """List of node names whose FSM reached OPEN or HALF_OPEN."""
        return [
            name for name, report in self.node_reports.items()
            if report.is_thrashing
        ]

    @property
    def similarity_timeline(self) -> list[SimilarityEvent]:
        """Chronological list of all similarity events across all nodes."""
        events: list[SimilarityEvent] = []
        for report in self.node_reports.values():
            events.extend(report.similarity_history)
        events.sort(key=lambda e: e.timestamp)
        return events

    def summary(self) -> str:
        """Human-readable summary of the run."""
        lines = [
            "══════════════════════════════════════════════════════",
            "  Circuit Breaker Report",
            "══════════════════════════════════════════════════════",
        ]

        if self.breaker_triggered:
            lines.append(
                f"  Status:       TRIGGERED at iteration {self.trigger_iteration}"
            )
            lines.append(f"  Triggered by: {self.triggered_by}")
        else:
            lines.append("  Status:       NOT TRIGGERED (clean run)")

        lines.append(f"  Total tokens: {self.total_tokens.total_tokens:,}")
        lines.append(f"  Duration:     {self.duration_ms:.1f}ms")
        lines.append("")

        for name, report in self.node_reports.items():
            status = "🔥 THRASHING" if report.is_thrashing else "✓ OK"
            lines.append(f"  ── {name} ({report.call_count} calls) {status} ──")

            if report.similarity_history:
                for event in report.similarity_history:
                    state_icon = {
                        BreakerState.CLOSED: "✓",
                        BreakerState.HALF_OPEN: "⚠",
                        BreakerState.OPEN: "🔥",
                    }.get(event.breaker_state, "?")
                    lines.append(
                        f"    Iter {event.iteration}: "
                        f"sim={event.similarity:.4f}  "
                        f"state={event.breaker_state.value} {state_icon}"
                    )
            else:
                lines.append("    (no similarity data — not enough calls)")

            lines.append(
                f"    Max similarity: {report.max_similarity:.4f}"
            )
            lines.append(
                f"    Tokens consumed: {report.token_usage.total_tokens:,}"
            )
            lines.append("")

        lines.append("══════════════════════════════════════════════════════")
        return "\n".join(lines)
