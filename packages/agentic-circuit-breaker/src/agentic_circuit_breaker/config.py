"""
Configuration for the Agentic Circuit Breaker.

Plain dataclass — no pydantic-settings dependency. Users pass values
directly or load from their own config system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class BreakerConfig:
    """
    Configuration for a CircuitBreaker instance.

    All values have sensible defaults. Override what you need:

        config = BreakerConfig(
            half_open_threshold=0.80,
            open_threshold=0.90,
            similarity_window=3,
        )

    Or pass kwargs directly to CircuitBreaker(...).
    """

    half_open_threshold: float = 0.85
    """Cosine similarity >= this enters HALF_OPEN (suspected thrashing).
    Default: 0.85 — one standard deviation above typical inter-iteration similarity."""

    open_threshold: float = 0.95
    """Cosine similarity >= this triggers OPEN (confirmed thrashing).
    Default: 0.95 — near-identical outputs."""

    similarity_window: int = 2
    """Compare current output embedding to the one N steps back.
    Default: 2 (N vs N-2). Higher values catch slower-onset thrashing.
    Recommended range: 1–4. This is an ablation parameter for experiments."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """Sentence-transformer model name for computing output embeddings.
    Default model: 384-dim, runs on CPU, ~10ms per text.
    Set to "" if providing a custom embed_fn."""

    embed_fn: Callable[[str], list[float]] | None = None
    """Optional custom embedding function. If provided, overrides
    embedding_model. Signature: (text: str) -> list[float]."""

    on_open: str | Callable[..., dict] = "flag"
    """What to do when the breaker fires (OPEN state).
    
    Built-in strategies:
      - "flag"  — inject circuit_breaker_triggered=True into node output (default)
      - "abort" — raise CircuitBreakerTripped exception
    
    Custom:
      - callable(state: dict, result: dict, node_name: str) -> dict
        Must return the (possibly modified) result dict.
    """

    watch_nodes: list[str] | None = None
    """Which nodes to monitor. None = watch ALL nodes wrapped via cb.watch().
    Example: ["researcher", "critic"] — only these nodes get similarity tracking."""

    text_extractor: Callable[[str, dict, dict], str] | None = None
    """Custom function to extract text from a node's output for embedding.
    
    Signature: (node_name: str, result: dict, state: dict) -> str
    
    If None, the default extractor is used:
      1. Looks for common fields: output, content, text, research_notes, etc.
      2. Falls back to last AIMessage content
      3. Falls back to concatenating all string values
    """

    max_iterations: int | None = None
    """Optional hard cap on iterations. If set, the breaker fires after this
    many calls to ANY watched node, regardless of similarity. None = no cap."""

    def validate(self) -> None:
        """Raise ValueError if config is invalid."""
        if not 0.0 <= self.half_open_threshold <= 1.0:
            raise ValueError(
                f"half_open_threshold must be in [0, 1], got {self.half_open_threshold}"
            )
        if not 0.0 <= self.open_threshold <= 1.0:
            raise ValueError(f"open_threshold must be in [0, 1], got {self.open_threshold}")
        if self.half_open_threshold >= self.open_threshold:
            raise ValueError(
                f"half_open_threshold ({self.half_open_threshold}) must be < "
                f"open_threshold ({self.open_threshold})"
            )
        if self.similarity_window < 1:
            raise ValueError(f"similarity_window must be >= 1, got {self.similarity_window}")
        if isinstance(self.on_open, str) and self.on_open not in ("flag", "abort"):
            raise ValueError(
                f"on_open must be 'flag', 'abort', or a callable, got {self.on_open!r}"
            )
