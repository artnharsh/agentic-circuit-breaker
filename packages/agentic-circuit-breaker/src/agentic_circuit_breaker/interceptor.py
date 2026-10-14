"""
Per-node interceptor — wraps individual LangGraph node functions.

Each watched node gets a NodeTracker (holds FSM + embeddings + history)
and a NodeInterceptor (the wrapper that calls the original function and
runs the breaker logic after each call).

Architecture:
  NodeTracker  — per-node mutable state (embeddings, FSM, similarity events)
  NodeInterceptor — the callable wrapper returned by CircuitBreaker.watch()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._utils import default_text_extractor, extract_token_usage_from_result
from .config import BreakerConfig
from .embeddings import EmbeddingService
from .intervention import CircuitBreakerTripped
from .similarity import get_recent_similarity
from .state_machine import CircuitBreakerStateMachine
from .types import BreakerState, SimilarityEvent, TokenUsage
from .verifier import HalfOpenVerifier

logger = logging.getLogger(__name__)


@dataclass
class NodeTracker:
    """
    Per-node mutable state accumulated during a pipeline run.

    Each watched node has its own NodeTracker with its own FSM instance,
    embedding history, and similarity events — enabling per-node
    thrashing diagnostics.
    """

    node_name: str
    """Name of the graph node being tracked."""

    call_count: int = 0
    """How many times this node has been invoked in this run."""

    embeddings: list[list[float]] = field(default_factory=list)
    """Chronological list of output embeddings (oldest first)."""

    similarity_history: list[SimilarityEvent] = field(default_factory=list)
    """Similarity measurements for this node."""

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    """Accumulated token usage for this node."""

    state_machine: CircuitBreakerStateMachine | None = None
    """This node's independent FSM instance."""

    @property
    def current_state(self) -> BreakerState:
        if self.state_machine is None:
            return BreakerState.CLOSED
        return self.state_machine.current_state

    @property
    def is_thrashing(self) -> bool:
        """True if this node's FSM has reached OPEN."""
        return self.current_state == BreakerState.OPEN or any(
            e.breaker_state == BreakerState.OPEN for e in self.similarity_history
        )

    @property
    def max_similarity(self) -> float:
        """Highest similarity score observed."""
        if not self.similarity_history:
            return 0.0
        return max(e.similarity for e in self.similarity_history)

    def reset(self) -> None:
        """Clear all accumulated state for a new run."""
        self.call_count = 0
        self.embeddings.clear()
        self.similarity_history.clear()
        self.token_usage = TokenUsage()
        if self.state_machine is not None:
            self.state_machine.reset()


class NodeInterceptor:
    """
    Wraps a single LangGraph node function with circuit breaker monitoring.

    On each call:
      1. Check if breaker already fired (on this or another node) → short-circuit
      2. Call the original node function
      3. Extract text from the output
      4. Embed the text
      5. Compute similarity against N-window-back embedding
      6. Update this node's FSM
      7. If OPEN → fire intervention (flag/abort/callback)
      8. Return the (possibly modified) result

    The wrapper is transparent — it does NOT modify the output unless the
    breaker fires with on_open="flag".
    """

    def __init__(
        self,
        node_name: str,
        original_fn: Callable[..., dict],
        config: BreakerConfig,
        embedder: EmbeddingService,
        tracker: NodeTracker,
        global_state: dict[str, Any],
    ) -> None:
        self._node_name = node_name
        self._original_fn = original_fn
        self._config = config
        self._embedder = embedder
        self._tracker = tracker
        self._global = global_state  # shared dict — cross-node breaker state
        self._text_extractor = config.text_extractor or default_text_extractor
        self._verifier = HalfOpenVerifier(
            half_open_threshold=config.half_open_threshold,
            open_threshold=config.open_threshold,
            max_iterations=config.max_iterations or 15,
        )

        # Preserve function metadata for LangGraph introspection
        self.__name__ = getattr(original_fn, "__name__", node_name)
        self.__qualname__ = getattr(original_fn, "__qualname__", node_name)

    def __call__(self, state: dict) -> dict:
        """Execute the wrapped node with circuit breaker monitoring."""
        tracker = self._tracker
        config = self._config

        # ── Pre-check: has breaker already fired on another node? ──────────
        if self._global.get("triggered"):
            logger.info(
                f"[Interceptor] {self._node_name} skipped — breaker already "
                f"fired on '{self._global.get('source', '?')}'"
            )
            return self._make_flag_result(self._global.get("source", "unknown"))

        # ── Pre-check: max iterations hard cap ──────────────────────────────
        if config.max_iterations and tracker.call_count >= config.max_iterations:
            logger.warning(
                f"[Interceptor] {self._node_name} hit max_iterations cap ({config.max_iterations})"
            )
            self._fire_breaker(iteration=tracker.call_count, similarity=1.0)
            return self._make_flag_result(self._node_name)

        # ── Run the original node ───────────────────────────────────────────
        result = self._original_fn(state)
        tracker.call_count += 1

        # ── Extract text and embed ──────────────────────────────────────────
        text = self._text_extractor(self._node_name, result, state)
        if text:
            try:
                embedding = self._embedder.embed(text)
                tracker.embeddings.append(embedding)
                logger.debug(
                    f"[Interceptor] {self._node_name} | call {tracker.call_count} | "
                    f"embedded {len(text)} chars → dim={len(embedding)}"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[Interceptor] Embedding failed for {self._node_name}: {e}")
        else:
            logger.debug(
                f"[Interceptor] {self._node_name} | call {tracker.call_count} | "
                f"no text extracted — skipping embedding"
            )

        # ── Extract token usage ────────────────────────────────────────────
        usage = extract_token_usage_from_result(result)
        tracker.token_usage = tracker.token_usage + usage

        # ── Compute similarity and update FSM ──────────────────────────────
        if len(tracker.embeddings) >= config.similarity_window + 1:
            sim = get_recent_similarity(tracker.embeddings, config.similarity_window)

            if sim is not None and tracker.state_machine is not None:
                new_state = tracker.state_machine.update(sim)

                # Record the event
                event = SimilarityEvent(
                    iteration=tracker.call_count,
                    similarity=sim,
                    breaker_state=new_state,
                    node_name=self._node_name,
                    timestamp=datetime.now(UTC),
                )
                tracker.similarity_history.append(event)

                logger.info(
                    f"[Interceptor] {self._node_name} | call {tracker.call_count} | "
                    f"sim={sim:.4f} | state={new_state.value} | "
                    f"tokens={usage.total_tokens}"
                )

                # HALF_OPEN → run secondary verifier (diagnostic only)
                if new_state == BreakerState.HALF_OPEN:
                    self._verifier.verify(
                        similarity=sim,
                        iteration=tracker.call_count,
                        total_tokens_used=tracker.token_usage.total_tokens,
                    )

                # OPEN → fire the breaker!
                if new_state == BreakerState.OPEN:
                    self._fire_breaker(
                        iteration=tracker.call_count,
                        similarity=sim,
                    )
                    result = self._apply_intervention(result, state, sim)
        else:
            logger.debug(
                f"[Interceptor] {self._node_name} | call {tracker.call_count} | "
                f"not enough embeddings for comparison "
                f"({len(tracker.embeddings)}/{config.similarity_window + 1})"
            )

        return result

    def _fire_breaker(self, iteration: int, similarity: float) -> None:
        """Record that this node fired the global breaker."""
        self._global["triggered"] = True
        self._global["source"] = self._node_name
        self._global["iteration"] = iteration
        self._global["similarity"] = similarity

        logger.warning(
            f"[Interceptor] 🔥 CIRCUIT BREAKER OPEN | "
            f"Node: {self._node_name} | "
            f"Call: {iteration} | "
            f"Similarity: {similarity:.4f}"
        )

    def _apply_intervention(self, result: dict, state: dict, similarity: float) -> dict:
        """Apply the configured intervention strategy."""
        on_open = self._config.on_open

        if on_open == "flag":
            result = dict(result)  # don't mutate original
            result["circuit_breaker_triggered"] = True
            result["circuit_breaker_source"] = self._node_name
            return result

        elif on_open == "abort":
            raise CircuitBreakerTripped(
                node_name=self._node_name,
                similarity=similarity,
                iteration=self._tracker.call_count,
            )

        elif callable(on_open):
            return on_open(state, result, self._node_name)

        return result

    def _make_flag_result(self, source: str) -> dict:
        """Create a minimal result dict with the breaker flag set."""
        return {
            "circuit_breaker_triggered": True,
            "circuit_breaker_source": source,
        }
