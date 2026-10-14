"""
CircuitBreaker — the main user-facing class.

This is the primary entry point for the library. Users create a
CircuitBreaker instance, wrap their LangGraph node functions with
cb.watch(), and get a RunReport after the pipeline completes.

Usage:
    from agentic_circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(
        half_open_threshold=0.85,
        open_threshold=0.95,
        similarity_window=2,
    )

    workflow = StateGraph(MyState)
    workflow.add_node("researcher", cb.watch(researcher_fn))
    workflow.add_node("critic", cb.watch(critic_fn))
    workflow.add_node("writer", writer_fn)  # not watched
    # ... add edges ...
    graph = workflow.compile()

    result = graph.invoke({"query": "..."})
    report = cb.report()
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .config import BreakerConfig
from .embeddings import EmbeddingService
from .interceptor import NodeInterceptor, NodeTracker
from .state_machine import CircuitBreakerStateMachine
from .types import (
    NodeReport,
    RunReport,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Detects and interrupts semantic thrashing in LangGraph multi-agent pipelines.

    Wrap your node functions with ``cb.watch()`` to monitor them for repeated
    outputs. When a node produces outputs that are too similar to its previous
    outputs (measured by cosine similarity of sentence embeddings), the circuit
    breaker fires.

    Each watched node gets its own independent finite state machine (FSM),
    so you can pinpoint exactly *which* agent is stuck in a loop.

    Args:
        half_open_threshold: Cosine similarity >= this enters HALF_OPEN. Default: 0.85.
        open_threshold: Cosine similarity >= this triggers OPEN. Default: 0.95.
        similarity_window: Compare N-th output to (N - window)-th. Default: 2.
        embedding_model: Sentence-transformer model name. Default: "all-MiniLM-L6-v2".
        embed_fn: Custom embedding function (overrides embedding_model).
        on_open: Intervention strategy - "flag", "abort", or callable. Default: "flag".
        watch_nodes: List of node names to monitor. None = all watched nodes.
        text_extractor: Custom function (node_name, result, state) -> str.
        max_iterations: Hard cap on iterations per node. None = no cap.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config = BreakerConfig(**kwargs)
        self._config.validate()

        self._embedder = EmbeddingService(
            model_name=self._config.embedding_model,
            custom_fn=self._config.embed_fn,
        )

        self._trackers: dict[str, NodeTracker] = {}
        self._global_state: dict[str, Any] = {}
        self._start_time: float | None = None
        self._last_report: RunReport | None = None

    @property
    def config(self) -> BreakerConfig:
        """The active configuration."""
        return self._config

    @property
    def last_run(self) -> RunReport | None:
        """
        Report for the most recent completed run.

        Call ``report()`` or ``reset()`` to generate/save the report.
        """
        return self._last_report

    def watch(
        self,
        node_fn: Callable[..., dict],
        name: str | None = None,
    ) -> Callable[..., dict]:
        """
        Wrap a LangGraph node function with circuit breaker monitoring.

        The returned function has the same signature as the original and
        can be used directly with ``workflow.add_node()``.

        Each watched node gets its own FSM, embedding history, and
        similarity tracking — enabling per-node thrashing diagnostics.

        Args:
            node_fn: The original node function. Must accept a state dict
                     and return a dict (standard LangGraph node signature).
            name: Optional explicit name. Defaults to node_fn.__name__.

        Returns:
            A wrapped function with the same signature. Use this in
            ``workflow.add_node("name", cb.watch(original_fn))``.

        Example:
            cb = CircuitBreaker()
            workflow.add_node("researcher", cb.watch(researcher_fn))
        """
        node_name = name or getattr(node_fn, "__name__", f"node_{len(self._trackers)}")

        # Check watch_nodes filter
        if self._config.watch_nodes is not None and node_name not in self._config.watch_nodes:
            logger.debug(
                f"[CircuitBreaker] Skipping {node_name} — not in watch_nodes"
            )
            return node_fn  # return unwrapped

        # Start timing on first watch() call
        if self._start_time is None:
            self._start_time = time.time()

        # Create per-node tracker with its own FSM
        tracker = NodeTracker(
            node_name=node_name,
            state_machine=CircuitBreakerStateMachine(
                half_open_threshold=self._config.half_open_threshold,
                open_threshold=self._config.open_threshold,
            ),
        )
        self._trackers[node_name] = tracker

        # Create the interceptor wrapper
        interceptor = NodeInterceptor(
            node_name=node_name,
            original_fn=node_fn,
            config=self._config,
            embedder=self._embedder,
            tracker=tracker,
            global_state=self._global_state,
        )

        logger.info(
            f"[CircuitBreaker] Watching node '{node_name}' | "
            f"thresholds: HALF_OPEN≥{self._config.half_open_threshold}, "
            f"OPEN≥{self._config.open_threshold} | "
            f"window={self._config.similarity_window}"
        )

        return interceptor

    def wrap(self, graph: Any) -> Any:
        """
        Wrap a LangGraph compiled graph or StateGraph with circuit breaker monitoring.

        Automatically wraps every node in the graph (or those specified in
        ``watch_nodes``) with circuit breaker tracking.

        Args:
            graph: A compiled LangGraph graph (CompiledStateGraph) or StateGraph builder.

        Returns:
            The wrapped graph, ready to be invoked or compiled.

        Example:
            cb = CircuitBreaker()
            protected_graph = cb.wrap(my_compiled_graph)
            result = protected_graph.invoke({"query": "..."})
            report = cb.report()
        """
        # If it's a CompiledStateGraph (Pregel instance)
        if hasattr(graph, "nodes") and isinstance(graph.nodes, dict):
            for node_name, pregel_node in graph.nodes.items():
                if node_name.startswith("__"):
                    continue
                if self._config.watch_nodes is not None and node_name not in self._config.watch_nodes:
                    continue
                if hasattr(pregel_node, "bound"):
                    if hasattr(pregel_node.bound, "func"):
                        original_func = pregel_node.bound.func
                        pregel_node.bound.func = self.watch(original_func, name=node_name)
                    elif callable(pregel_node.bound):
                        original_bound = pregel_node.bound
                        pregel_node.bound = self.watch(original_bound, name=node_name)
            return graph

        # If it's a StateGraph builder before compile()
        if hasattr(graph, "nodes") and hasattr(graph, "compile"):
            for node_name, spec in list(graph.nodes.items()):
                if node_name.startswith("__"):
                    continue
                if self._config.watch_nodes is not None and node_name not in self._config.watch_nodes:
                    continue
                runnable = getattr(spec, "runnable", None)
                if runnable is not None and hasattr(runnable, "func"):
                    runnable.func = self.watch(runnable.func, name=node_name)
            return graph

        raise TypeError(
            f"Expected a LangGraph StateGraph or CompiledStateGraph, got {type(graph).__name__}"
        )

    def report(self) -> RunReport:
        """
        Generate a RunReport from the current accumulated state.

        Can be called at any time — during or after a run.

        Returns:
            RunReport with per-node breakdown.
        """
        duration_ms = 0.0
        if self._start_time is not None:
            duration_ms = (time.time() - self._start_time) * 1000

        # Build per-node reports
        node_reports: dict[str, NodeReport] = {}
        total_tokens = TokenUsage()

        for name, tracker in self._trackers.items():
            node_reports[name] = NodeReport(
                node_name=name,
                call_count=tracker.call_count,
                similarity_history=list(tracker.similarity_history),
                token_usage=tracker.token_usage,
                final_state=tracker.current_state,
            )
            total_tokens = total_tokens + tracker.token_usage

        return RunReport(
            breaker_triggered=self._global_state.get("triggered", False),
            triggered_by=self._global_state.get("source"),
            trigger_iteration=self._global_state.get("iteration"),
            node_reports=node_reports,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )

    def reset(self) -> RunReport:
        """
        Save the current run report and reset all state for a new run.

        Returns:
            The RunReport for the run that just ended.
        """
        # Save current report
        self._last_report = self.report()

        # Reset all trackers (same objects, cleared state)
        for tracker in self._trackers.values():
            tracker.reset()

        # Reset global state
        self._global_state.clear()
        self._start_time = time.time()

        logger.info("[CircuitBreaker] Reset — ready for new run.")
        return self._last_report

    @property
    def is_triggered(self) -> bool:
        """True if the circuit breaker has fired during the current run."""
        return self._global_state.get("triggered", False)

    @property
    def triggered_by(self) -> str | None:
        """Name of the node that triggered the breaker (None if not triggered)."""
        return self._global_state.get("source")

    def __repr__(self) -> str:
        watched = list(self._trackers.keys())
        triggered = self.is_triggered
        return (
            f"CircuitBreaker("
            f"watching={watched}, "
            f"triggered={triggered}, "
            f"window={self._config.similarity_window}, "
            f"thresholds=[{self._config.half_open_threshold}, "
            f"{self._config.open_threshold}])"
        )
