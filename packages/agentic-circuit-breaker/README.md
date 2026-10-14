# Agentic Circuit Breaker

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time middleware that detects and interrupts semantic thrashing loops in stateful, LangGraph-style multi-agent pipelines with **per-node observability**.

---

## The Problem: Semantic Thrashing

In multi-agent systems (e.g. Researcher ⇄ Critic), agents often fall into non-progress loops where they generate semantically redundant outputs while consuming millions of tokens and never reaching termination until hard recursion limits kill the execution.

**Agentic Circuit Breaker** intercepts each node's output, monitors the cosine similarity of sentence embeddings across iterations using an independent Finite State Machine (FSM) per agent, and safely breaks thrashing loops before tokens are wasted.

---

## Key Features

- 🎯 **Per-Node Observability**: Pinpoint *which* agent in your graph is stuck in a loop (`triggered_by: "researcher"`).
- ⚡ **Minimal Overhead**: Embedding + cosine similarity adds < 15ms per agent call.
- 🔌 **Zero Graph Architecture Changes**: Wrap individual node functions with `cb.watch()` or wrap an entire graph with `cb.wrap(graph)`.
- 🎛️ **Configurable Thresholds & Window**: Tune `half_open_threshold` (default 0.85), `open_threshold` (default 0.95), and `similarity_window` (default 2).
- 🛡️ **Flexible Interventions**: 
  - `"flag"`: injects `circuit_breaker_triggered=True` so your pipeline can gracefully summarize partial results.
  - `"abort"`: raises `CircuitBreakerTripped`.
  - Custom callback: execute your own remediation logic.
- 📊 **Detailed Diagnostics**: Complete `RunReport` with iteration-by-iteration similarity tracking and token usage per node.

---

## Installation

```bash
pip install agentic-circuit-breaker
```

---

## Quickstart

### Option 1: Wrap an Entire Graph

```python
from langgraph.graph import StateGraph, START, END
from agentic_circuit_breaker import CircuitBreaker

# 1. Define your LangGraph pipeline as usual
workflow = StateGraph(MyState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
# ... edges ...
graph = workflow.compile()

# 2. Attach CircuitBreaker
cb = CircuitBreaker(
    half_open_threshold=0.85,
    open_threshold=0.95,
    similarity_window=2,
    watch_nodes=["researcher", "critic"],  # or None to watch all
)
protected_graph = cb.wrap(graph)

# 3. Invoke as normal
result = protected_graph.invoke({"query": "Compare Quantum Computing vs Neuromorphic Computing"})

# 4. Inspect per-node diagnostic report
report = cb.report()
print(report.summary())
```

### Option 2: Wrap Individual Node Functions

```python
cb = CircuitBreaker()

workflow = StateGraph(MyState)
workflow.add_node("researcher", cb.watch(researcher_node))
workflow.add_node("critic", cb.watch(critic_node))
workflow.add_node("writer", writer_node)  # unmonitored writer
graph = workflow.compile()
```

---

## Sample Diagnostic Report

```
══════════════════════════════════════════════════════
  Circuit Breaker Report
══════════════════════════════════════════════════════
  Status:       TRIGGERED at iteration 4
  Triggered by: researcher
  Total tokens: 2,847
  Duration:     3200.0ms

  ── researcher (4 calls) 🔥 THRASHING ──
    Iter 2: sim=0.7200  state=CLOSED ✓
    Iter 3: sim=0.8900  state=HALF_OPEN ⚠
    Iter 4: sim=0.9700  state=OPEN 🔥
    Max similarity: 0.9700
    Tokens consumed: 1,823

  ── critic (4 calls) ✓ OK ──
    Iter 2: sim=0.4500  state=CLOSED ✓
    Iter 3: sim=0.5100  state=CLOSED ✓
    Iter 4: sim=0.4800  state=CLOSED ✓
    Max similarity: 0.5100
    Tokens consumed: 1,024

══════════════════════════════════════════════════════
```

In the report above:
- **`researcher`** repeatedly output almost identical notes (`sim=0.9700`).
- **`critic`** produced distinct critiques (`sim=0.45`–`0.51`), proving the critic was working properly and isolating the bug to the researcher.
- The pipeline halted at iteration 4 instead of running until the recursion limit (e.g. 25 iterations), saving thousands of tokens.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `half_open_threshold` | `float` | `0.85` | Similarity threshold to enter `HALF_OPEN` (warning state). |
| `open_threshold` | `float` | `0.95` | Similarity threshold to enter `OPEN` (trips breaker). |
| `similarity_window` | `int` | `2` | Number of steps back to compare against ($N$ vs $N - W$). |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | SentenceTransformer model name. |
| `embed_fn` | `Callable` | `None` | Optional custom embedding function `(str) -> list[float]`. |
| `on_open` | `str \| Callable` | `"flag"` | `"flag"`, `"abort"`, or custom `(state, result, node_name) -> dict`. |
| `watch_nodes` | `list[str]` | `None` | Specific nodes to monitor. `None` = monitor all wrapped nodes. |
| `text_extractor` | `Callable` | `None` | Custom extractor `(node_name, result, state) -> str`. |
| `max_iterations` | `int` | `None` | Optional hard iteration ceiling. |

---

## License

MIT License.
