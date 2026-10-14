# Agentic Circuit Breaker

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A real-time, lightweight middleware that detects and interrupts **semantic thrashing loops** in stateful multi-agent RAG pipelines (LangGraph) with **per-node fault localization**.

Instead of letting agents burn thousands of tokens in non-progress retry loops until hard limits terminate the execution, `agentic-circuit-breaker` intercepts node outputs, monitors cosine similarity across an independent Finite State Machine (FSM) per agent, isolates the malfunctioning agent, and triggers graceful remediation.

---

## The Problem: Semantic Thrashing

In multi-agent architectures (e.g. Researcher ⇄ Critic), when an agent encounters an unanswerable adversarial query or missing corpus data, it often falls into a **semantic thrashing loop**:

```
           ┌──────────────┐
  START ──►│  Researcher  │◄─────────────┐
           └──────┬───────┘              │
                  │                      │
                  ▼                      │ (RETRY: notes insufficient)
           ┌──────────────┐              │
           │    Critic    ├──────────────┘ (Semantic Thrashing Loop)
           └──────┬───────┘                 Burns tokens until recursion_limit crash!
                  │
                  ▼ (SATISFIED)
           ┌──────────────┐
           │    Writer    ├──► END
           └──────────────┘
```

- **Without Circuit Breaker (Baseline):** Agents run until reaching `recursion_limit` and crash with `GraphRecursionError`, discarding partial work and wasting thousands of API tokens.
- **With Circuit Breaker:** The middleware detects that the Researcher is outputting semantically near-identical findings ($cosine \ge 0.95$), trips the breaker (`OPEN`), identifies `researcher` as the failing agent, and reroutes execution to synthesize a graceful partial answer.

---

## Quickstart: Use as a Library in Any LangGraph Pipeline

### 1. Installation

```bash
pip install agentic-circuit-breaker
```
*(Or install directly from GitHub: `pip install git+https://github.com/artnharsh/agentic-circuit-breaker.git#subdirectory=packages/agentic-circuit-breaker`)*

---

### 2. Wrap Your Compiled Graph (Option A: 3 Lines of Code)

You can wrap **any** compiled LangGraph graph without changing your agent functions or graph topology:

```python
from langgraph.graph import StateGraph, START, END
from agentic_circuit_breaker import CircuitBreaker

# 1. Build your standard LangGraph pipeline
workflow = StateGraph(MyAgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
# ... add conditional edges ...
graph = workflow.compile()

# 2. Attach the Circuit Breaker
cb = CircuitBreaker(
    half_open_threshold=0.85,
    open_threshold=0.95,
    similarity_window=2,
    watch_nodes=["researcher", "critic"],  # or None to watch all nodes
    on_open="flag",                        # "flag" | "abort" | custom callback
)
protected_graph = cb.wrap(graph)

# 3. Invoke as normal — circuit breaker monitors transparently
result = protected_graph.invoke({"query": "Compare Quantum Computing vs Neuromorphic Computing"})

# 4. Access per-node diagnostics
report = cb.report()
print(report.summary())
```

---

### 3. Wrap Individual Nodes (Option B)

Alternatively, wrap individual node functions during graph construction:

```python
cb = CircuitBreaker()

workflow = StateGraph(MyAgentState)
workflow.add_node("researcher", cb.watch(researcher_node))
workflow.add_node("critic", cb.watch(critic_node))
workflow.add_node("writer", writer_node)  # unmonitored final node
graph = workflow.compile()
```

---

## Per-Node Observability: Knowing WHICH Agent is Looping

A critical pain point in debugging multi-agent systems is identifying *which agent* is failing:
- Is the **Researcher** stuck repeating the same passage?
- Or is the **Critic** stuck rejecting valid answers?

Because `agentic-circuit-breaker` maintains an **independent FSM and embedding history per watched node**, developers receive actionable localization:

```python
report = cb.report()

# Check if breaker tripped
if report.breaker_triggered:
    print(f"Breaker tripped by: {report.triggered_by}")        # e.g. "researcher"
    print(f"Iteration tripped: {report.trigger_iteration}")      # e.g. 2
    print(f"Stuck nodes:       {report.thrashing_nodes}")        # ["researcher"]

# Access detailed trace per node
researcher_report = report.node_reports["researcher"]
print(f"Researcher max similarity: {researcher_report.max_similarity:.4f}")
print(f"Researcher tokens consumed: {researcher_report.token_usage.total_tokens}")
```

### Sample Diagnostic Output (`report.summary()`)

```text
══════════════════════════════════════════════════════
  Circuit Breaker Report
══════════════════════════════════════════════════════
  Status:       TRIGGERED at iteration 2
  Triggered by: researcher
  Total tokens: 4,143 (Saved: 5,380 tokens / 56.5%)
  Duration:     45.2ms

  ── researcher (2 calls) 🔥 THRASHING ──
    Iter 1: sim=0.7200  state=CLOSED ✓
    Iter 2: sim=0.9961  state=OPEN 🔥
    Max similarity: 0.9961
    Tokens consumed: 3,488

  ── critic (2 calls) ✓ OK ──
    Iter 1: sim=0.4500  state=CLOSED ✓
    Iter 2: sim=0.4800  state=CLOSED ✓
    Max similarity: 0.4800
    Tokens consumed: 528

══════════════════════════════════════════════════════
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `half_open_threshold` | `float` | `0.85` | Cosine similarity $\ge$ this enters `HALF_OPEN` (warning state). |
| `open_threshold` | `float` | `0.95` | Cosine similarity $\ge$ this triggers `OPEN` (trips circuit breaker). |
| `similarity_window` | `int` | `2` | Number of steps back to compare embeddings ($N$ vs $N - W$). |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Local sentence-transformers model (CPU, ~10ms per text). |
| `embed_fn` | `Callable` | `None` | Custom embedding callback `(str) -> list[float]` (overrides model). |
| `on_open` | `str \| Callable` | `"flag"` | `"flag"` (injects state), `"abort"` (raises `CircuitBreakerTripped`), or custom `(state, result, node_name) -> dict`. |
| `watch_nodes` | `list[str]` | `None` | Explicit list of node names to monitor (`None` = monitor all). |
| `text_extractor` | `Callable` | `None` | Custom text extractor `(node_name, result, state) -> str`. |
| `max_iterations` | `int` | `None` | Hard iteration limit per node (optional fallback). |

---

## Research & Empirical Evaluation

In benchmarks over adversarial retrieval queries against an enterprise document corpus:

| Metric | Baseline (Unprotected) | Circuit Breaker (Protected) | Improvement / Delta |
|:---|:---:|:---:|:---:|
| **Mean Token Consumption** | 9,497 tokens | 4,131 tokens | **-56.5% reduction** |
| **Crash Rate** | 100.0% (`GraphRecursionError`) | 0.0% (Graceful synthesis) | **-100.0% crashes** |
| **Paired $t$-test** | — | $t(2) = 727.98$ | **$p = 0.000002$ ($p < 0.001$)** |
| **Effect Size (Cohen's $d$)** | — | $d = 420.30$ | **Massive effect** |
| **Fault Localization Accuracy** | 0.0% (unobserved) | 100.0% (`researcher`) | **Pinpoints exact culprit** |

*Full experiment data and statistical evaluation details can be found in [`data/research_paper_report.md`](file:///home/xcaliber/Projects/agentic-circuit-breaker/data/research_paper_report.md).*

---

## Repository Structure

```
agentic-circuit-breaker/
├── packages/
│   └── agentic-circuit-breaker/      ← Reusable, pip-installable library
│       ├── pyproject.toml            ← Standalone library metadata (hatchling)
│       ├── README.md
│       ├── src/agentic_circuit_breaker/
│       │   ├── breaker.py            ← CircuitBreaker class (watch & wrap)
│       │   ├── interceptor.py        ← Per-node tracker & FSM wrapper
│       │   ├── state_machine.py      ← CLOSED / HALF_OPEN / OPEN FSM
│       │   ├── verifier.py           ← Multi-signal secondary verifier
│       │   ├── similarity.py         ← Cosine similarity math
│       │   ├── embeddings.py         ← Pluggable SentenceTransformer service
│       │   ├── intervention.py       ← Interruption strategies & exceptions
│       │   ├── types.py              ← RunReport, NodeReport, TokenUsage
│       │   └── config.py             ← BreakerConfig dataclass
│       └── tests/
│           └── test_library.py       ← Comprehensive library unit tests
├── apps/
│   └── engine/                       ← FastAPI demo service consuming the library
│       ├── src/engine/
│       │   ├── agentic_graph/        ← Multi-agent RAG pipeline (Researcher, Critic, Writer)
│       │   ├── api/                  ← REST routes (/runs, /health)
│       │   └── storage/              ← SQLite trace persistence
│       └── tests/                    ← Engine integration tests
├── scripts/
│   ├── run_experiment.py             ← Automated batch experiment benchmark
│   └── generate_report.py            ← Statistical report & t-test generator
├── data/
│   ├── corpus/                       ← Document corpus for RAG retrieval
│   └── adversarial_queries/          ← Thrashing-inducing benchmark queries
└── docs/                             ← Research plans and architecture specs
```

---

## Development & Testing

### Running Tests

Run all unit and integration tests (45 tests across the workspace):

```bash
pytest packages/agentic-circuit-breaker/tests/ apps/engine/tests/ -v
```

### Running Benchmark Experiments

Execute the automated baseline vs. protected comparison on adversarial queries:

```bash
python scripts/run_experiment.py --queries data/adversarial_queries/thrashing.jsonl
```

Generate statistical significance analysis and markdown tables:

```bash
python scripts/generate_report.py
```

### Running the Engine Service Locally

```bash
uvicorn engine.main:app --host 0.0.0.0 --port 8000 --reload
# API docs available at http://localhost:8000/docs
```

---

## Academic Context

**B.Tech Research Project**  
Department of Computer Engineering, Pimpri Chinchwad College of Engineering (PCCOE), Pune  
*Academic Year 2026–27*

- **Supervisor:** Prof. Sonika Gill
- **Team Members:**
  - Harshal Patil
  - Piyush Patil
  - Sayali Pawar
  - Shreya Mahadik

---

## License

This project is licensed under the [MIT License](LICENSE).
