# Agentic Circuit Breaker — Root Repository

## What This Is
A real-time headless middleware that detects semantic thrashing in stateful, LangGraph-style
multi-agent RAG pipelines and forces graceful summarization instead of a hard, budget-exhausting
crash.

**B.Tech Project** — Department of Computer Engineering, PCCOE Pune, A.Y. 2026-27

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/artnharsh/agentic-circuit-breaker.git
cd agentic-circuit-breaker

# 2. Install dependencies (creates .env from .env.example automatically)
make install

# 3. Edit .env — set LLM_PROVIDER=mock for zero-cost dev, or add real API keys
nano .env

# 4. Seed the document corpus
make seed-corpus

# 5. Start the engine
make dev
# → API running at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

## Reproduce the Crash (Baseline Behaviour)
```bash
make demo-crash
# → {"status": "crashed", "error": "GraphRecursionError", "iterations": 6}
```

## Test the Circuit Breaker (Day 4+)
```bash
make demo-normal
# → {"status": "completed", "output": "...", "iterations": 3}
```

## Project Structure
```
apps/engine/     ← Core service (FastAPI + LangGraph pipeline)
data/corpus/     ← Document corpus the agents retrieve from
data/adversarial_queries/  ← Thrashing-inducing test queries
scripts/         ← Experiment runner & report generator
docs/            ← PLAN.md, research PDFs, architecture diagrams
```

## Architecture

```
Query → [Researcher] → [Critic] → [Writer] → Answer
                  ↑         │
                  └── RETRY ┘   (semantic thrashing loop)

Middleware Interceptor (Day 2+):
  Per-node hooks → embeddings → token count → SQLite log

Heuristic Engine (Day 3+):
  cosine(N, N-2) → CLOSED / HALF-OPEN / OPEN state machine
  On OPEN → forced-summarization override injected
```

## Tech Stack
| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph |
| API | FastAPI + Pydantic |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI gpt-4o-mini / Anthropic Claude Haiku (switchable) |
| DB | SQLite → Postgres |
| Testing | pytest |

## Running Tests
```bash
make test
```

## Team
- Shreya Mahadik (121B1B150)
- Harshal Patil (123B1B217)
- Piyush Patil (123B1B220)
- Sayali Pawar (123B1B229)

**Supervisor:** Prof. Sonika Gill, PCCOE Pune
