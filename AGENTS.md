# Agentic Circuit Breaker — AI Agent Instructions

## Project Overview
This is a B.Tech research project implementing a real-time middleware (the "Agentic Circuit
Breaker") that detects and interrupts semantic thrashing loops in stateful, LangGraph-style
multi-agent RAG pipelines.

## Repository Layout
```
apps/engine/   ← Single deployable Python service (FastAPI + LangGraph)
data/          ← Corpus + adversarial queries (source of truth for experiments)
scripts/       ← Batch runner + report generator (not part of the API)
docs/          ← Research docs, architecture diagrams, PLAN.md
infra/         ← Docker/Postgres/Grafana configs (post-50% milestone)
```

## Core Conventions
1. **One deployable unit** — `apps/engine` is the only real service. Do NOT split into
   microservices until explicitly asked.
2. **Empty files are stubs** — all source files were scaffolded empty. Always implement the
   full module when touching a file.
3. **Mock LLM first** — default `LLM_PROVIDER=mock` means zero API cost during dev.
   Real LLM calls only for Day 4+ demo runs.
4. **SQLite for now** — `DB_URL` points to SQLite. Do not add Postgres config until
   post-50% milestone.
5. **No dashboard yet** — `apps/dashboard/` is a stretch goal. Don't touch it.
6. **Thresholds** — HALF-OPEN at 0.85, OPEN at ≥ 0.95 cosine similarity.
   These are in `.env` and referenced via `config.py` — never hardcode them.

## Key Files to Understand First
- `docs/PLAN.md` — The authoritative 5-day delivery plan
- `apps/engine/src/engine/config.py` — All configuration (pydantic-settings)
- `apps/engine/src/engine/agentic_graph/state.py` — Shared agent state TypedDict
- `apps/engine/src/engine/agentic_graph/graph.py` — LangGraph StateGraph definition

## Python / uv Conventions
- **Package manager:** `uv` — use `uv run <cmd>` or `uv sync` to manage deps
- **Python version:** 3.11+ (3.14 on this machine)
- **Imports:** absolute imports using the `engine.*` package path
- **Formatting:** `ruff` — run `make format` before committing
- **Type hints:** everywhere — no bare `Any` or untyped functions
