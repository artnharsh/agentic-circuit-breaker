# Agentic Circuit Breaker — Engine Service

## What This Service Does
Core Python service that:
1. Runs the Researcher → Critic → Writer LangGraph pipeline
2. (Day 2+) Wraps every node with the Middleware Interceptor
3. (Day 3+) Feeds the Heuristic Engine for real-time loop detection
4. (Day 4+) Injects forced-summarization on OPEN state
5. Exposes a FastAPI HTTP API for triggering runs

## Key Entry Points
- `src/engine/main.py` — FastAPI application
- `src/engine/agentic_graph/graph.py` — build_graph() returns the compiled LangGraph
- `src/engine/config.py` — Settings singleton (pydantic-settings)

## Module Map
```
src/engine/
├── config.py           ← Settings (pydantic-settings, reads .env)
├── main.py             ← FastAPI app
├── api/                ← HTTP routes + Pydantic schemas
├── agentic_graph/      ← LangGraph graph, state, agent nodes
├── interceptor/        ← Node hooks, embeddings, token counter, log store
├── heuristic_engine/   ← Cosine similarity, state machine, verifier
├── intervention/       ← Override injector, escalation stub
├── storage/            ← SQLAlchemy models + DB session
└── llm/                ← LLM client factory + tokenizer
```

## Running Locally
```bash
# From repo root
make install
make dev       # starts FastAPI on :8000 with hot-reload
make test      # runs pytest
```

## Testing Without a Real LLM
Set `LLM_PROVIDER=mock` in `.env`. The mock client returns deterministic canned responses
without any API calls — ideal for unit testing the state machine and graph logic.
