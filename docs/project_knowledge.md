# Agentic Circuit Breaker — Project Knowledge Summary

## What This Project Is

**Agentic Circuit Breakers: Semantic Loop Detection and Cost Mitigation in Stateful Multi-Agent RAG Architectures**

A B.Tech final-year research project (PCCOE, Pune, A.Y. 2026-27) that builds a **real-time, headless middleware** that:
1. Detects "semantic thrashing" — when a multi-agent LangGraph pipeline (Researcher → Critic → Writer) enters a recursive hallucination loop, re-asking the same question in slightly different phrasing
2. Intervenes gracefully before the run crashes on a hard `recursion_limit`, returning usable partial output instead of a hard failure

---

## The Core Research Claim

Traditional approach: Fixed `recursion_limit` = blunt trip-wire → entire run crashes, progress discarded.

**This project's approach:** A CLOSED → HALF-OPEN → OPEN state machine driven by **cosine similarity of consecutive reasoning attempt embeddings** (iteration N vs N-2):
- `Sim < 0.85` → **CLOSED** (agent making progress, let it run)
- `0.85 ≤ Sim < 0.95` → **HALF-OPEN** (suspected thrashing, lightweight re-check)
- `Sim ≥ 0.95` → **OPEN** (thrashing confirmed → inject forced-summarization override)

Key math: `Sim(N, N−2) = (e_N · e_(N−2)) / (||e_N|| × ||e_(N−2)||)`

---

## Tech Stack

| Layer | Choice |
|---|---|
| Core | Python 3.11+ |
| Agent orchestration | LangGraph (stateful graph, `recursion_limit`, `GraphRecursionError`) |
| API layer | FastAPI + Pydantic |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — local/CPU |
| LLM provider | OpenAI (gpt-4o-mini) or Anthropic (Claude Haiku) |
| DB | SQLite (dev) → Postgres (later) via SQLAlchemy |
| Testing | pytest |
| Containers | Docker + docker-compose |
| Dashboard | React + TS + Vite (STRETCH — after 50% milestone) |

---

## Repository Structure

```
agentic-circuit-breaker/
├── apps/
│   ├── engine/          ← THE core service (build first)
│   │   ├── src/engine/
│   │   │   ├── config.py
│   │   │   ├── main.py                      # FastAPI entrypoint
│   │   │   ├── api/routes/runs.py, health.py
│   │   │   ├── api/schemas.py
│   │   │   ├── agentic_graph/
│   │   │   │   ├── graph.py                 # LangGraph StateGraph
│   │   │   │   ├── state.py                 # Shared agent state
│   │   │   │   └── nodes/researcher.py, critic.py, writer.py
│   │   │   ├── interceptor/
│   │   │   │   ├── hooks.py                 # wraps each node
│   │   │   │   ├── embeddings.py            # sentence-transformer wrapper
│   │   │   │   ├── token_counter.py         # reads `usage` off every LLM call
│   │   │   │   └── log_store.py             # writes trace rows to SQLite
│   │   │   ├── heuristic_engine/
│   │   │   │   ├── similarity.py            # cosine sim (N vs N-2)
│   │   │   │   ├── state_machine.py         # CLOSED/HALF-OPEN/OPEN
│   │   │   │   └── verifier.py              # HALF-OPEN lightweight re-check
│   │   │   ├── intervention/
│   │   │   │   ├── override.py              # forced-summarization injector
│   │   │   │   └── escalation.py            # human-in-the-loop stub
│   │   │   ├── storage/models.py, db.py
│   │   │   └── llm/client.py, tokenizer.py
│   │   └── tests/
│   └── dashboard/       ← STRETCH GOAL only
├── data/corpus/, adversarial_queries/, ground_truth_labels.csv
├── scripts/run_experiment.py, generate_report.py, seed_corpus.py
└── infra/
```

**Current state:** All files and folders have been scaffolded, but ALL source files are empty (0 bytes). Everything needs to be written from scratch.

---

## 5-Day / 50% Delivery Plan (due Friday)

| Day | Focus | Deliverable |
|---|---|---|
| Day 1 | Repo scaffold + baseline pipeline | `apps/engine` running; Researcher→Critic→Writer LangGraph; adversarial query reproduces `GraphRecursionError` at `recursion_limit=6` |
| Day 2 | Middleware Interceptor | Every node wrapped by `hooks.py`; embeddings per attempt; real token `usage` read from LLM API; rows logged to SQLite |
| Day 3 | Heuristic Engine (isolated) | `similarity.py` cosine; `state_machine.py` CLOSED/HALF-OPEN/OPEN; unit tests pass with fake vectors — no LLM calls needed |
| Day 4 | Wire together + intervention | Heuristic engine subscribes to interceptor live; on OPEN, `override.py` injects forced-summarization; same adversarial query: baseline crashes, breaker doesn't |
| Day 5 | Small real experiment + packaging | `run_experiment.py` on 10–20 queries; `generate_report.py` produces real Table 1/2; Dockerize; push to GitHub; 2-min live demo |

**Friday target:** Live demo showing same adversarial question: baseline crashes, circuit breaker saves it — with real logged token numbers.

---

## Token Usage Implementation Detail

- **A. Billed numbers** (use in results tables):
  - OpenAI: `response.usage.prompt_tokens` / `response.usage.completion_tokens`
  - Anthropic: `response.usage.input_tokens` / `response.usage.output_tokens`
- **B. Pre-call estimate** (for HALF-OPEN verifier or context window checks):
  - OpenAI → `tiktoken`
  - Anthropic → built-in SDK token-counting method
- **Do NOT mix** embedding model tokens with billing tokens

---

## Key Research Context

- **Problem motivator:** 2026 MAESTRO study found 75.17% of multi-agent failures never trigger a system exception — they silently degrade
- **Gap this fills:** No existing work combines a stateful Hystrix-style finite-state machine with an LLM-native semantic signal for real-time loop detection
- **Comparison baselines:** 
  1. Baseline (no breaker) — crashes on `recursion_limit`
  2. Circuit breaker (this project)
  3. GPTCache-style exact-match (after 50% milestone)
- **Statistical tests planned:** paired t-test, Cohen's d, one-way ANOVA

---

## After 50% (Day 6+)

1. Scale dataset from ~10-20 to 40+ queries
2. Tune thresholds with ablation (vary 0.85/0.95, window size, toggle HALF-OPEN)
3. Add exact-match/GPTCache baseline (3rd arm in comparison table)
4. Real human-in-the-loop escalation path
5. Dashboard (React+TS+Vite)
6. CI, test coverage, Prometheus/Grafana
7. Final report/paper
