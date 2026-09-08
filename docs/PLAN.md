# Agentic Circuit Breaker — Full Build Plan

This document has four parts:
1. Recommended tech stack (with reasoning)
2. Clean folder structure for the whole repo
3. The **100% plan** (every phase, end to end)
4. The **5-day / 50% plan** — exactly what to build this week, and where to pick up after

---

## 1. Tech Stack — What to Use and Why

| Layer | Choice | Why |
|---|---|---|
| Core engine | **Python 3.11+** | LangGraph, sentence-transformers, numpy/scipy/statistics all live natively in Python. This is an ML/research-heavy project — the hard part is the similarity math and the agent graph, not raw request throughput. Rewriting that in Node/Go buys you nothing and costs you days. |
| Agent orchestration | **LangGraph** | It's literally what your synopsis is built around (stateful graph, `recursion_limit`, `GraphRecursionError`). No reason to swap it. |
| API layer | **FastAPI** | Async, typed (Pydantic), auto-generates OpenAPI docs, trivial to Dockerize. Standard choice for a Python backend service in 2025-26. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Free, runs locally/CPU, fast enough for real-time similarity checks, matches your synopsis's "local sentence-embedding model." |
| LLM provider | **OpenAI (gpt-4o-mini) or Anthropic (Claude Haiku)** via official SDK | Cheap + fast enough for Researcher/Critic/Writer roles; both SDKs return exact token usage per call (see Section 5 below). |
| Structured log store | **SQLite → Postgres later** | SQLite needs zero setup for week 1; swap the connection string for Postgres later, same SQLAlchemy models. |
| Testing | **pytest** | Standard; lets you unit-test the state machine with fake embeddings, without ever calling a real LLM. |
| Containers | **Docker + docker-compose** | One command to run engine + DB together; matches the clean structure you want. |
| Dashboard (stretch, optional) | **React + TypeScript + Vite** | Only build this after the 50% milestone. Not needed to prove the core claim. |

**Why not Node or Go for this project:** Node/Go are the right call when you're building a high-throughput API gateway or a real-time multiplayer backend. Here, the actual intellectual product is the similarity math + state machine + statistics — all of that is a few lines in Python/numpy/scipy and painful to reimplement in Node or Go. Use Python for the engine; you can *always* put a thin Node/Go gateway in front of it later if you ever need one — you will not need one this week.

---

## 2. Clean Folder Structure

```
agentic-circuit-breaker/
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pre-commit-config.yaml
├── README.md
├── AGENTS.md
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── docs/
│   ├── synopsis.pdf
│   ├── architecture-diagrams/
│   └── literature-review.md
├── data/
│   ├── corpus/                      # documents the agents retrieve from
│   ├── adversarial_queries/         # thrashing-inducing + normal queries (jsonl)
│   └── ground_truth_labels.csv
├── apps/
│   ├── engine/                      # THE core service — build this first
│   │   ├── AGENTS.md
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/engine/
│   │   │   ├── config.py
│   │   │   ├── main.py                      # FastAPI entrypoint
│   │   │   ├── api/
│   │   │   │   ├── routes/
│   │   │   │   │   ├── runs.py               # POST /runs, GET /runs/{id}
│   │   │   │   │   └── health.py
│   │   │   │   └── schemas.py                # pydantic request/response models
│   │   │   ├── agentic_graph/
│   │   │   │   ├── graph.py                  # LangGraph StateGraph definition
│   │   │   │   ├── state.py                  # shared agent state schema
│   │   │   │   └── nodes/
│   │   │   │       ├── researcher.py
│   │   │   │       ├── critic.py
│   │   │   │       └── writer.py
│   │   │   ├── interceptor/
│   │   │   │   ├── hooks.py                  # wraps each LangGraph node
│   │   │   │   ├── embeddings.py             # sentence-transformer wrapper
│   │   │   │   ├── token_counter.py          # reads `usage` off every LLM call
│   │   │   │   └── log_store.py              # writes trace rows to DB
│   │   │   ├── heuristic_engine/
│   │   │   │   ├── similarity.py             # cosine similarity (N vs N-2)
│   │   │   │   ├── state_machine.py          # CLOSED / HALF-OPEN / OPEN
│   │   │   │   └── verifier.py               # HALF-OPEN lightweight re-check
│   │   │   ├── intervention/
│   │   │   │   ├── override.py               # forced-summarization injector
│   │   │   │   └── escalation.py             # human-in-the-loop stub
│   │   │   ├── storage/
│   │   │   │   ├── models.py                 # SQLAlchemy models
│   │   │   │   ├── db.py
│   │   │   │   └── migrations/
│   │   │   └── llm/
│   │   │       ├── client.py                 # wraps OpenAI/Anthropic client
│   │   │       └── tokenizer.py              # tiktoken / anthropic token counting
│   │   └── tests/
│   │       ├── test_similarity.py
│   │       ├── test_state_machine.py
│   │       └── test_end_to_end.py
│   └── dashboard/                   # STRETCH GOAL — build only after 50%
│       ├── Dockerfile
│       ├── package.json
│       └── src/
│           ├── App.tsx
│           ├── components/
│           │   ├── RunTimeline.tsx
│           │   ├── StateBadge.tsx
│           │   └── TokenCostChart.tsx
│           └── services/api.ts
├── scripts/
│   ├── run_experiment.py            # batch runner (this becomes your Assignment-2 data, for real)
│   ├── generate_report.py           # stats + charts, same shape as Assignment 2
│   └── seed_corpus.py
└── infra/
    ├── docker/postgres/
    └── grafana/                     # optional, later
```

Notes on the conventions borrowed from your friend's repo:
- `AGENTS.md` at root and per-app — a short file telling any AI coding tool (Claude Code, Cursor) what that folder is and its conventions. Cheap to add, genuinely useful once you're moving fast.
- One `apps/<name>/` per deployable unit. You only have **one** real deployable unit this week (`engine`) — resist the urge to split into `api`/`worker`/`web` microservices like your friend's project until you actually need to scale something. Splitting early costs you days you don't have.
- `packages/shared` isn't included yet because you have nothing to share between two TypeScript apps yet — add it only once the dashboard exists and needs the same types as something else.

---

## 3. The 100% Plan (full project, all phases)

| Phase | What it delivers |
|---|---|
| **0. Setup** | Repo scaffold above, Docker Compose skeleton, `.env.example`, CI skeleton |
| **1. Baseline pipeline** | Researcher → Critic → Writer LangGraph pipeline over a small document corpus; hard `recursion_limit`; reliably reproduces a crash on a crafted adversarial question |
| **2. Middleware Interceptor** | Node-hook wrapper, embedding generator, token counter, structured log store (SQLite) — all logging without touching agent prompts |
| **3. Heuristic Engine** | Cosine similarity (N vs N-2), CLOSED/HALF-OPEN/OPEN state machine, configurable thresholds, unit-tested in isolation with fake embeddings |
| **4. Intervention layer** | Forced-summarization override injector; human-in-the-loop escalation stub |
| **5. Small real experiment** | Run on ~10–20 real queries, confirm the breaker actually trips and the baseline actually crashes, log real token numbers |
| **6. Full experiment harness** | Scale to 40+ queries (thrashing + normal), rerun the Assignment-2-style analysis with **real** numbers instead of simulated ones |
| **7. Ablations & baselines** | Vary thresholds/window size, disable HALF-OPEN, compare against a GPTCache-style exact-match baseline |
| **8. Dashboard (stretch)** | React+TS visualizer showing the trace timeline and the exact millisecond of intervention |
| **9. Infra polish** | Full CI, tests coverage, Prometheus/Grafana if time allows, deploy demo |
| **10. Write-up** | Final report / paper draft, ready for conference or journal submission per your synopsis's target venues |

---

## 4. The 5-Day / 50% Plan — What You're Actually Building This Week

**Target for Friday: a real, running pipeline where you can show your teacher a live query that crashes the baseline and a live query that the circuit breaker gracefully saves — with real logged token numbers, not simulated ones.** That is a legitimate 50%, because it proves the entire core claim of your synopsis works.

| Day | Focus | Concrete deliverable |
|---|---|---|
| **Day 1** | Repo scaffold + baseline pipeline | `apps/engine` running locally; LangGraph Researcher→Critic→Writer graph over a small document corpus; one crafted adversarial question reliably reproduces a `GraphRecursionError` crash at `recursion_limit=6` |
| **Day 2** | Middleware Interceptor | Every node call wrapped by `hooks.py`; `embeddings.py` computes a vector per attempt; `token_counter.py` reads real `usage` from the LLM API response; `log_store.py` writes rows (iteration_id, embedding, tokens, node, timestamp) to SQLite |
| **Day 3** | Heuristic Engine (build + unit test in isolation) | `similarity.py` computes cosine(N, N-2); `state_machine.py` implements CLOSED (<0.85) / HALF-OPEN (0.85–0.95) / OPEN (≥0.95); `tests/test_state_machine.py` passes using fake embedding vectors — **no LLM calls needed to prove this part works** |
| **Day 4** | Wire it all together + intervention | Heuristic engine subscribes to interceptor's log stream live during a run; on OPEN, `override.py` injects the forced-summarization prompt and the run returns partial output instead of crashing; run the *same* adversarial question from Day 1 through both configurations and confirm: baseline crashes, breaker doesn't |
| **Day 5** | Small real experiment + packaging | `scripts/run_experiment.py` runs both configs on ~10–20 real queries (mix of thrashing-inducing and normal); `generate_report.py` produces a real Table 1 / Table 2 (same shape as your Assignment 2, but with numbers from an actual run); Dockerize `apps/engine`; push clean repo to GitHub; prep a 2-minute live demo |

### After the 50% milestone — where to pick up (Day 6 onward)

1. **Scale the dataset** from ~10–20 queries to the full 40 (or more), rerun `run_experiment.py`, and replace every simulated number in your Assignment 2 with the real ones — same tables, same statistical tests, just real data now.
2. **Tune thresholds** — right now 0.85/0.95 are guesses from the synopsis; run the ablation (vary threshold, vary window size, toggle HALF-OPEN off) and pick the values your real data supports.
3. **Add the exact-match/GPTCache-style baseline** so your comparison table has three arms, not two, matching what you already promised in the literature review.
4. **Build the human-in-the-loop escalation path** for real (right now it can stay a stub/log line).
5. **Build the dashboard** (`apps/dashboard`) once the engine's API is stable — this is presentation polish, not core proof.
6. **CI, tests coverage, and infra polish** (`infra/`, GitHub Actions) to match your friend's repo's maturity.
7. **Write the final report/paper** using your real results.

---

## 5. How You'll Actually Check Token Usage on GPT / Claude

Two different mechanisms, used for two different purposes in your system:

**A. The real, billed number — read it off the API response.**
Every OpenAI and Anthropic API call returns a `usage` object in the response itself:
- OpenAI: `response.usage.prompt_tokens`, `response.usage.completion_tokens`
- Anthropic: `response.usage.input_tokens`, `response.usage.output_tokens`

This is the actual number you were billed for — not an estimate. Your `token_counter.py` in the interceptor should just read this field off every LLM call and log it. This is what feeds your "tokens consumed" column in every results table.

**B. A pre-call estimate — use a tokenizer library, when you need a number *before* calling the API.**
Sometimes you want to estimate token count without spending a call (e.g., checking if a prompt fits a context window, or in the HALF-OPEN lightweight verifier if it's a local/non-billed model):
- OpenAI models → `tiktoken` (exact match to their tokenizer, official library)
- Claude models → Anthropic's Python SDK has a built-in token-counting method for Claude's tokenizer — also exact, not something you approximate yourself

**One thing to keep separate:** your sentence-transformer embedding model has its own tokenizer, completely unrelated to the LLM's billing tokenizer. Don't mix "tokens the embedding model saw" with "tokens OpenAI/Anthropic billed you for" — they're two different numbers, and only the second one belongs in your cost-savings math.
