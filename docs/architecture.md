# Agentic Circuit Breaker — Architecture Diagrams

---

## Diagram 1 — Full System Architecture (Component View)

```mermaid
graph TB
    subgraph CLIENT["🖥️ Client / Researcher"]
        HTTP["HTTP Request\nPOST /runs"]
        SCRIPTS["scripts/\nrun_experiment.py\ngenerate_report.py"]
    end

    subgraph API["⚡ API Layer — FastAPI"]
        HEALTH["GET /health"]
        RUNS["POST /runs"]
        RUNS_ID["GET /runs/{id}"]
        SCHEMAS["Pydantic Schemas\nRunRequest / RunResponse"]
    end

    subgraph PIPELINE["🤖 LangGraph Pipeline — AgentState"]
        direction TB

        subgraph NODES["Agent Nodes"]
            RESEARCHER["Researcher Node\nRAG over corpus docs\nProduces research_notes"]
            CRITIC["Critic Node\nEvaluates research_notes\nVerdict: RETRY / SATISFIED"]
            BREAKER["BreakerCheck Node\nRuns Heuristic Engine\nDecides: continue / fire"]
            WRITER["Writer Node\nNormal: compose answer\nForced: partial summary"]
        end

        START([START]) --> RESEARCHER
        RESEARCHER --> CRITIC
        CRITIC -->|SATISFIED| WRITER
        CRITIC -->|RETRY| BREAKER
        BREAKER -->|CLOSED / HALF-OPEN| RESEARCHER
        BREAKER -->|OPEN 🔥| WRITER
        WRITER --> STOP([END])
    end

    subgraph INTERCEPTOR["🔍 Middleware Interceptor"]
        HOOKS["hooks.py\nWraps every node call\nAccumulates PendingLog list"]
        EMBED["embeddings.py\nall-MiniLM-L6-v2\n384-dim vectors"]
        TOKENS["token_counter.py\nReads billed tokens\nfrom LLM usage_metadata"]
        LOGSTORE["log_store.py\nwrite: create_run, log_iteration\nread: get_researcher_embeddings"]
    end

    subgraph HEURISTIC["📐 Heuristic Engine"]
        SIM["similarity.py\ncosine sim (N vs N-2)\nnumpy dot product"]
        FSM["state_machine.py\nCLOSED → HALF-OPEN → OPEN\nThresholds: 0.85 / 0.95"]
        VERIFY["verifier.py\nSecondary check in HALF-OPEN\nSimilarity + token budget score"]
    end

    subgraph INTERVENTION["🛡️ Intervention Layer"]
        OVERRIDE["override.py\nForced-summarization strategy\nInjects circuit_breaker_triggered=True"]
        ESCALATION["escalation.py\nToken savings calculation\nHuman-in-the-loop stub"]
    end

    subgraph STORAGE["💾 Storage"]
        DB["SQLite → PostgreSQL\naiosqlite + SQLAlchemy"]
        RUNS_T["runs table\nid, query, status, tokens\ncircuit_breaker_triggered"]
        ITER_T["iteration_logs table\nrun_id, node, iteration\nembedding_json, tokens\nbreaker_state"]
    end

    subgraph LLM["🧠 LLM Providers"]
        MOCK["MockChatModel\nZero cost, deterministic\nDev / testing"]
        OPENAI["OpenAI\ngpt-4o-mini\nLLM_PROVIDER=openai"]
        ANTHROPIC["Anthropic\nClaude Haiku\nLLM_PROVIDER=anthropic"]
    end

    subgraph DASHBOARD["📊 Dashboard — Stretch Goal"]
        REACT["React + TypeScript + Vite"]
        TIMELINE["RunTimeline component"]
        BADGE["StateBadge component"]
        CHART["TokenCostChart component"]
    end

    subgraph INFRA["🐳 Infrastructure"]
        DOCKER["Docker + docker-compose\nengine + db"]
        CI["GitHub Actions CI\ntest + lint on push"]
        GRAFANA["Prometheus + Grafana\nMonitoring — optional"]
    end

    HTTP --> RUNS
    SCRIPTS --> RUNS
    RUNS --> PIPELINE
    PIPELINE -.->|"wrap_node()"| INTERCEPTOR
    INTERCEPTOR --> HEURISTIC
    HEURISTIC --> INTERVENTION
    INTERVENTION --> PIPELINE
    INTERCEPTOR -->|"flush_pending_logs()"| STORAGE
    RUNS -->|"finalize_run()"| STORAGE
    RESEARCHER & CRITIC & WRITER -->|"llm.invoke()"| LLM
    STORAGE --> RUNS_T & ITER_T
    RUNS_ID --> STORAGE
    DASHBOARD -->|"GET /runs"| API
    DOCKER --> PIPELINE
```

---

## Diagram 2 — Request Lifecycle (Sequence View)

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI POST /runs
    participant Graph as LangGraph Graph
    participant R as Researcher Node
    participant C as Critic Node
    participant BC as BreakerCheck Node
    participant W as Writer Node
    participant INT as Interceptor Hooks
    participant HE as Heuristic Engine
    participant DB as SQLite DB
    participant LLM as LLM Provider

    User->>API: POST /runs {query}
    API->>DB: create_run(run_id, query)
    API->>Graph: graph.invoke(state)

    loop Agent Loop (max recursion_limit times)
        Graph->>R: researcher_node(state)
        R->>LLM: llm.invoke([system, human])
        LLM-->>R: AIMessage + usage_metadata
        R-->>Graph: {research_notes, messages}
        Graph->>INT: wrap_node fires
        INT->>INT: embed(research_notes) → 384-dim vector
        INT->>INT: extract_token_usage(messages[-1])
        INT->>INT: append PendingLog to ctx

        Graph->>C: critic_node(state)
        C->>LLM: llm.invoke([system, human])
        LLM-->>C: RETRY or SATISFIED
        C-->>Graph: {critique, verdict}
        Graph->>INT: wrap_node fires → PendingLog

        alt Critic says SATISFIED
            Graph->>W: writer_node(state, normal mode)
        else Critic says RETRY
            Graph->>BC: breaker_check_node(state)
            BC->>HE: get embeddings from ctx.pending_logs
            HE->>HE: cosine_sim(embed[N], embed[N-1])
            HE->>HE: state_machine.update(similarity)

            alt similarity >= 0.95 → OPEN
                HE-->>BC: BreakerState.OPEN
                BC-->>Graph: {circuit_breaker_triggered: True}
                Graph->>W: writer_node(state, FORCED SUMMARIZATION)
            else similarity >= 0.85 → HALF-OPEN
                HE->>HE: verifier.verify(sim, iter, tokens)
                HE-->>BC: BreakerState.HALF_OPEN
                BC-->>Graph: {circuit_breaker_triggered: False}
                Note over Graph: Give agent one more chance
            else similarity < 0.85 → CLOSED
                HE-->>BC: BreakerState.CLOSED
                BC-->>Graph: {circuit_breaker_triggered: False}
            end
        end
    end

    Graph->>W: writer_node(state)
    W->>LLM: llm.invoke([system, human])
    LLM-->>W: final answer
    W-->>Graph: {final_output}

    API->>INT: flush_pending_logs(ctx, db)
    INT->>DB: INSERT iteration_logs (batch)
    API->>DB: finalize_run(status, tokens, breaker_triggered)
    API-->>User: RunResponse {run_id, status, output, tokens_used, tokens_saved}
```

---

## Diagram 3 — State Machine (Circuit Breaker FSM)

```mermaid
stateDiagram-v2
    [*] --> CLOSED: Run starts

    CLOSED --> CLOSED: similarity < 0.85\n(agent making progress)
    CLOSED --> HALF_OPEN: 0.85 ≤ similarity < 0.95\n(suspected thrashing)
    CLOSED --> OPEN: similarity ≥ 0.95\n(confirmed thrashing - fast path)

    HALF_OPEN --> CLOSED: similarity < 0.85\n(agent recovered)
    HALF_OPEN --> OPEN: similarity ≥ 0.85 again\n(thrashing confirmed)

    OPEN --> [*]: Breaker fires 🔥\nForced summarization\nPartial answer returned

    note right of CLOSED
        Normal operation
        Researcher making progress
    end note

    note right of HALF_OPEN
        One warning given
        Verifier runs secondary check
        (token budget + sim score)
    end note

    note right of OPEN
        Intervention triggered
        Writer uses forced-summary prompt
        Tokens saved vs baseline crash
    end note
```

---

## Diagram 4 — Data Model (Storage Layer)

```mermaid
erDiagram
    runs {
        string id PK "UUID"
        text query
        string status "running/completed/crashed/breaker_triggered"
        int iterations
        bool circuit_breaker_triggered
        int total_prompt_tokens
        int total_completion_tokens
        text final_output
        text error
        datetime created_at
        datetime updated_at
    }

    iteration_logs {
        int id PK "autoincrement"
        string run_id FK
        int iteration
        string node "researcher/critic/writer"
        text embedding_json "list[float] 384-dim"
        int embedding_dim
        int prompt_tokens
        int completion_tokens
        string breaker_state "CLOSED/HALF_OPEN/OPEN"
        text input_text "first 500 chars"
        text output_text "first 500 chars"
        datetime timestamp
    }

    runs ||--o{ iteration_logs : "has many"
```

---

## Diagram 5 — Folder Structure Map

```mermaid
graph LR
    ROOT["agentic-circuit-breaker/"]

    ROOT --> APPS["apps/"]
    ROOT --> DATA["data/"]
    ROOT --> SCRIPTS["scripts/"]
    ROOT --> DOCS["docs/"]
    ROOT --> INFRA2["infra/"]

    APPS --> ENGINE["engine/ ← core service"]
    APPS --> DASH["dashboard/ ← stretch goal"]

    ENGINE --> SRC["src/engine/"]
    SRC --> CONFIG["config.py"]
    SRC --> MAIN["main.py"]
    SRC --> API2["api/"]
    SRC --> GRAPH["agentic_graph/"]
    SRC --> INTERCEPT["interceptor/"]
    SRC --> HEUR["heuristic_engine/"]
    SRC --> INTERV["intervention/"]
    SRC --> STORE["storage/"]
    SRC --> LLMDIR["llm/"]

    DATA --> CORPUS["corpus/ — 5 docs"]
    DATA --> ADV["adversarial_queries/"]

    SCRIPTS --> EXP["run_experiment.py"]
    SCRIPTS --> REPORT["generate_report.py"]

    style ENGINE fill:#1a6b3c,color:#fff
    style GRAPH fill:#1a3d6b,color:#fff
    style INTERCEPT fill:#1a3d6b,color:#fff
    style HEUR fill:#1a3d6b,color:#fff
    style INTERV fill:#5a3d6b,color:#fff
    style DASH fill:#555,color:#aaa
    style INFRA2 fill:#555,color:#aaa
```
