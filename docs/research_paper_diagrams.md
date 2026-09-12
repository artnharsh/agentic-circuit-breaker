# Agentic Circuit Breaker — Research Paper Diagrams

> Render PlantUML at: https://www.plantuml.com/plantuml/uml/
> Render Mermaid at: https://mermaid.live/

---

## Figure 1 — System Architecture (PlantUML)
*Use this as "Fig. 1" in your paper*

```plantuml
@startuml AgenticCircuitBreaker_Architecture
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Arial
skinparam defaultFontSize 11
skinparam ArrowColor #333333
skinparam ArrowThickness 1.5

skinparam component {
  BackgroundColor #F5F5F5
  BorderColor #666666
  FontColor #111111
}

skinparam package {
  BackgroundColor #EEF4FF
  BorderColor #2255AA
  FontColor #1a2744
}

skinparam rectangle {
  BackgroundColor #F5F5F5
  BorderColor #666666
}

title Figure 1. Architecture of the Agentic Circuit Breaker System

' ── Client ────────────────────────────────────────────────────
package "Client Layer" as CL #F0F0F0 {
  [User Query] as UQ
  [Experiment Scripts\n(run_experiment.py)] as ES
}

' ── API Gateway ───────────────────────────────────────────────
package "API Gateway (FastAPI)" as API #F0F0F0 {
  [POST /runs] as PR
  [GET /runs/{id}] as GR
  [Pydantic Schemas] as PS
}

' ── LangGraph Pipeline ────────────────────────────────────────
package "Agent Pipeline  (LangGraph StateGraph)" as PIPE #EEF4FF {

  package "Agent Nodes" as NODES #DDEEFF {
    [Researcher\nAgent] as RA
    [Critic\nAgent] as CA
    [Circuit Breaker\nCheck] as CBC #FFE4B5
    [Writer\nAgent] as WA
  }

  package "Middleware Interceptor" as MI #E8F5E9 {
    [Node Hook\nWrappers] as NHW
    [Embedding Service\nall-MiniLM-L6-v2\n(384-dim)] as EMB
    [Token Counter\n(billed tokens)] as TC
    [PendingLog\nAccumulator] as PLA
  }

  package "Heuristic Engine" as HE #FFF8E1 {
    [Cosine Similarity\ncos(N, N-2)] as CS
    [State Machine\nCLOSED / HALF-OPEN / OPEN\n(θ₁=0.85, θ₂=0.95)] as SM
    [Half-Open\nVerifier] as HOV
  }
}

' ── LLM Providers ─────────────────────────────────────────────
package "LLM Providers" as LLM #F5F5F5 {
  [OpenAI\ngpt-4o-mini] as OAI
  [Anthropic\nClaude Haiku] as ANT
  [Mock LLM\n(dev/test)] as MOCK
}

' ── Storage ───────────────────────────────────────────────────
package "Storage (SQLite → PostgreSQL)" as STORE #F5F5F5 {
  database "runs" as RT
  database "iteration_logs\n(embedding vectors)" as ILT
}

' ── Edges: Client → API ───────────────────────────────────────
UQ --> PR
ES --> PR
PR --> GR

' ── Edges: API → Pipeline ─────────────────────────────────────
PR --> RA : invoke(state)

' ── Edges: Agent flow ─────────────────────────────────────────
RA --> CA : research_notes
CA --> CBC : RETRY
CA --> WA : SATISFIED
CBC --> RA : CLOSED / HALF-OPEN
CBC --> WA : OPEN (breaker fires)
WA --> GR : final_output

' ── Edges: Interceptor ────────────────────────────────────────
NHW .> RA : wraps (transparent)
NHW .> CA : wraps (transparent)
NHW .> WA : wraps (transparent)
EMB --> PLA : embed(research_notes)
TC --> PLA : token_usage
PLA --> HE : embeddings[]

' ── Edges: Heuristic Engine ───────────────────────────────────
CS --> SM : similarity score
SM --> HOV : HALF-OPEN state
SM --> CBC : BreakerState

' ── Edges: LLM ───────────────────────────────────────────────
RA --> OAI : llm.invoke()
CA --> OAI : llm.invoke()
WA --> OAI : llm.invoke()
RA ..> MOCK : (dev mode)

' ── Edges: Storage ───────────────────────────────────────────
PLA --> ILT : flush_pending_logs()
PR --> RT : create_run()
WA --> RT : finalize_run()

@enduml
```

---

## Figure 2 — Circuit Breaker State Machine (PlantUML)
*Use this as "Fig. 2" in your paper*

```plantuml
@startuml CircuitBreaker_StateMachine
skinparam backgroundColor #FFFFFF
skinparam defaultFontName Arial
skinparam defaultFontSize 12

skinparam state {
  BackgroundColor #EEF4FF
  BorderColor #2255AA
  FontColor #111111
  ArrowColor #333333
}

skinparam state<<OPEN>> {
  BackgroundColor #FFEBEE
  BorderColor #CC0000
}

skinparam state<<HALF>> {
  BackgroundColor #FFF8E1
  BorderColor #E65100
}

title Figure 2. Circuit Breaker Finite State Machine

[*] --> CLOSED : Run begins

state "CLOSED" as CLOSED {
  CLOSED : Normal operation
  CLOSED : Agent making progress
  CLOSED : sim(N, N-2) < θ₁
}

state "HALF-OPEN" as HALFOPEN <<HALF>> {
  HALFOPEN : Suspected thrashing
  HALFOPEN : One warning given
  HALFOPEN : Verifier runs
}

state "OPEN" as OPEN <<OPEN>> {
  OPEN : Thrashing confirmed
  OPEN : Breaker fires
  OPEN : Forced summarization
}

CLOSED --> CLOSED : sim < θ₁ (0.85)\nAgent progressing
CLOSED --> HALFOPEN : θ₁ ≤ sim < θ₂\n(0.85 ≤ sim < 0.95)
CLOSED --> OPEN : sim ≥ θ₂ (0.95)\nFast-path trigger

HALFOPEN --> CLOSED : sim < θ₁ (0.85)\nAgent recovered
HALFOPEN --> OPEN : sim ≥ θ₁ again\nThrashing confirmed

OPEN --> [*] : Intervention\nPartial answer returned\nTokens saved

note right of HALFOPEN
  Verifier combines:
  • Cosine similarity score
  • Recursion budget progress
  • Token consumption rate
  → Confidence score ∈ [0,1]
end note

note right of OPEN
  Writer receives:
  circuit_breaker_triggered = True
  → Forced-summarization prompt
  → Honest partial answer
  → No GraphRecursionError
end note

@enduml
```

---

## Figure 3 — Agent Pipeline Flow (Mermaid — for GitHub/Notion)
*Simpler flow diagram, good for presentations*

```mermaid
flowchart TD
    Q([User Query]) --> API[FastAPI POST /runs]
    API --> R

    subgraph PIPELINE["LangGraph Agent Pipeline"]
        R["🔍 Researcher Agent\nRAG retrieval over corpus\nProduces: research_notes"]
        C["⚖️ Critic Agent\nEvaluates research quality\nVerdict: RETRY / SATISFIED"]
        B["⚡ Circuit Breaker Check\ncos sim N vs N−2\nFSM: CLOSED / HALF-OPEN / OPEN"]
        W_NORMAL["✍️ Writer Agent\nNormal mode\nComprehensive answer"]
        W_FORCED["✍️ Writer Agent\nForced-summarization mode\nPartial answer + limitations"]
    end

    R -->|research_notes| C
    C -->|SATISFIED| W_NORMAL
    C -->|RETRY| B
    B -->|CLOSED or HALF-OPEN\nsim < 0.95| R
    B -->|OPEN: sim ≥ 0.95\n🔥 Breaker fires| W_FORCED

    W_NORMAL --> OUT([Final Answer])
    W_FORCED --> OUT2([Partial Answer\n+ honest limitations])

    subgraph INT["Middleware Interceptor (transparent)"]
        E["Embedding Service\nall-MiniLM-L6-v2, 384-dim"]
        T["Token Counter\nbilled usage_metadata"]
        L["Log Store\niteration_logs table"]
    end

    R & C & W_NORMAL & W_FORCED -.->|observed| INT

    style B fill:#ff9800,color:#000,stroke:#e65100
    style W_FORCED fill:#ffebee,stroke:#cc0000
    style PIPELINE fill:#eef4ff,stroke:#2255aa
    style INT fill:#e8f5e9,stroke:#2e7d32
```

---

## How to Include in Your Paper

### LaTeX (IEEE format):
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/architecture.png}
  \caption{System architecture of the Agentic Circuit Breaker. The
           middleware interceptor observes each agent node transparently,
           computing sentence embeddings per iteration. The heuristic engine
           detects semantic thrashing via cosine similarity and drives a
           three-state FSM (CLOSED/HALF-OPEN/OPEN). When the breaker opens,
           the Writer agent produces a graceful partial answer instead of
           crashing.}
  \label{fig:architecture}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.85\columnwidth]{figures/state_machine.png}
  \caption{Circuit Breaker Finite State Machine. Thresholds $\theta_1=0.85$
           and $\theta_2=0.95$ are configurable via environment variables.
           The HALF-OPEN state prevents false positives by requiring two
           consecutive high-similarity readings before firing.}
  \label{fig:fsm}
\end{figure}
```

### Caption for Fig. 1:
> *Fig. 1. System architecture of the Agentic Circuit Breaker middleware. Dashed lines indicate transparent observation by the interceptor layer. The circuit breaker check node (highlighted) is the key contribution, preventing GraphRecursionError by detecting semantic thrashing via cosine similarity of sentence embeddings.*

### Caption for Fig. 2:
> *Fig. 2. Three-state finite state machine governing circuit breaker behavior. θ₁ = 0.85 and θ₂ = 0.95 are the HALF-OPEN and OPEN cosine similarity thresholds respectively, configurable via environment variables.*
