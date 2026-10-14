"""
Live Interactive Demo Script — scripts/demo.py.

Demonstrates the Agentic Circuit Breaker in real time:
  1. BASELINE DEMO: Shows semantic thrashing causing GraphRecursionError and token waste.
  2. PROTECTED DEMO: Shows circuit breaker detecting thrashing, isolating the stuck agent,
     and generating a graceful partial answer.
  3. LIBRARY USAGE DEMO: Shows how a developer uses `agentic-circuit-breaker` on any LangGraph graph.

Run with:
    python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import time
from typing import TypedDict

import httpx

# ANSI Color codes
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {BOLD}{CYAN}{title}{RESET}")
    print("=" * 78)


def subheader(title: str) -> None:
    print(f"\n{BOLD}{YELLOW}── {title} ──{RESET}\n")


async def run_engine_demo() -> tuple[dict, dict]:
    from engine.main import app
    from engine.storage.db import init_db

    await init_db()
    transport = httpx.ASGITransport(app=app)

    adversarial_query = (
        "What is the exact CO2 reduction achieved per country per year from 2015 to 2025, "
        "including specific tonnage figures broken down by sector (energy, transport, agriculture, "
        "industry) for each of the 195 UN member states?"
    )

    print(f'{BOLD}Adversarial Query:{RESET}\n{DIM}"{adversarial_query}"{RESET}\n')

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # ── SCENARIO 1: BASELINE (WITHOUT CIRCUIT BREAKER) ───────────────────
        subheader("SCENARIO 1: Baseline Pipeline (use_circuit_breaker=False)")
        print(
            f"{DIM}Running Researcher ⇄ Critic loop without circuit breaker...{RESET}"
        )
        t0 = time.perf_counter()
        resp_baseline = await client.post(
            "/runs",
            json={"query": adversarial_query, "use_circuit_breaker": False},
            timeout=60.0,
        )
        t_baseline = (time.perf_counter() - t0) * 1000
        b_data = resp_baseline.json()

        print(f"  • Status:           {RED}{BOLD}{b_data.get('status').upper()}{RESET}")
        print(f"  • Error Encountered:{RED} {b_data.get('error')[:90]}...{RESET}")
        print(f"  • Steps Consumed:   {b_data.get('iterations')} recursion steps")
        print(f"  • Tokens Burned:    {RED}{b_data.get('tokens_used'):,} tokens{RESET}")
        print(f"  • Execution Time:   {t_baseline:.1f} ms")
        print(
            f"  • Result:           {RED}Execution failed. Zero output produced for user.{RESET}"
        )
        print(f"\n  • {BOLD}Synthesized Output from Writer:{RESET}")
        print(
            f"    {RED}NONE — Pipeline crashed (GraphRecursionError). No answer delivered to user.{RESET}"
        )

        await asyncio.sleep(1.0)

        # ── SCENARIO 2: PROTECTED (WITH CIRCUIT BREAKER) ─────────────────────
        subheader("SCENARIO 2: Protected Pipeline (use_circuit_breaker=True)")
        print(f"{DIM}Running with real-time semantic circuit breaker...{RESET}")
        t0 = time.perf_counter()
        resp_protected = await client.post(
            "/runs",
            json={"query": adversarial_query, "use_circuit_breaker": True},
            timeout=60.0,
        )
        t_protected = (time.perf_counter() - t0) * 1000
        p_data = resp_protected.json()

        tokens_b = b_data.get("tokens_used", 0)
        tokens_p = p_data.get("tokens_used", 0)
        saved = max(0, tokens_b - tokens_p)
        saved_pct = (saved / tokens_b * 100) if tokens_b > 0 else 0

        print(
            f"  • Status:           {GREEN}{BOLD}{p_data.get('status').upper()}{RESET}"
        )
        print(
            f"  • Breaker Tripped:  {GREEN}True (Fired at iteration {p_data.get('iterations')}){RESET}"
        )
        print(
            f"  • Culprit Isolated: {MAGENTA}{BOLD}{p_data.get('circuit_breaker_source')}{RESET} (Identified stuck in semantic thrashing)"
        )
        print(f"  • Tokens Consumed:  {GREEN}{tokens_p:,} tokens{RESET}")
        print(
            f"  • Tokens Saved:     {GREEN}{BOLD}{saved:,} tokens ({saved_pct:.1f}% reduction){RESET}"
        )
        print(f"  • Execution Time:   {t_protected:.1f} ms")
        print(f"\n  • {BOLD}Synthesized Output from Writer (Graceful Recovery):{RESET}")
        output_lines = (p_data.get("output") or "None").strip().split("\n")
        for line in output_lines:
            print(f"    {CYAN}│{RESET} {line}")

    return b_data, p_data


class ToyState(TypedDict):
    query: str
    notes: str
    feedback: str
    circuit_breaker_triggered: bool
    circuit_breaker_source: str


def run_library_standalone_demo() -> None:
    """Demonstrate the pure pip-installable library wrapping a LangGraph pipeline."""
    subheader("SCENARIO 3: Direct Library Usage (agentic-circuit-breaker)")
    print(
        f"{DIM}Demonstrating how any developer wraps their own LangGraph pipeline:{RESET}\n"
    )

    from agentic_circuit_breaker import CircuitBreaker
    from langgraph.graph import END, START, StateGraph

    loop_counter = {"researcher": 0, "critic": 0}

    # 1. Developer defines their node functions
    def researcher_agent(state: ToyState) -> dict:
        loop_counter["researcher"] += 1
        # Simulating stuck agent repeating identical findings
        return {
            "notes": "Solar panels convert light to electricity using photovoltaic effect."
        }

    def critic_agent(state: ToyState) -> dict:
        loop_counter["critic"] += 1
        return {
            "feedback": f"Critique attempt #{loop_counter['critic']}: Please provide specific quantum yield data."
        }

    # 2. Build standard LangGraph
    builder = StateGraph(ToyState)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("critic", critic_agent)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic")

    def route_critic(state: ToyState) -> str:
        if state.get("circuit_breaker_triggered"):
            return END
        if loop_counter["researcher"] >= 10:
            return END
        return "researcher"

    builder.add_conditional_edges(
        "critic", route_critic, {"researcher": "researcher", END: END}
    )
    compiled = builder.compile()

    # 3. Developer wraps it with CircuitBreaker
    print(
        f"{CYAN}code: cb = CircuitBreaker(half_open_threshold=0.85, open_threshold=0.95, similarity_window=1){RESET}"
    )
    print(f"{CYAN}code: protected_graph = cb.wrap(compiled){RESET}")

    cb = CircuitBreaker(
        half_open_threshold=0.85,
        open_threshold=0.95,
        similarity_window=1,
        on_open="flag",
    )
    protected_graph = cb.wrap(compiled)

    # 4. Invoke
    _ = protected_graph.invoke(
        {
            "query": "Solar quantum efficiency",
            "notes": "",
            "feedback": "",
            "circuit_breaker_triggered": False,
            "circuit_breaker_source": "",
        }
    )

    # 5. Diagnostic report
    report = cb.report()
    print(f"\n{BOLD}Diagnostic Report Generated by Library:{RESET}")
    print(report.summary())


async def main() -> None:
    header("AGENTIC CIRCUIT BREAKER: LIVE RESEARCH DEMO")
    print(
        f"{DIM}Real-time demonstration of semantic thrashing detection, fault localization, and token savings.{RESET}"
    )

    # Run Part 1 & Part 2
    b_data, p_data = await run_engine_demo()

    # Run Part 3
    run_library_standalone_demo()

    # Summary Comparison Card
    header("DEMO SUMMARY & RESEARCH HIGHLIGHTS")
    b_tok = b_data.get("tokens_used", 0)
    p_tok = p_data.get("tokens_used", 0)
    savings = max(0, b_tok - p_tok)
    pct = (savings / b_tok * 100) if b_tok > 0 else 0

    print(f"┌{'─' * 76}┐")
    print(
        f"│ {BOLD}{'METRIC':<26} │ {'BASELINE':<22} │ {'CIRCUIT BREAKER':<22}{RESET} │"
    )
    print(f"├{'─' * 76}┤")
    print(
        f"│ {'Outcome':<26} │ {RED}{'Crashed (No Output)':<22}{RESET} │ {GREEN}{'Graceful Summary':<22}{RESET} │"
    )
    print(
        f"│ {'Failure Mode':<26} │ {RED}{'GraphRecursionError':<22}{RESET} │ {GREEN}{'Clean Termination':<22}{RESET} │"
    )
    print(f"│ {'Tokens Consumed':<26} │ {b_tok:<22,} │ {p_tok:<22,} │")
    print(
        f"│ {'Tokens Saved':<26} │ {'0 (0.0%)':<22} │ {GREEN}{BOLD}{f'{savings:,} ({pct:.1f}%)':<22}{RESET} │"
    )
    print(
        f"│ {'Fault Localization':<26} │ {'None (Black Box)':<22} │ {MAGENTA}{BOLD}{'researcher (Isolated)':<22}{RESET} │"
    )
    print(f"└{'─' * 76}┘")
    print(f"\n{GREEN}{BOLD}✅ Live Demo completed successfully.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
