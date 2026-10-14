"""
Single Query Runner — scripts/run_query.py.

Executes a single query and prints the complete formatted JSON response
along with the synthesized answer. Works both with a running server
(http://localhost:8000) or directly in-process if the server is offline.

Usage:
    python scripts/run_query.py --baseline
    python scripts/run_query.py --protected
    python scripts/run_query.py --query "What is solar energy?"
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx

DEFAULT_ADVERSARIAL_QUERY = (
    "What is the exact CO2 reduction achieved per country per year from 2015 to 2025, "
    "including specific tonnage figures broken down by sector (energy, transport, agriculture, "
    "industry) for each of the 195 UN member states?"
)


async def execute_query(query: str, use_circuit_breaker: bool) -> dict:
    mode_label = (
        "CIRCUIT BREAKER (PROTECTED)"
        if use_circuit_breaker
        else "BASELINE (UNPROTECTED)"
    )
    print(f"\n{'=' * 78}")
    print(f" RUNNING: {mode_label}")
    print(f"{'=' * 78}")
    print(f'Query: "{query}"\n')

    payload = {"query": query, "use_circuit_breaker": use_circuit_breaker}

    # 1. Try connecting to live server first if running
    try:
        async with httpx.AsyncClient(
            base_url="http://localhost:8000", timeout=2.0
        ) as client:
            resp = await client.get("/health")
            if resp.status_code == 200:
                print("[Connected to live engine server on http://localhost:8000]")
                resp = await client.post("/runs", json=payload, timeout=60.0)
                return resp.json()
    except (httpx.HTTPError, OSError):
        pass

    # 2. If server not running, execute in-process
    print("[Engine server not running on port 8000 — executing directly in-process]")
    from engine.main import app
    from engine.storage.db import init_db

    await init_db()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json=payload, timeout=60.0)
        return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single query and inspect raw output."
    )
    parser.add_argument(
        "--protected",
        action="store_true",
        help="Enable circuit breaker (default: baseline if omitted).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Disable circuit breaker (baseline).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_ADVERSARIAL_QUERY,
        help="Query to run.",
    )
    args = parser.parse_args()

    use_cb = bool(args.protected)
    if args.baseline:
        use_cb = False

    data = asyncio.run(execute_query(args.query, use_circuit_breaker=use_cb))

    # Print Full Formatted JSON
    print("\n--- RAW API RESPONSE JSON ---")
    print(json.dumps(data, indent=2))

    # Print Highlighted Outcome
    print("\n" + "=" * 78)
    print(" SUMMARY OF RETURNED OUTPUT:")
    print("=" * 78)
    print(f"Status:                    {data.get('status')}")
    print(f"Circuit Breaker Triggered: {data.get('circuit_breaker_triggered')}")
    print(f"Triggered By:              {data.get('circuit_breaker_source')}")
    print(f"Iterations:                {data.get('iterations')}")
    print(f"Tokens Used:               {data.get('tokens_used')}")

    if data.get("error"):
        print(f"\nError: {data.get('error')}")

    if data.get("output"):
        print(f"\nFinal Synthesized Output:\n\n{data.get('output')}")
    else:
        print("\nFinal Synthesized Output: NONE (Pipeline crashed with no answer)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
