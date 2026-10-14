"""
Run Experiment Script — scripts/run_experiment.py.

Batch runner that executes both baseline (unprotected) and treatment
(circuit-breaker protected) configurations against a benchmark query set,
logs detailed token counts and per-node localization, and outputs results.

Usage:
    python scripts/run_experiment.py --queries data/adversarial_queries/thrashing.jsonl
    python scripts/run_experiment.py --queries data/adversarial_queries/normal.jsonl
    python scripts/run_experiment.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


async def run_single_query(
    client: httpx.AsyncClient,
    query: str,
    use_circuit_breaker: bool,
) -> dict[str, Any]:
    """Execute a single query against /runs."""
    start_time = time.perf_counter()
    resp = await client.post(
        "/runs",
        json={"query": query, "use_circuit_breaker": use_circuit_breaker},
        timeout=120.0,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    if resp.status_code != 200:
        return {
            "status": "error",
            "error": f"HTTP {resp.status_code}: {resp.text}",
            "iterations": 0,
            "tokens_used": 0,
            "duration_ms": duration_ms,
            "circuit_breaker_triggered": False,
            "circuit_breaker_source": None,
        }

    data = resp.json()
    data["duration_ms"] = duration_ms
    return data


async def main_async(args: argparse.Namespace) -> None:
    query_file = Path(args.queries)
    if not query_file.exists():
        logger.error(f"Query file not found: {query_file}")
        sys.exit(1)

    queries: list[dict[str, Any]] = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    if args.limit:
        queries = queries[: args.limit]

    logger.info(f"Loaded {len(queries)} queries from {query_file}")

    # Set up client: remote HTTP or in-process ASGI
    if args.base_url:
        logger.info(f"Connecting to server at: {args.base_url}")
        client_cm = httpx.AsyncClient(base_url=args.base_url)
    else:
        logger.info("Running in-process via engine FastAPI app...")
        from engine.main import app
        from engine.storage.db import init_db

        await init_db()
        transport = httpx.ASGITransport(app=app)
        client_cm = httpx.AsyncClient(transport=transport, base_url="http://test")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"experiment_results_{timestamp}.jsonl"
    latest_path = out_dir / "experiment_results_latest.jsonl"

    results: list[dict[str, Any]] = []

    async with client_cm as client:
        for idx, item in enumerate(queries, 1):
            query = item["query"]
            label = item.get("label", "unknown")
            logger.info(f"\n[{idx}/{len(queries)}] Query ({label}): {query[:70]}...")

            # 1. Baseline Run (No Circuit Breaker)
            logger.info("  Running Baseline (use_circuit_breaker=False)...")
            baseline = await run_single_query(client, query, use_circuit_breaker=False)
            b_status = baseline.get("status")
            b_tokens = baseline.get("tokens_used") or 0
            b_iters = baseline.get("iterations", 0)
            b_time = baseline.get("duration_ms", 0.0)
            logger.info(
                f"  Baseline: status={b_status} | iters={b_iters} | tokens={b_tokens:,} | time={b_time:.0f}ms"
            )

            # 2. Treatment Run (With Circuit Breaker)
            logger.info("  Running Protected (use_circuit_breaker=True)...")
            treatment = await run_single_query(client, query, use_circuit_breaker=True)
            t_status = treatment.get("status")
            t_tokens = treatment.get("tokens_used") or 0
            t_iters = treatment.get("iterations", 0)
            t_time = treatment.get("duration_ms", 0.0)
            t_triggered = treatment.get("circuit_breaker_triggered", False)
            t_source = treatment.get("circuit_breaker_source")
            logger.info(
                f"  Protected: status={t_status} | iters={t_iters} | tokens={t_tokens:,} | "
                f"triggered={t_triggered} (by {t_source}) | time={t_time:.0f}ms"
            )

            # Metrics
            saved_tokens = max(0, b_tokens - t_tokens)
            savings_pct = (saved_tokens / b_tokens * 100.0) if b_tokens > 0 else 0.0

            result_entry = {
                "id": idx,
                "query": query,
                "label": label,
                "baseline": {
                    "status": b_status,
                    "iterations": b_iters,
                    "tokens_used": b_tokens,
                    "duration_ms": b_time,
                },
                "treatment": {
                    "status": t_status,
                    "iterations": t_iters,
                    "tokens_used": t_tokens,
                    "duration_ms": t_time,
                    "circuit_breaker_triggered": t_triggered,
                    "circuit_breaker_source": t_source,
                },
                "token_savings": saved_tokens,
                "token_savings_pct": round(savings_pct, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(result_entry)

    # Write output JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)

    # Update latest symlink or copy
    with open(latest_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)

    logger.info(f"\nExperiment complete. Saved {len(results)} rows to {out_path} and {latest_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("  EXPERIMENT SUMMARY TABLE")
    print("=" * 90)
    print(f"{'#':<3} | {'Type':<10} | {'Baseline (Tokens/Stat)':<24} | {'Protected (Tokens/Stat)':<26} | {'Saved Tokens (%)':<18}")
    print("-" * 90)
    for r in results:
        b_str = f"{r['baseline']['tokens_used']:,} ({r['baseline']['status']})"
        t_src = f" [by {r['treatment']['circuit_breaker_source']}]" if r['treatment']['circuit_breaker_source'] else ""
        t_str = f"{r['treatment']['tokens_used']:,} ({r['treatment']['status']}){t_src}"
        sav_str = f"{r['token_savings']:,} ({r['token_savings_pct']:.1f}%)"
        print(f"{r['id']:<3} | {r['label']:<10} | {b_str:<24} | {t_str:<26} | {sav_str:<18}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic Circuit Breaker experiment.")
    parser.add_argument(
        "--queries",
        type=str,
        default="data/adversarial_queries/thrashing.jsonl",
        help="Path to JSONL queries file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of running Engine service (optional).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to run.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
