"""
Run Experiment Script — scripts/run_experiment.py (Day 5 implementation).

Batch runner that executes both baseline and circuit-breaker configurations
against a set of queries and logs the results.

Usage:
    uv run python scripts/run_experiment.py --queries data/adversarial_queries/thrashing.jsonl
    uv run python scripts/run_experiment.py --queries data/adversarial_queries/normal.jsonl

Stub for Day 1 — full implementation in Day 5.
"""

# TODO (Day 5): Implement:
#   1. Load queries from --queries JSONL file
#   2. For each query, run POST /runs with use_circuit_breaker=False (baseline)
#   3. For each query, run POST /runs with use_circuit_breaker=True (treatment)
#   4. Log results to data/experiment_results_{timestamp}.jsonl
#   5. Print a summary table (matches Table 1 shape from Assignment 2)
