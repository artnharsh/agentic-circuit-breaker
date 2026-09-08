"""
Middleware Interceptor — log_store.py (Day 2 implementation).

Writes trace rows (iteration_id, embedding, tokens, node, timestamp) to SQLite.
These rows are what the heuristic engine reads to compute cosine similarity.

Stub for Day 1 — full implementation in Day 2 alongside storage/models.py and storage/db.py.
"""

# TODO (Day 2): Implement LogStore class that:
#   1. Accepts a DB session from storage/db.py
#   2. Writes IterationLog rows after each node call
#   3. Provides get_recent_embeddings(run_id, n) -> list[list[float]]
