"""
Middleware Interceptor — hooks.py (Day 2 implementation).

This module wraps each LangGraph node to log:
  - Per-iteration embeddings
  - Token counts (read from LLM API usage field)
  - State transitions
  - Timestamps

Stub for Day 1 — full implementation in Day 2.
"""

# TODO (Day 2): Implement node-wrapping hooks that:
#   1. Call embeddings.py to embed the research_notes before and after each node
#   2. Call token_counter.py to extract usage from the LLM response
#   3. Call log_store.py to write the trace row to SQLite
