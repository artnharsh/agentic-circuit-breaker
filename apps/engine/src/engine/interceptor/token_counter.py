"""
Middleware Interceptor — token_counter.py (Day 2 implementation).

Reads the actual billed token count off the LLM API response's `usage` field.
This is the number used in cost-savings calculations.

Stub for Day 1 — full implementation in Day 2.

Notes (from PLAN.md Section 5):
  - OpenAI: response.usage.prompt_tokens + response.usage.completion_tokens
  - Anthropic: response.usage.input_tokens + response.usage.output_tokens
  - Do NOT mix these with embedding model tokens (separate tokenizer, not billed here)
"""

# TODO (Day 2): Implement extract_token_usage(response) -> dict that reads
#   the usage field from OpenAI/Anthropic API responses.
