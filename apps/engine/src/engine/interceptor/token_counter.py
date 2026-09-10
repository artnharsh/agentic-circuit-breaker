"""
Middleware Interceptor — token_counter.py

Extracts the actual billed token counts from LLM API responses.

IMPORTANT distinction (from PLAN.md Section 5):
  A. Billed tokens  → read from response.usage (THIS module)
  B. Pre-call estimates → tokenizer.py (tiktoken / Anthropic SDK)

Only the numbers from THIS module belong in your cost-savings math.
The embedding model has its own separate tokenizer — do not mix them.

Supported providers:
  - OpenAI / LangChain ChatOpenAI  → response.usage_metadata (via LangChain)
  - Anthropic / LangChain ChatAnthropic → response.usage_metadata
  - MockChatModel → uses usage_metadata set by the mock itself
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token counts extracted from a single LLM API response."""

    prompt_tokens: int = 0
    """Tokens in the input/prompt (what you send to the LLM)."""

    completion_tokens: int = 0
    """Tokens in the output/completion (what the LLM returns)."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


ZERO_USAGE = TokenUsage(0, 0)


def extract_token_usage(response: AIMessage) -> TokenUsage:
    """
    Extract token usage from a LangChain AIMessage response.

    LangChain normalizes the provider-specific usage fields into
    response.usage_metadata with keys:
      - "input_tokens"  (maps to OpenAI prompt_tokens / Anthropic input_tokens)
      - "output_tokens" (maps to OpenAI completion_tokens / Anthropic output_tokens)

    Falls back to 0 for any missing field (e.g., streaming responses).

    Args:
        response: The AIMessage returned by llm.invoke()

    Returns:
        TokenUsage with prompt and completion counts.
    """
    if response is None:
        return ZERO_USAGE

    # LangChain's unified usage_metadata interface (works for OpenAI, Anthropic, and Mock)
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        prompt = int(usage_meta.get("input_tokens", 0))
        completion = int(usage_meta.get("output_tokens", 0))
        return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)

    # Fallback: try raw response_metadata (provider-specific)
    response_meta = getattr(response, "response_metadata", None)
    if response_meta and isinstance(response_meta, dict):
        # OpenAI format
        token_usage = response_meta.get("token_usage", {})
        if token_usage:
            return TokenUsage(
                prompt_tokens=int(token_usage.get("prompt_tokens", 0)),
                completion_tokens=int(token_usage.get("completion_tokens", 0)),
            )
        # Anthropic format
        usage = response_meta.get("usage", {})
        if usage:
            return TokenUsage(
                prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)),
            )

    logger.debug("Could not extract token usage from response — returning zeros.")
    return ZERO_USAGE


def sum_token_usages(usages: list[TokenUsage]) -> TokenUsage:
    """Sum a list of TokenUsage objects into one total."""
    result = ZERO_USAGE
    for u in usages:
        result = result + u
    return result
