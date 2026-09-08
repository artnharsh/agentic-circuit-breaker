"""
Token counting utilities — pre-call estimates.

Two separate concerns (do NOT mix these numbers):
  A. LLM billing tokens  → read off the API response's `usage` field (see interceptor/token_counter.py)
  B. Pre-call estimates  → this module, used for context-window checks and HALF-OPEN verifier

For OpenAI models: uses tiktoken (exact match to their tokenizer).
For Anthropic models: uses anthropic SDK's count_tokens (exact).
For mock: simple char/4 approximation.
"""

from __future__ import annotations

from functools import lru_cache

from engine.config import get_settings


def count_tokens(text: str) -> int:
    """
    Estimate token count for the given text string.
    Uses the appropriate tokenizer for the configured LLM provider.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        return _count_openai_tokens(text, settings.openai_model)
    elif provider == "anthropic":
        return _count_anthropic_tokens(text)
    else:
        # Mock / fallback: rough approximation (~4 chars per token)
        return max(1, len(text) // 4)


def _count_openai_tokens(text: str, model: str) -> int:
    """Use tiktoken for exact OpenAI token counts."""
    try:
        import tiktoken
        enc = _get_tiktoken_encoder(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@lru_cache(maxsize=4)
def _get_tiktoken_encoder(model: str):  # type: ignore[return]
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_anthropic_tokens(text: str) -> int:
    """
    Use Anthropic SDK's token counter.
    Falls back to char/4 approximation if SDK not available.
    """
    try:
        import anthropic
        settings = get_settings()
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.count_tokens(
            model=settings.anthropic_model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception:
        return max(1, len(text) // 4)
