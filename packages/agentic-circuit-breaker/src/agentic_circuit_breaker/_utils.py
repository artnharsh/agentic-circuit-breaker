"""
Internal utilities for the Agentic Circuit Breaker library.
"""

from __future__ import annotations

import logging

from .types import TokenUsage

logger = logging.getLogger(__name__)


def default_text_extractor(node_name: str, result: dict, state: dict) -> str:
    """
    Default strategy to extract text from a node's output for embedding.

    Tries multiple approaches in priority order:
      1. Common named output fields (output, content, text, research_notes, etc.)
      2. Last LangChain AIMessage content
      3. Concatenation of all string values in the result dict

    Args:
        node_name: Name of the node (for field-name hinting).
        result: The dict returned by the node function.
        state: The full graph state before this node ran.

    Returns:
        Extracted text string. May be empty if no text found.
    """
    # Priority 1: Common field names for agent outputs
    common_fields = (
        "output", "content", "text", "response", "answer",
        "research_notes", "critique", "final_output", "draft",
        "summary", "result", "reasoning", "analysis",
    )
    for field_name in common_fields:
        value = result.get(field_name)
        if isinstance(value, str) and value.strip():
            return value

    # Priority 2: Extract from LangChain messages
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1] if isinstance(messages, list) else messages
        content = getattr(last_msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content

    # Priority 3: Concatenate all string values
    parts: list[str] = []
    for key, value in result.items():
        if key == "messages":
            continue  # already handled above
        if isinstance(value, str) and value.strip():
            parts.append(value)

    return " ".join(parts) if parts else ""


def extract_token_usage_from_result(result: dict) -> TokenUsage:
    """
    Best-effort extraction of token usage from a node's result dict.

    Checks for LangChain's usage_metadata on AIMessage objects,
    then falls back to provider-specific formats.

    Args:
        result: The dict returned by the node function.

    Returns:
        TokenUsage (may be zero if no usage data found).
    """
    messages = result.get("messages", [])
    if not messages:
        return TokenUsage()

    last_msg = messages[-1] if isinstance(messages, list) else messages

    # LangChain's unified usage_metadata (works for OpenAI, Anthropic, Mock)
    usage_meta = getattr(last_msg, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        return TokenUsage(
            prompt_tokens=int(usage_meta.get("input_tokens", 0)),
            completion_tokens=int(usage_meta.get("output_tokens", 0)),
        )

    # Fallback: provider-specific response_metadata
    response_meta = getattr(last_msg, "response_metadata", None)
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

    return TokenUsage()
