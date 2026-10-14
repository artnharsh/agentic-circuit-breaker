"""
Pluggable embedding service for the Agentic Circuit Breaker.

Supports two modes:
  1. Built-in: uses sentence-transformers (default: all-MiniLM-L6-v2)
  2. Custom: user provides their own embed_fn(text) -> list[float]

The model is loaded lazily on first embed() call and cached.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Cache loaded models at module level (keyed by model name)
_MODEL_CACHE: dict[str, object] = {}


def _load_sentence_transformer(model_name: str):
    """Load and cache a sentence-transformer model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for built-in embeddings. "
            "Install it: pip install sentence-transformers"
        )

    logger.info(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # Get embedding dimension
    dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
    dim = dim_fn() if callable(dim_fn) else "unknown"
    logger.info(f"Embedding model loaded. Output dim: {dim}")

    _MODEL_CACHE[model_name] = model
    return model


class EmbeddingService:
    """
    Produces fixed-size embedding vectors from text.

    These vectors are what the circuit breaker uses to compute cosine
    similarity between agent outputs across iterations.

    Usage with built-in model:
        embedder = EmbeddingService()
        vec = embedder.embed("The agent found relevant information.")
        # vec is a list[float] of length 384

    Usage with custom function:
        embedder = EmbeddingService(custom_fn=my_embed_function)
        vec = embedder.embed("Custom embedding.")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        custom_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._model_name = model_name
        self._custom_fn = custom_fn

    def embed(self, text: str) -> list[float]:
        """
        Embed a text string into a fixed-dimensional vector.

        Args:
            text: The text to embed.

        Returns:
            A normalized list[float] vector.
        """
        if not text or not text.strip():
            # Return a zero vector — will produce 0.0 similarity
            dim = self._get_dim()
            return [0.0] * dim

        if self._custom_fn is not None:
            return self._custom_fn(text)

        model = _load_sentence_transformer(self._model_name)
        vec: np.ndarray = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in one batch call (more efficient than looping).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self._custom_fn is not None:
            return [self._custom_fn(t) for t in texts]

        model = _load_sentence_transformer(self._model_name)
        vecs: np.ndarray = model.encode(
            texts, normalize_embeddings=True, batch_size=32
        )
        return vecs.tolist()

    def _get_dim(self) -> int:
        """Best-effort dimension detection for zero-vector fallback."""
        if self._custom_fn is not None:
            # Can't know dimension of custom fn without calling it
            return 384  # safe default

        model = _load_sentence_transformer(self._model_name)
        dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
        return dim_fn() if callable(dim_fn) else 384
