"""
Middleware Interceptor — embeddings.py

Wraps sentence-transformers to produce a fixed-size vector for each
Researcher reasoning attempt. These vectors are what the Heuristic Engine
uses to compute cosine similarity (N vs N-2) to detect semantic thrashing.

Model: all-MiniLM-L6-v2
  - 384-dimensional output
  - Runs entirely on CPU, no GPU required
  - Fast: ~10ms per short text on modern hardware
  - Free: no API calls, no billing

The model is loaded once as a module-level singleton (load takes ~2s on first call).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """
    Load and cache the sentence-transformer model.
    First call takes ~1-2 seconds; subsequent calls are instant.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {_MODEL_NAME} ...")
        model = SentenceTransformer(_MODEL_NAME)
        dim = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
        logger.info(f"Embedding model loaded. Output dim: {dim() if callable(dim) else dim}")
        return model
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: uv sync  (it should already be in pyproject.toml)"
        )


class EmbeddingService:
    """
    Thin wrapper around sentence-transformers.

    Usage:
        svc = EmbeddingService()
        vec = svc.embed("The Researcher found relevant information about solar energy.")
        # vec is a list[float] of length 384
    """

    def embed(self, text: str) -> list[float]:
        """
        Embed a text string into a 384-dimensional vector.

        Args:
            text: The text to embed (typically research_notes from the Researcher node).

        Returns:
            A normalized list[float] of length 384.
        """
        if not text or not text.strip():
            # Return a zero vector for empty text
            return [0.0] * 384

        model = _get_model()
        # encode() returns a numpy array; convert to plain Python list for JSON-serializability
        vec: np.ndarray = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in one batch call (more efficient than looping).

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dimensional float vectors.
        """
        if not texts:
            return []

        model = _get_model()
        vecs: np.ndarray = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors (384 for all-MiniLM-L6-v2)."""
        return 384


# Module-level singleton — import and use directly
embedding_service = EmbeddingService()
