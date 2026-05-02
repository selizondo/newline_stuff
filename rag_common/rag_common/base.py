"""
Abstract base classes for swappable RAG components.

All concrete implementations must subclass one of these ABCs so experiment
runners can substitute components via config without touching retrieval or
evaluation code.

Usage in downstream projects::

    from rag_common.base import BaseChunker, BaseEmbedder, BaseReranker

    class MyChunker(BaseChunker):
        def chunk(self, text, metadata=None):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rag_common.models import Chunk, RetrievalResult


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split `text` into chunks, merging `metadata` into each Chunk.metadata."""
        ...


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier used for cache keys and experiment IDs."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Returns:
            float32 ndarray of shape (N, D), L2-normalised.
        """
        ...


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return top-k most relevant chunks for `query`, best first."""
        ...


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score and reorder `results`, returning the best `top_k`."""
        ...


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for `prompt`."""
        ...
