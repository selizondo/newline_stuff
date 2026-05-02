"""
SentenceTransformers embedder with two-level disk cache.

Cache layout (mirrors P3 pattern):
    data/embed_cache/{model_label}/{chunk_label}.pkl
        → {"chunk_ids": list[str], "embeddings": np.ndarray}

Cache is invalidated when stored chunk IDs are no longer a subset of the
current chunk set (same logic as P3's embed_cache invalidation).

SentenceTransformers has built-in batching via encode(batch_size=...) so
no ThreadPoolExecutor is needed — the library handles parallelism internally.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_common.models import Chunk
from src.base import BaseEmbedder

# Dimension map for known models — used to report dimension without loading the model.
_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2":           384,
    "all-mpnet-base-v2":           768,
    "multi-qa-MiniLM-L6-cos-v1":  384,
}


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Wraps sentence_transformers.SentenceTransformer with disk caching.

    The model is lazy-loaded on first use so that importing this module
    does not pay the model-load cost (~0.5–2s) in scripts that only
    need config/model metadata.

    Args:
        model_name: SentenceTransformers model identifier
        cache_dir:  root directory for embedding caches
        batch_size: passed to SentenceTransformer.encode(); 64 is a good
                    default for CPU; increase to 128–256 on GPU.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path = Path("data/embed_cache"),
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._batch_size = batch_size
        self._model = None       # lazy-loaded
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts, returning L2-normalised float32 (N, D) array.

        SentenceTransformers returns normalised embeddings by default when
        `normalize_embeddings=True`. We always set this so the output is
        consistent with FAISSVectorStore's IndexFlatIP (inner-product == cosine).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Cached chunk embedding (used by ingestion pipeline)
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk], chunk_label: str) -> np.ndarray:
        """
        Embed chunks with two-level disk cache.

        Returns (N, D) float32 ndarray aligned with `chunks`.
        Cache miss or stale cache → re-embeds and writes cache.
        """
        cache_path = self._cache_path(chunk_label)

        if cache_path.exists():
            cached = self._load_cache(cache_path)
            current_ids = {c.id_str() for c in chunks}
            if set(cached["chunk_ids"]).issubset(current_ids):
                return cached["embeddings"]

        embeddings = self.embed([c.content for c in chunks])
        self._save_cache(cache_path, [c.id_str() for c in chunks], embeddings)
        return embeddings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    def _cache_path(self, chunk_label: str) -> Path:
        # Sanitise model name for use as a directory name.
        model_dir = self._model_name.replace("/", "_")
        return self._cache_dir / model_dir / f"{chunk_label}.pkl"

    def _save_cache(
        self, path: Path, chunk_ids: list[str], embeddings: np.ndarray
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": chunk_ids, "embeddings": embeddings}, f)

    def _load_cache(self, path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)
