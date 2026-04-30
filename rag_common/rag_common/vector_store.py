"""
Vector store adapter layer for RAG pipelines.

Design
------
`VectorStoreProtocol` is a Python structural Protocol — any class that
implements its four methods satisfies it without inheriting from anything.
Callers type-hint against the Protocol so the concrete store can be swapped
in one line (FAISS → Qdrant → Pinecone → LanceDB) with zero changes to
retrieval or evaluation code.

Implementations provided here
------------------------------
FAISSVectorStore   — local, file-backed; uses IndexFlatIP with L2-normalised
                     embeddings so inner-product == cosine similarity.
InMemoryVectorStore — brute-force cosine; no FAISS dependency; intended for
                      unit tests and tiny prototypes only.

Adding a new backend
--------------------
Implement the four Protocol methods and annotate your store as
`VectorStoreProtocol` at the call site — no base class required:

    class QdrantVectorStore:
        def add(self, chunks, embeddings): ...
        def search(self, query_embedding, top_k): ...
        def save(self, path): ...
        def load(self, path): ...
        def __len__(self): ...

    store: VectorStoreProtocol = QdrantVectorStore(...)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from rag_common.models import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Protocol (the "Interface")
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Structural interface for all vector store backends.

    `runtime_checkable` lets callers use `isinstance(store, VectorStoreProtocol)`
    at runtime — useful for validation in CLI entry-points and tests.
    """

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Index `chunks` with their corresponding `embeddings`.

        Args:
            chunks:     list of N Chunk objects
            embeddings: float32 array of shape (N, D); must be L2-normalised
                        if the backend uses inner product for cosine similarity.
        """
        ...

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        """
        Return the top-k most similar chunks for `query_embedding`.

        Args:
            query_embedding: 1-D float32 array of shape (D,); caller is
                             responsible for normalising it to unit length.
            top_k:           number of results to return.

        Returns:
            List of RetrievalResult sorted by descending score (best first).
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index and chunk metadata to `path` (file or directory)."""
        ...

    def load(self, path: str) -> None:
        """Restore the index and chunk metadata from `path`."""
        ...

    def __len__(self) -> int:
        """Return the number of chunks currently indexed."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Avoids division by zero for zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# FAISSVectorStore
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    FAISS-backed vector store using IndexFlatIP (exact inner-product search).

    Embeddings are L2-normalised on `add` and `search` so that inner-product
    equals cosine similarity. IndexFlatIP is chosen over IndexFlatL2 because:
    - Results are naturally in [0, 1] after normalisation (cosine range).
    - Scores can be directly used in HybridRetriever fusion without extra rescaling.
    - For corpora < ~1 M chunks, exhaustive flat search is fast enough.

    File layout when saved:
        <path>/index.faiss   — FAISS binary index
        <path>/chunks.pkl    — pickled list[Chunk] in insertion order
    """

    def __init__(self) -> None:
        self._index = None   # lazily initialised on first add()
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import faiss

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )

        vecs = _l2_normalise(embeddings.astype(np.float32))
        dim = vecs.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(vecs)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if self._index is None or len(self._chunks) == 0:
            return []

        k = min(top_k, len(self._chunks))
        q = _l2_normalise(query_embedding.reshape(1, -1).astype(np.float32))
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                score=float(score),
                retriever_type="dense",
            ))
        return results

    def save(self, path: str) -> None:
        import faiss

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dest / "index.faiss"))
        with open(dest / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: str) -> None:
        import faiss

        src = Path(path)
        self._index = faiss.read_index(str(src / "index.faiss"))
        with open(src / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# InMemoryVectorStore  (test / prototype use only)
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """
    Brute-force cosine similarity store backed by a plain numpy matrix.

    No FAISS dependency — suitable for unit tests, tiny corpora, and CI
    environments where installing FAISS would be burdensome. Not intended
    for production use; O(N) search complexity.

    `save` / `load` use numpy `.npz` for embeddings and pickle for chunks,
    keeping the file format human-inspectable and dependency-free.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[np.ndarray] = []  # list of 1-D arrays

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )
        vecs = _l2_normalise(embeddings.astype(np.float64))
        self._chunks.extend(chunks)
        self._embeddings.extend(vecs)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if not self._chunks:
            return []

        q = _l2_normalise(query_embedding.reshape(1, -1).astype(np.float64))[0]
        matrix = np.stack(self._embeddings)          # (N, D)
        scores = matrix @ q                          # cosine sim (already normalised)

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                retriever_type="dense",
            )
            for i in top_indices
        ]

    def save(self, path: str) -> None:
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        np.save(str(dest / "embeddings.npy"), np.stack(self._embeddings))
        with open(dest / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: str) -> None:
        src = Path(path)
        matrix = np.load(str(src / "embeddings.npy"))
        self._embeddings = list(matrix)
        with open(src / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

    def __len__(self) -> int:
        return len(self._chunks)
