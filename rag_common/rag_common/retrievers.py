"""
Retriever adapter layer for RAG pipelines.

All retrievers implement `RetrieverProtocol`:

    retriever.retrieve(query: str, top_k: int) -> list[RetrievalResult]

Retrievers provided
-------------------
BM25Retriever    — sparse keyword matching via rank-bm25
DenseRetriever   — embedding similarity via any VectorStoreProtocol backend
HybridRetriever  — score fusion of dense + BM25 with min-max normalisation

Score fusion details (HybridRetriever)
---------------------------------------
BM25 scores are raw, unbounded positive floats.
Dense scores are cosine similarities in [-1, 1].
Fusing without normalisation lets BM25 dominate (magnitudes are orders larger),
making hybrid effectively identical to BM25-only. Fix:

  1. Retrieve `fetch_k` candidates from each retriever independently.
  2. Take the union; assign 0.0 to any score missing from one side.
  3. Min-max normalise each score set to [0, 1] within the candidate pool.
     If all scores are equal (e.g. all zero), skip normalisation (already uniform).
  4. Fuse:  final = alpha * dense_norm + (1 - alpha) * bm25_norm
  5. Sort descending, return top_k.

`fetch_k` defaults to max(top_k * 3, 20) so the pool is large enough that
top-k after fusion is drawn from a meaningful candidate set.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from rag_common.models import Chunk, RetrievalResult
from rag_common.vector_store import VectorStoreProtocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RetrieverProtocol(Protocol):
    """Structural interface for all retriever backends."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return the top-k most relevant chunks for `query`, best first."""
        ...


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Sparse keyword retrieval using BM25Okapi (rank-bm25).

    Chunks are tokenised (lowercased whitespace split) at construction time.
    Retrieval tokenises the query the same way to keep the vocabulary consistent.

    BM25 scores are raw positive floats with no fixed upper bound. Do not
    compare them directly with dense cosine scores — always normalise before
    fusing in HybridRetriever.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenised = [chunk.content.lower().split() for chunk in chunks]
        # BM25Okapi raises if the corpus is empty; guard against that.
        self._bm25 = BM25Okapi(tokenised) if tokenised else None

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self._chunks or self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        k = min(top_k, len(self._chunks))
        # argpartition is O(N) vs argsort O(N log N); fine for large corpora.
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                retriever_type="bm25",
            )
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Embedding-based semantic retrieval backed by any VectorStoreProtocol.

    `embed_fn` accepts a list of strings and returns an (N, D) ndarray.
    Passing a list (rather than a single string) matches the batch interface
    used by both OpenAI and SentenceTransformers, so no wrapper is needed.

    The query embedding is L2-normalised inside the vector store's `search`
    method, so callers do not need to normalise before calling `retrieve`.
    """

    def __init__(
        self,
        store: VectorStoreProtocol,
        embed_fn: Callable[[list[str]], np.ndarray],
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        embedding = self._embed_fn([query])           # (1, D)
        query_vec = np.array(embedding).flatten()     # (D,)
        return self._store.search(query_vec, top_k)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Score fusion helpers
# ---------------------------------------------------------------------------

def _min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1]. Returns unchanged dict if all scores are equal."""
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    if span == 0.0:
        # All scores identical — uniform distribution, keep as-is (all become 0.5
        # would be arbitrary; returning the originals is safer for callers).
        return dict(scores)
    return {k: (v - lo) / span for k, v in scores.items()}


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Fuses dense and BM25 retrieval via min-max normalised score combination.

    Args:
        dense:   DenseRetriever instance (already loaded with an index)
        bm25:    BM25Retriever instance (built from the same chunk set)
        alpha:   weight for dense scores; (1 - alpha) for BM25.
                 alpha=1.0 → pure dense; alpha=0.0 → pure BM25.
        fetch_k: candidate pool size per retriever before fusion.
                 Defaults to max(top_k * 3, 20) when None.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.5,
        fetch_k: int | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._dense = dense
        self._bm25 = bm25
        self.alpha = alpha
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        fetch_k = self._fetch_k if self._fetch_k is not None else max(top_k * 3, 20)

        dense_results = self._dense.retrieve(query, fetch_k)
        bm25_results = self._bm25.retrieve(query, fetch_k)

        # Build score maps keyed by chunk ID string.
        dense_scores: dict[str, float] = {r.chunk.id_str(): r.score for r in dense_results}
        bm25_scores: dict[str, float] = {r.chunk.id_str(): r.score for r in bm25_results}
        chunk_by_id: dict[str, Chunk] = {
            r.chunk.id_str(): r.chunk
            for r in dense_results + bm25_results
        }

        # Union of all candidate IDs; missing side gets 0.0.
        all_ids = set(dense_scores) | set(bm25_scores)
        dense_full = {cid: dense_scores.get(cid, 0.0) for cid in all_ids}
        bm25_full = {cid: bm25_scores.get(cid, 0.0) for cid in all_ids}

        # Normalise each score set independently to [0, 1].
        dense_norm = _min_max_normalise(dense_full)
        bm25_norm = _min_max_normalise(bm25_full)

        # Fuse and rank.
        fused = {
            cid: self.alpha * dense_norm[cid] + (1.0 - self.alpha) * bm25_norm[cid]
            for cid in all_ids
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_by_id[cid],
                score=score,
                retriever_type="hybrid",
            )
            for cid, score in ranked
        ]
