"""
Unit tests for rag_common.retrievers.

DenseRetriever and HybridRetriever use InMemoryVectorStore so tests run
without a FAISS install. The embed_fn stub returns deterministic vectors
based on a fixed vocabulary so retrieval results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_common.models import Chunk, RetrievalResult
from rag_common.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrieverProtocol,
    _min_max_normalise,
)
from rag_common.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "The mitochondria produces ATP through oxidative phosphorylation.",
    "Photosynthesis converts sunlight into glucose in chloroplasts.",
    "Neural networks learn via gradient descent and backpropagation.",
    "DNA replication is semi-conservative and requires primase.",
    "Quantum entanglement allows non-local correlations between particles.",
]

DIM = 16


def _make_chunks(texts: list[str] | None = None) -> list[Chunk]:
    texts = texts or CORPUS
    return [Chunk(content=t, chunk_index=i) for i, t in enumerate(texts)]


def _deterministic_embed(sentences: list[str]) -> np.ndarray:
    """Hash-based embed: unique, reproducible, unit-norm."""
    rng = np.random.default_rng(abs(hash(sentences[0][:20])) % (2**31))
    vecs = rng.standard_normal((len(sentences), DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _build_dense_retriever(chunks: list[Chunk] | None = None) -> tuple[DenseRetriever, list[Chunk]]:
    chunks = chunks or _make_chunks()
    embeddings = _deterministic_embed([ch.content for ch in chunks])
    store = InMemoryVectorStore()
    store.add(chunks, embeddings)
    return DenseRetriever(store, _deterministic_embed), chunks


def _build_hybrid_retriever(
    chunks: list[Chunk] | None = None, alpha: float = 0.5
) -> HybridRetriever:
    chunks = chunks or _make_chunks()
    dense, _ = _build_dense_retriever(chunks)
    bm25 = BM25Retriever(chunks)
    return HybridRetriever(dense, bm25, alpha=alpha)


# ---------------------------------------------------------------------------
# Protocol checks
# ---------------------------------------------------------------------------

class TestRetrieverProtocol:
    def test_bm25_satisfies_protocol(self):
        assert isinstance(BM25Retriever(_make_chunks()), RetrieverProtocol)

    def test_dense_satisfies_protocol(self):
        dense, _ = _build_dense_retriever()
        assert isinstance(dense, RetrieverProtocol)

    def test_hybrid_satisfies_protocol(self):
        assert isinstance(_build_hybrid_retriever(), RetrieverProtocol)

    def test_incomplete_class_fails(self):
        class Bad:
            pass
        assert not isinstance(Bad(), RetrieverProtocol)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_returns_results(self):
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("mitochondria ATP", top_k=3)
        assert len(results) > 0

    def test_retriever_type_bm25(self):
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("mitochondria", top_k=3)
        assert all(res.retriever_type == "bm25" for res in results)

    def test_top_k_respected(self):
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("the", top_k=2)
        assert len(results) <= 2

    def test_top_k_capped_at_corpus_size(self):
        chunks = _make_chunks()
        r = BM25Retriever(chunks)
        results = r.retrieve("the", top_k=100)
        assert len(results) <= len(chunks)

    def test_scores_sorted_descending(self):
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("photosynthesis sunlight", top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_chunk_ranks_high(self):
        # Query contains exact terms from corpus[0] only.
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("mitochondria ATP oxidative phosphorylation", top_k=5)
        top_content = results[0].chunk.content
        assert "mitochondria" in top_content.lower()

    def test_empty_corpus_returns_empty(self):
        r = BM25Retriever([])
        assert r.retrieve("anything", top_k=5) == []

    def test_len(self):
        chunks = _make_chunks()
        assert len(BM25Retriever(chunks)) == len(chunks)

    def test_results_are_retrieval_result_instances(self):
        r = BM25Retriever(_make_chunks())
        results = r.retrieve("neural network", top_k=3)
        assert all(isinstance(res, RetrievalResult) for res in results)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    def test_returns_results(self):
        dense, _ = _build_dense_retriever()
        results = dense.retrieve("mitochondria ATP", top_k=3)
        assert len(results) > 0

    def test_retriever_type_dense(self):
        dense, _ = _build_dense_retriever()
        results = dense.retrieve("photosynthesis", top_k=3)
        assert all(res.retriever_type == "dense" for res in results)

    def test_top_k_respected(self):
        dense, _ = _build_dense_retriever()
        results = dense.retrieve("test query", top_k=2)
        assert len(results) <= 2

    def test_scores_sorted_descending(self):
        dense, _ = _build_dense_retriever()
        results = dense.retrieve("neural network gradient", top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_ranks_first(self):
        # When the query embedding is identical to a stored embedding, that
        # chunk should be the top result (cosine sim = 1.0).
        chunks = _make_chunks()
        embeddings = _deterministic_embed([ch.content for ch in chunks])
        store = InMemoryVectorStore()
        store.add(chunks, embeddings)

        # Use a fixed embed_fn that always returns embeddings[2] for any query.
        def pinned_embed(texts):
            return embeddings[2:3]

        dense = DenseRetriever(store, pinned_embed)
        results = dense.retrieve("anything", top_k=5)
        assert results[0].chunk.content == chunks[2].content

    def test_empty_store_returns_empty(self):
        store = InMemoryVectorStore()
        dense = DenseRetriever(store, _deterministic_embed)
        assert dense.retrieve("test", top_k=5) == []

    def test_len_reflects_store(self):
        dense, chunks = _build_dense_retriever()
        assert len(dense) == len(chunks)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_returns_results(self):
        h = _build_hybrid_retriever()
        results = h.retrieve("mitochondria ATP", top_k=3)
        assert len(results) > 0

    def test_retriever_type_hybrid(self):
        h = _build_hybrid_retriever()
        results = h.retrieve("photosynthesis", top_k=3)
        assert all(res.retriever_type == "hybrid" for res in results)

    def test_top_k_respected(self):
        h = _build_hybrid_retriever()
        results = h.retrieve("the", top_k=2)
        assert len(results) <= 2

    def test_scores_sorted_descending(self):
        h = _build_hybrid_retriever()
        results = h.retrieve("neural network", top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_zero_one_range(self):
        # After min-max normalisation and convex combination, fused scores ∈ [0, 1].
        h = _build_hybrid_retriever()
        results = h.retrieve("DNA replication primase", top_k=5)
        for res in results:
            assert 0.0 <= res.score <= 1.0 + 1e-9

    def test_alpha_one_behaves_like_dense(self):
        # alpha=1 → pure dense; result order should match DenseRetriever.
        chunks = _make_chunks()
        dense, _ = _build_dense_retriever(chunks)
        hybrid = HybridRetriever(dense, BM25Retriever(chunks), alpha=1.0)
        query = "photosynthesis sunlight"
        dense_top = dense.retrieve(query, top_k=1)[0].chunk.content
        hybrid_top = hybrid.retrieve(query, top_k=1)[0].chunk.content
        assert dense_top == hybrid_top

    def test_alpha_zero_behaves_like_bm25(self):
        # alpha=0 → pure BM25; top result should match BM25Retriever.
        chunks = _make_chunks()
        dense, _ = _build_dense_retriever(chunks)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, alpha=0.0)
        query = "mitochondria ATP oxidative phosphorylation"
        bm25_top = bm25.retrieve(query, top_k=1)[0].chunk.content
        hybrid_top = hybrid.retrieve(query, top_k=1)[0].chunk.content
        assert bm25_top == hybrid_top

    def test_invalid_alpha_raises(self):
        chunks = _make_chunks()
        dense, _ = _build_dense_retriever(chunks)
        with pytest.raises(ValueError):
            HybridRetriever(dense, BM25Retriever(chunks), alpha=1.5)
        with pytest.raises(ValueError):
            HybridRetriever(dense, BM25Retriever(chunks), alpha=-0.1)

    def test_custom_fetch_k(self):
        h = _build_hybrid_retriever()
        h._fetch_k = 3
        results = h.retrieve("quantum entanglement", top_k=2)
        assert len(results) <= 2

    def test_chunks_drawn_from_indexed_set(self):
        chunks = _make_chunks()
        h = _build_hybrid_retriever(chunks)
        results = h.retrieve("photosynthesis", top_k=5)
        indexed_contents = {ch.content for ch in chunks}
        for res in results:
            assert res.chunk.content in indexed_contents


# ---------------------------------------------------------------------------
# _min_max_normalise helper
# ---------------------------------------------------------------------------

class TestMinMaxNormalise:
    def test_range_zero_to_one(self):
        scores = {"a": 2.0, "b": 5.0, "c": 8.0}
        normed = _min_max_normalise(scores)
        assert normed["a"] == pytest.approx(0.0)
        assert normed["c"] == pytest.approx(1.0)

    def test_all_equal_unchanged(self):
        scores = {"a": 3.0, "b": 3.0}
        normed = _min_max_normalise(scores)
        assert normed == scores

    def test_empty_returns_empty(self):
        assert _min_max_normalise({}) == {}

    def test_single_entry_unchanged(self):
        scores = {"only": 7.5}
        normed = _min_max_normalise(scores)
        assert normed == scores

    def test_negative_scores_normalised(self):
        scores = {"a": -1.0, "b": 0.0, "c": 1.0}
        normed = _min_max_normalise(scores)
        assert normed["a"] == pytest.approx(0.0)
        assert normed["b"] == pytest.approx(0.5)
        assert normed["c"] == pytest.approx(1.0)
