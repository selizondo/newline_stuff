"""
Unit tests for rag_common.vector_store.

Tests run against both FAISSVectorStore and InMemoryVectorStore using the
same parametrised suite, ensuring they satisfy the VectorStoreProtocol
contract identically.

InMemoryVectorStore is also used as the canonical correct implementation
when verifying that FAISSVectorStore returns equivalent results.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_common.models import Chunk
from rag_common.vector_store import (
    FAISSVectorStore,
    InMemoryVectorStore,
    VectorStoreProtocol,
    _l2_normalise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(content=f"chunk {i}", chunk_index=i) for i in range(n)]


def _random_embeddings(n: int, dim: int = 16, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Protocol structural check
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_faiss_satisfies_protocol(self):
        assert isinstance(FAISSVectorStore(), VectorStoreProtocol)

    def test_in_memory_satisfies_protocol(self):
        assert isinstance(InMemoryVectorStore(), VectorStoreProtocol)

    def test_arbitrary_class_with_methods_satisfies_protocol(self):
        class MinimalStore:
            def add(self, chunks, embeddings): ...
            def search(self, query_embedding, top_k): return []
            def save(self, path): ...
            def load(self, path): ...
            def __len__(self): return 0

        assert isinstance(MinimalStore(), VectorStoreProtocol)

    def test_incomplete_class_fails_protocol(self):
        class BadStore:
            def add(self, chunks, embeddings): ...
            # missing search, save, load, __len__

        assert not isinstance(BadStore(), VectorStoreProtocol)


# ---------------------------------------------------------------------------
# Shared contract tests — run against both backends
# ---------------------------------------------------------------------------

@pytest.fixture(params=["faiss", "in_memory"])
def store(request):
    if request.param == "faiss":
        return FAISSVectorStore()
    return InMemoryVectorStore()


class TestVectorStoreContract:
    def test_empty_store_len_zero(self, store):
        assert len(store) == 0

    def test_add_increases_len(self, store):
        chunks = _make_chunks(5)
        embeddings = _random_embeddings(5)
        store.add(chunks, embeddings)
        assert len(store) == 5

    def test_add_multiple_batches(self, store):
        store.add(_make_chunks(3), _random_embeddings(3, seed=1))
        store.add(_make_chunks(4), _random_embeddings(4, seed=2))
        assert len(store) == 7

    def test_search_empty_returns_empty(self, store):
        q = _random_embeddings(1)[0]
        assert store.search(q, top_k=5) == []

    def test_search_returns_top_k(self, store):
        store.add(_make_chunks(10), _random_embeddings(10))
        results = store.search(_random_embeddings(1)[0], top_k=3)
        assert len(results) == 3

    def test_search_top_k_capped_at_n(self, store):
        store.add(_make_chunks(2), _random_embeddings(2))
        results = store.search(_random_embeddings(1)[0], top_k=100)
        assert len(results) == 2

    def test_search_results_sorted_descending(self, store):
        store.add(_make_chunks(10), _random_embeddings(10))
        results = store.search(_random_embeddings(1)[0], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_retrieval_results_with_dense_type(self, store):
        store.add(_make_chunks(5), _random_embeddings(5))
        results = store.search(_random_embeddings(1)[0], top_k=3)
        assert all(r.retriever_type == "dense" for r in results)

    def test_search_chunks_match_indexed_content(self, store):
        chunks = _make_chunks(5)
        embeddings = _random_embeddings(5)
        store.add(chunks, embeddings)
        results = store.search(_random_embeddings(1)[0], top_k=5)
        contents = {r.chunk.content for r in results}
        indexed_contents = {ch.content for ch in chunks}
        assert contents.issubset(indexed_contents)

    def test_exact_query_match_ranks_first(self, store):
        # Chunk 2's embedding is also used as the query — should rank #1.
        embeddings = _random_embeddings(5)
        chunks = _make_chunks(5)
        store.add(chunks, embeddings)
        query = embeddings[2]
        results = store.search(query, top_k=5)
        assert results[0].chunk.content == "chunk 2"

    def test_mismatch_chunks_embeddings_raises(self, store):
        with pytest.raises(ValueError):
            store.add(_make_chunks(3), _random_embeddings(5))

    def test_save_and_load_roundtrip(self, store, tmp_path):
        chunks = _make_chunks(4)
        embeddings = _random_embeddings(4)
        store.add(chunks, embeddings)

        store.save(str(tmp_path / "idx"))
        store.load(str(tmp_path / "idx"))

        assert len(store) == 4
        results = store.search(embeddings[0], top_k=1)
        assert results[0].chunk.content == "chunk 0"

    def test_save_creates_directory(self, store, tmp_path):
        store.add(_make_chunks(2), _random_embeddings(2))
        dest = str(tmp_path / "nested" / "dir" / "idx")
        store.save(dest)
        assert Path(dest).exists()

    def test_scores_in_cosine_range(self, store):
        store.add(_make_chunks(10), _random_embeddings(10))
        results = store.search(_random_embeddings(1)[0], top_k=10)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# FAISSVectorStore-specific tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def test_save_creates_faiss_and_pkl_files(self, tmp_path):
        store = FAISSVectorStore()
        store.add(_make_chunks(3), _random_embeddings(3))
        store.save(str(tmp_path / "idx"))
        assert (tmp_path / "idx" / "index.faiss").exists()
        assert (tmp_path / "idx" / "chunks.pkl").exists()

    def test_loaded_chunks_intact(self, tmp_path):
        store = FAISSVectorStore()
        chunks = _make_chunks(3)
        store.add(chunks, _random_embeddings(3))
        store.save(str(tmp_path / "idx"))

        store2 = FAISSVectorStore()
        store2.load(str(tmp_path / "idx"))
        assert len(store2) == 3
        contents = {ch.content for ch in store2._chunks}
        assert contents == {ch.content for ch in chunks}


# ---------------------------------------------------------------------------
# InMemoryVectorStore-specific tests
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def test_save_creates_npy_and_pkl(self, tmp_path):
        store = InMemoryVectorStore()
        store.add(_make_chunks(3), _random_embeddings(3))
        store.save(str(tmp_path / "idx"))
        assert (tmp_path / "idx" / "embeddings.npy").exists()
        assert (tmp_path / "idx" / "chunks.pkl").exists()

    def test_results_agree_with_faiss(self):
        # Both stores given identical data should return the same top chunk.
        embeddings = _random_embeddings(20, dim=32, seed=7)
        chunks = _make_chunks(20)
        query = embeddings[5]

        faiss_store = FAISSVectorStore()
        faiss_store.add(chunks, embeddings)

        mem_store = InMemoryVectorStore()
        mem_store.add(chunks, embeddings.astype(np.float64))

        faiss_top = faiss_store.search(query, top_k=1)[0].chunk.content
        mem_top = mem_store.search(query, top_k=1)[0].chunk.content
        assert faiss_top == mem_top


# ---------------------------------------------------------------------------
# _l2_normalise helper
# ---------------------------------------------------------------------------

class TestL2Normalise:
    def test_unit_norm_after_normalise(self):
        m = np.array([[3.0, 4.0], [1.0, 0.0]])
        normed = _l2_normalise(m)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_zero_vector_no_nan(self):
        m = np.array([[0.0, 0.0], [1.0, 0.0]])
        normed = _l2_normalise(m)
        assert not np.any(np.isnan(normed))
