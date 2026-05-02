"""Tests for SentenceTransformersEmbedder — mocks model loading."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_common.models import Chunk
from src.embedders import SentenceTransformersEmbedder

DIM = 384


def _fake_model(dim: int = DIM):
    """Return a mock SentenceTransformer that returns random unit vectors."""
    m = MagicMock()
    m.get_sentence_embedding_dimension.return_value = dim
    m.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), dim).astype(np.float32)
    return m


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(content=f"chunk {i}", chunk_index=i) for i in range(n)]


class TestSentenceTransformersEmbedder:
    def test_embed_shape(self):
        embedder = SentenceTransformersEmbedder()
        with patch("src.embedders.SentenceTransformer", return_value=_fake_model()):
            result = embedder.embed(["hello", "world"])
        assert result.shape == (2, DIM)
        assert result.dtype == np.float32

    def test_embed_empty(self):
        embedder = SentenceTransformersEmbedder()
        with patch("src.embedders.SentenceTransformer", return_value=_fake_model()):
            result = embedder.embed([])
        assert result.shape[0] == 0

    def test_model_name_property(self):
        embedder = SentenceTransformersEmbedder(model_name="all-mpnet-base-v2")
        assert embedder.model_name == "all-mpnet-base-v2"

    def test_dimension_from_known_map(self):
        embedder = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.dimension == 384

    def test_dimension_from_model_when_unknown(self):
        embedder = SentenceTransformersEmbedder(model_name="unknown-model")
        with patch("src.embedders.SentenceTransformer", return_value=_fake_model(512)):
            assert embedder.dimension == 512

    def test_embed_chunks_shape(self, tmp_path):
        embedder = SentenceTransformersEmbedder(cache_dir=tmp_path)
        chunks = _make_chunks(5)
        with patch("src.embedders.SentenceTransformer", return_value=_fake_model()):
            result = embedder.embed_chunks(chunks, "test_label")
        assert result.shape == (5, DIM)

    def test_embed_chunks_cache_written(self, tmp_path):
        embedder = SentenceTransformersEmbedder(cache_dir=tmp_path)
        chunks = _make_chunks(3)
        with patch("src.embedders.SentenceTransformer", return_value=_fake_model()):
            embedder.embed_chunks(chunks, "label1")
        cache_files = list(tmp_path.rglob("*.pkl"))
        assert len(cache_files) == 1

    def test_embed_chunks_cache_hit_skips_model(self, tmp_path):
        embedder = SentenceTransformersEmbedder(cache_dir=tmp_path)
        chunks = _make_chunks(3)
        mock_model = _fake_model()
        with patch("src.embedders.SentenceTransformer", return_value=mock_model):
            embedder.embed_chunks(chunks, "label1")   # populates cache
            call_count = mock_model.encode.call_count
            embedder.embed_chunks(chunks, "label1")   # should hit cache
        assert mock_model.encode.call_count == call_count   # no extra call

    def test_embed_chunks_stale_cache_regenerates(self, tmp_path):
        embedder = SentenceTransformersEmbedder(cache_dir=tmp_path)
        old_chunks = _make_chunks(3)
        new_chunks = _make_chunks(5)   # different UUIDs
        mock_model = _fake_model()
        with patch("src.embedders.SentenceTransformer", return_value=mock_model):
            embedder.embed_chunks(old_chunks, "label1")
            first_calls = mock_model.encode.call_count
            embedder.embed_chunks(new_chunks, "label1")   # stale → regenerate
        assert mock_model.encode.call_count > first_calls
