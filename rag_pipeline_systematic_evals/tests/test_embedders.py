"""
Tests for embedders.py.

All OpenAI calls are intercepted by a monkeypatched client that returns
deterministic vectors — no API key or network required.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_common.models import Chunk
from src.config import EmbedConfig, EmbedModel
from src.embedders import (
    _batch_embed, _cache_path, _load_cache, _save_cache,
    cache_exists, embed_chunks, embed_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(content=f"chunk text {i}", chunk_index=i) for i in range(n)]


def _fake_embed_response(texts: list[str]) -> object:
    """Returns an OpenAI-shaped response object with random unit vectors."""
    data = []
    for i, _ in enumerate(texts):
        vec = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        data.append(SimpleNamespace(embedding=vec.tolist()))
    return SimpleNamespace(data=data)


def _patched_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = (
        lambda input, model: _fake_embed_response(input)
    )
    monkeypatch.setattr("src.embedders._client", mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCacheHelpers:
    def test_cache_path_structure(self, tmp_path):
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)
        path = _cache_path(cfg, "fixed_256_ol50")
        assert path.parent.name == "small"
        assert path.name == "fixed_256_ol50.pkl"

    def test_save_and_load_roundtrip(self, tmp_path):
        chunks = _make_chunks(3)
        embeddings = np.random.rand(3, DIM).astype(np.float32)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)
        path = _cache_path(cfg, "test_label")

        _save_cache(path, [c.id_str() for c in chunks], embeddings)
        loaded = _load_cache(path)

        assert loaded["chunk_ids"] == [c.id_str() for c in chunks]
        np.testing.assert_array_almost_equal(loaded["embeddings"], embeddings)

    def test_cache_exists_false_initially(self, tmp_path):
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)
        assert not cache_exists(cfg, "missing_label")

    def test_cache_exists_true_after_save(self, tmp_path):
        chunks = _make_chunks(2)
        embeddings = np.random.rand(2, DIM).astype(np.float32)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)
        _save_cache(_cache_path(cfg, "my_label"), [c.id_str() for c in chunks], embeddings)
        assert cache_exists(cfg, "my_label")


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def test_returns_correct_shape(self, monkeypatch, tmp_path):
        _patched_client(monkeypatch)
        chunks = _make_chunks(5)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)
        embeddings = embed_chunks(chunks, cfg, "label")
        assert embeddings.shape == (5, DIM)

    def test_reads_from_cache_on_second_call(self, monkeypatch, tmp_path):
        mock = _patched_client(monkeypatch)
        chunks = _make_chunks(3)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)

        embed_chunks(chunks, cfg, "cached_label")
        call_count_after_first = mock.embeddings.create.call_count

        embed_chunks(chunks, cfg, "cached_label")
        # Second call should NOT hit the API.
        assert mock.embeddings.create.call_count == call_count_after_first

    def test_force_reembed_skips_cache(self, monkeypatch, tmp_path):
        mock = _patched_client(monkeypatch)
        chunks = _make_chunks(2)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)

        embed_chunks(chunks, cfg, "force_label")
        first_count = mock.embeddings.create.call_count

        embed_chunks(chunks, cfg, "force_label", force_reembed=True)
        assert mock.embeddings.create.call_count > first_count

    def test_cache_invalidated_on_id_mismatch(self, monkeypatch, tmp_path):
        mock = _patched_client(monkeypatch)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)

        old_chunks = _make_chunks(3)
        embed_chunks(old_chunks, cfg, "mismatch_label")
        first_count = mock.embeddings.create.call_count

        new_chunks = _make_chunks(3)   # different UUIDs
        embed_chunks(new_chunks, cfg, "mismatch_label")
        assert mock.embeddings.create.call_count > first_count

    def test_different_chunk_labels_get_separate_cache(self, monkeypatch, tmp_path):
        _patched_client(monkeypatch)
        chunks = _make_chunks(2)
        cfg = EmbedConfig(model=EmbedModel.SMALL, cache_dir=tmp_path)

        embed_chunks(chunks, cfg, "label_a")
        embed_chunks(chunks, cfg, "label_b")

        assert cache_exists(cfg, "label_a")
        assert cache_exists(cfg, "label_b")


# ---------------------------------------------------------------------------
# _batch_embed
# ---------------------------------------------------------------------------

class TestBatchEmbed:
    def test_batch_sizes(self, monkeypatch, tmp_path):
        mock = _patched_client(monkeypatch)
        texts = [f"text {i}" for i in range(7)]
        cfg = EmbedConfig(model=EmbedModel.SMALL, batch_size=3, cache_dir=tmp_path)
        result = _batch_embed(texts, cfg)
        assert result.shape == (7, DIM)
        # 7 texts at batch_size=3 → 3 API calls (batches of 3, 3, 1)
        assert mock.embeddings.create.call_count == 3
