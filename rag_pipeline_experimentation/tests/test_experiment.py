"""Tests for experiment.py — build_* factories and run_experiment resume logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunkers_ext import RecursiveChunker, SlidingWindowChunker
from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModelName,
    ExperimentConfig, RetrievalConfig,
)
from src.experiment import build_chunker, build_embedder, build_pipeline, run_experiment
from src.models import ExperimentResult


# ---------------------------------------------------------------------------
# build_chunker
# ---------------------------------------------------------------------------

class TestBuildChunker:
    def test_fixed(self):
        from rag_common.chunkers import FixedSizeChunker
        cfg = ChunkConfig(strategy=ChunkStrategy.FIXED, chunk_size=256, overlap=32)
        c = build_chunker(cfg)
        assert isinstance(c, FixedSizeChunker)

    def test_recursive(self):
        cfg = ChunkConfig(strategy=ChunkStrategy.RECURSIVE)
        c = build_chunker(cfg)
        assert isinstance(c, RecursiveChunker)

    def test_sliding_window(self):
        cfg = ChunkConfig(strategy=ChunkStrategy.SLIDING_WINDOW)
        c = build_chunker(cfg)
        assert isinstance(c, SlidingWindowChunker)

    def test_unknown_strategy_raises(self):
        cfg = ChunkConfig(strategy=ChunkStrategy.SEMANTIC)
        with pytest.raises(ValueError):
            build_chunker(cfg)


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------

class TestBuildEmbedder:
    def test_returns_embedder_with_correct_model(self):
        from src.embedders import SentenceTransformersEmbedder
        cfg = EmbedConfig(model=EmbedModelName.MINILM)
        emb = build_embedder(cfg)
        assert isinstance(emb, SentenceTransformersEmbedder)
        assert emb.model_name == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# run_experiment — resume logic
# ---------------------------------------------------------------------------

_DUMMY_TEXT = "Machine learning for retrieval. " * 30


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        chunk=ChunkConfig(),
        embed=EmbedConfig(),
        retrieval=RetrievalConfig(),
        n_queries=2,
    )


def _make_qrels(n: int = 2) -> dict[str, dict]:
    return {
        f"q{i:03d}": {
            "query": f"Test query {i}",
            "relevant_doc_ids": [f"doc_{i}"],
        }
        for i in range(n)
    }


def _fake_embedder(dim: int = 384) -> MagicMock:
    import numpy as np
    emb = MagicMock()
    emb.model_name = "all-MiniLM-L6-v2"
    emb.dimension = dim
    emb.embed.side_effect = lambda texts: np.random.randn(len(texts), dim).astype(np.float32)
    emb.embed_chunks.side_effect = lambda chunks, label: np.random.randn(
        len(chunks), dim
    ).astype(np.float32)
    return emb


class TestRunExperiment:
    def test_result_written_to_disk(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        qrels = _make_qrels(2)
        config = _make_config()
        result_dir = tmp_path / "results"
        index_dir = tmp_path / "indices"

        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 2)), \
             patch("src.experiment.build_embedder", return_value=_fake_embedder()):
            run_experiment(
                config=config,
                pdf_paths=[pdf],
                qrels=qrels,
                result_dir=result_dir,
                index_base_dir=index_dir,
            )

        result_path = result_dir / f"{config.experiment_id}.json"
        assert result_path.exists()

    def test_resume_skips_reingest(self, tmp_path):
        """If result file exists and force=False, pipeline.ingest must not be called."""
        config = _make_config()
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        result_path = result_dir / f"{config.experiment_id}.json"

        existing = ExperimentResult(
            experiment_id=config.experiment_id,
            config={},
            metrics={"mrr": 0.5},
            n_queries=2,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        result_path.write_text(existing.model_dump_json())

        mock_pipeline = MagicMock()
        with patch("src.experiment.build_pipeline", return_value=mock_pipeline):
            result = run_experiment(
                config=config,
                pdf_paths=[],
                qrels=_make_qrels(2),
                result_dir=result_dir,
                force=False,
            )

        mock_pipeline.ingest.assert_not_called()
        assert result.metrics["mrr"] == pytest.approx(0.5)

    def test_force_reruns(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        config = _make_config()
        result_dir = tmp_path / "results"
        result_dir.mkdir()

        stale = ExperimentResult(
            experiment_id=config.experiment_id,
            config={},
            metrics={"mrr": 0.0},
            n_queries=2,
        )
        result_path = result_dir / f"{config.experiment_id}.json"
        result_path.write_text(stale.model_dump_json())

        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 1)), \
             patch("src.experiment.build_embedder", return_value=_fake_embedder()):
            result = run_experiment(
                config=config,
                pdf_paths=[pdf],
                qrels=_make_qrels(2),
                result_dir=result_dir,
                index_base_dir=tmp_path / "idx",
                force=True,
            )

        assert result.n_queries == 2
