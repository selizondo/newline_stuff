"""Tests for RAGPipeline — mocks PDF parsing and embedder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_common.models import Chunk
from src.pipeline import RAGPipeline

DIM = 384
_DUMMY_TEXT = "Machine learning methods for document retrieval. " * 20


def _fake_embedder(dim: int = DIM) -> MagicMock:
    emb = MagicMock()
    emb.model_name = "all-MiniLM-L6-v2"
    emb.dimension = dim
    emb.embed.side_effect = lambda texts: np.random.randn(len(texts), dim).astype(np.float32)
    emb.embed_chunks.side_effect = lambda chunks, label: np.random.randn(
        len(chunks), dim
    ).astype(np.float32)
    return emb


def _fake_chunker() -> MagicMock:
    chunker = MagicMock()
    chunker.chunk.side_effect = lambda text, metadata=None: [
        Chunk(
            content=f"chunk {i}: {text[:30]}",
            chunk_index=i,
            document_id=(metadata or {}).get("document_id"),
            source=(metadata or {}).get("source"),
            metadata=metadata or {},
        )
        for i in range(5)
    ]
    return chunker


class TestRAGPipelineIngest:
    def test_ingest_returns_chunks(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")   # minimal placeholder

        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 3)):
            pipeline = RAGPipeline(
                chunker=_fake_chunker(),
                embedder=_fake_embedder(),
            )
            chunks = pipeline.ingest([pdf], index_dir=tmp_path / "idx")

        assert len(chunks) == 5
        assert all(c.document_id == "paper" for c in chunks)

    def test_ingest_saves_faiss_index(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 2)):
            pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
            pipeline.ingest([pdf], index_dir=tmp_path / "idx")

        assert (tmp_path / "idx" / "faiss_index" / "index.faiss").exists()
        assert (tmp_path / "idx" / "faiss_index" / "chunks.pkl").exists()

    def test_ingest_saves_document_metadata(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 2)):
            pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
            pipeline.ingest([pdf], index_dir=tmp_path / "idx")

        assert (tmp_path / "idx" / "documents.json").exists()

    def test_empty_pdf_skipped(self, tmp_path):
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch("src.pipeline._parse_pdf", return_value=("", 0)):
            pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
            with pytest.raises(ValueError, match="No chunks"):
                pipeline.ingest([pdf], index_dir=tmp_path / "idx")


class TestRAGPipelineQuery:
    def _build_pipeline(self, tmp_path: Path) -> RAGPipeline:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 2)):
            pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
            pipeline.ingest([pdf], index_dir=tmp_path / "idx")
        return pipeline

    def test_query_returns_results(self, tmp_path):
        pipeline = self._build_pipeline(tmp_path)
        results = pipeline.query("What are the methods?", top_k=3)
        assert len(results) <= 3
        assert all(r.chunk.content for r in results)

    def test_query_before_ingest_raises(self):
        pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
        with pytest.raises(RuntimeError):
            pipeline.query("test")

    def test_query_timed_returns_tuple(self, tmp_path):
        pipeline = self._build_pipeline(tmp_path)
        results, elapsed = pipeline.query_timed("test question", top_k=2)
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_hybrid_retriever(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 2)):
            pipeline = RAGPipeline(
                chunker=_fake_chunker(),
                embedder=_fake_embedder(),
                retrieval_method="hybrid",
                alpha=0.6,
            )
            pipeline.ingest([pdf], index_dir=tmp_path / "idx")
        results = pipeline.query("hybrid retrieval test", top_k=3)
        assert len(results) >= 1


class TestRAGPipelineConfig:
    def test_documents_property(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        with patch("src.pipeline._parse_pdf", return_value=(_DUMMY_TEXT, 1)):
            pipeline = RAGPipeline(chunker=_fake_chunker(), embedder=_fake_embedder())
            pipeline.ingest([pdf], index_dir=tmp_path / "idx")
        assert len(pipeline.documents) == 1
        assert pipeline.documents[0].id == "doc"
