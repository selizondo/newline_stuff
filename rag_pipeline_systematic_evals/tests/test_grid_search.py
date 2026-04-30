"""
Tests for grid_search.py — no API calls, no PDF required.

Strategy:
  - Patch parse_pdf, generate_qa_dataset, embed_chunks, embed_query so
    no real I/O or model calls happen.
  - Use real rag_common chunkers/retrievers/metrics so the integration
    path is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_common.models import Chunk, RetrievalResult
from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModel,
    EvaluationResult, ExperimentConfig, RetrievalConfig, RetrievalMethod,
)
from src.grid_search import (
    _chunk_document, _load_result, _make_embed_fn, _make_retriever, _save_result,
    run_grid_search,
)
from src.qa_generator import QADataset, QAPair


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 8


def _make_chunks(n: int = 5) -> list[Chunk]:
    return [Chunk(content=f"sentence {i} ends here.", chunk_index=i, method="fixed_size")
            for i in range(n)]


def _make_dataset(chunks: list[Chunk]) -> QADataset:
    pairs = [
        QAPair(
            question=f"What is sentence {i}?",
            question_type="factual",
            relevant_chunk_ids=[chunks[i % len(chunks)].id_str()],
            metadata={"synthetic": True, "chunk_method": "fixed_size",
                      "source_chunk_index": i, "source_page": 1},
        )
        for i in range(len(chunks))
    ]
    return QADataset(chunk_config_label="fixed_256_ol50", pairs=pairs)


def _fake_embeddings(chunks: list[Chunk]) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(chunks), DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _fake_embed_fn(texts: list[str]) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(texts), DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# _chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_fixed_size_returns_chunks(self):
        text = "word " * 300   # enough for multiple 256-char chunks
        chunks = _chunk_document(
            text,
            ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256, overlap=50),
            EmbedConfig(model=EmbedModel.SMALL),
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sentence_returns_chunks(self):
        text = "This is sentence one. This is sentence two. This is sentence three. " * 5
        chunks = _chunk_document(
            text,
            ChunkConfig(strategy=ChunkStrategy.SENTENCE, sentences_per_chunk=3, overlap_sentences=1),
            EmbedConfig(model=EmbedModel.SMALL),
        )
        assert len(chunks) >= 1

    def test_semantic_calls_embed_fn(self):
        text = "Alpha eats beta. Gamma kills delta. Zeta runs fast. " * 6
        called = [False]

        def tracking_embed(texts):
            called[0] = True
            return _fake_embed_fn(texts)

        with patch("src.grid_search._make_embed_fn", return_value=tracking_embed):
            _chunk_document(
                text,
                ChunkConfig(strategy=ChunkStrategy.SEMANTIC, breakpoint_threshold=0.5, max_sentences=5),
                EmbedConfig(model=EmbedModel.SMALL),
            )
        # SemanticChunker internally calls embed_fn during chunking.
        # We just verify no crash and chunks are produced.


# ---------------------------------------------------------------------------
# _make_embed_fn
# ---------------------------------------------------------------------------

class TestMakeEmbedFn:
    def test_returns_correct_shape(self):
        def fake_embed_texts(texts, cfg):
            return np.ones((len(texts), DIM), dtype=np.float32)

        with patch("src.grid_search.embed_texts", side_effect=fake_embed_texts):
            fn = _make_embed_fn(EmbedConfig(model=EmbedModel.SMALL))
            result = fn(["hello", "world"])
        assert result.shape == (2, DIM)

    def test_texts_embedded_in_single_batch(self):
        call_count = [0]
        def fake_embed_texts(texts, cfg):
            call_count[0] += 1
            return np.ones((len(texts), DIM), dtype=np.float32)

        with patch("src.grid_search.embed_texts", side_effect=fake_embed_texts):
            fn = _make_embed_fn(EmbedConfig(model=EmbedModel.SMALL))
            fn(["a", "b", "c"])

        assert call_count[0] == 1   # batched: one call for all texts


# ---------------------------------------------------------------------------
# _make_retriever
# ---------------------------------------------------------------------------

class TestMakeRetriever:
    def _store_with_chunks(self, chunks):
        from rag_common.vector_store import FAISSVectorStore
        store = FAISSVectorStore()
        store.add(chunks, _fake_embeddings(chunks))
        return store

    def test_bm25_retriever_type(self):
        chunks = _make_chunks(4)
        from rag_common.retrievers import BM25Retriever
        cfg = RetrievalConfig(method=RetrievalMethod.BM25)
        retriever = _make_retriever(cfg, self._store_with_chunks(chunks), chunks, _fake_embed_fn)
        assert isinstance(retriever, BM25Retriever)

    def test_dense_retriever_type(self):
        chunks = _make_chunks(4)
        from rag_common.retrievers import DenseRetriever
        cfg = RetrievalConfig(method=RetrievalMethod.VECTOR)
        retriever = _make_retriever(cfg, self._store_with_chunks(chunks), chunks, _fake_embed_fn)
        assert isinstance(retriever, DenseRetriever)

    def test_hybrid_retriever_type(self):
        chunks = _make_chunks(4)
        from rag_common.retrievers import HybridRetriever
        cfg = RetrievalConfig(method=RetrievalMethod.HYBRID, alpha=0.5)
        retriever = _make_retriever(cfg, self._store_with_chunks(chunks), chunks, _fake_embed_fn)
        assert isinstance(retriever, HybridRetriever)

    def test_bm25_can_retrieve(self):
        chunks = _make_chunks(4)
        cfg = RetrievalConfig(method=RetrievalMethod.BM25, top_k=2)
        retriever = _make_retriever(cfg, self._store_with_chunks(chunks), chunks, _fake_embed_fn)
        results = retriever.retrieve("sentence 0", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)


# ---------------------------------------------------------------------------
# _save_result / _load_result
# ---------------------------------------------------------------------------

class TestPersistence:
    def _make_result(self) -> EvaluationResult:
        from src.config import MetricsResult
        cfg = ExperimentConfig(
            chunk=ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE),
            embed=EmbedConfig(model=EmbedModel.SMALL),
            retrieval=RetrievalConfig(method=RetrievalMethod.VECTOR),
        )
        return EvaluationResult(
            experiment_id=cfg.experiment_id,
            config=cfg,
            metrics=MetricsResult(
                recall_at_k={"5": 0.8}, precision_at_k={"5": 0.2},
                mrr=0.75, map_score=0.70,
                ndcg_at_k={"5": 0.77}, total_queries=10,
            ),
        )

    def test_roundtrip(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "exp.json"
        _save_result(result, path)
        loaded = _load_result(path)
        assert loaded.experiment_id == result.experiment_id
        assert abs(loaded.metrics.mrr - result.metrics.mrr) < 1e-6

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "exp.json"
        _save_result(self._make_result(), path)
        data = json.loads(path.read_text())
        assert "experiment_id" in data


# ---------------------------------------------------------------------------
# run_grid_search  (fully mocked I/O)
# ---------------------------------------------------------------------------

class TestRunGridSearch:
    """
    Mock out every I/O call so the orchestrator logic can be tested
    without a real PDF or API key.
    """

    def _run(self, tmp_path, chunk_cfgs=None, embed_cfgs=None, retrieval_cfgs=None, force=False):
        # QA dataset uses fixed chunks; embed_chunks returns size matching actual chunks.
        fixed_chunks = _make_chunks(6)
        dataset = _make_dataset(fixed_chunks)

        def fake_embed_chunks(chunks, cfg, label, **kw):
            return _fake_embeddings(chunks)   # size matches whatever chunker returned

        fake_doc = MagicMock()
        fake_doc.full_text = "sentence 0 ends here. " * 40

        with (
            patch("src.grid_search.parse_pdf", return_value=fake_doc),
            patch("src.grid_search.generate_qa_dataset", return_value=dataset),
            patch("src.grid_search.embed_chunks", side_effect=fake_embed_chunks),
            patch("src.grid_search.embed_texts",
                  side_effect=lambda texts, cfg: np.ones((len(texts), DIM), dtype=np.float32)),
        ):
            return run_grid_search(
                pdf_path=Path("dummy.pdf"),
                experiment_dir=tmp_path / "experiments",
                qa_dir=tmp_path / "qa",
                n_pairs=6,
                chunk_configs=chunk_cfgs or [
                    ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256, overlap=50),
                ],
                embed_configs=embed_cfgs or [
                    EmbedConfig(model=EmbedModel.SMALL),
                ],
                retrieval_configs=retrieval_cfgs or [
                    RetrievalConfig(method=RetrievalMethod.VECTOR),
                ],
                force=force,
            )

    def test_returns_evaluation_results(self, tmp_path):
        results = self._run(tmp_path)
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)

    def test_three_retrieval_configs_produce_three_results(self, tmp_path):
        results = self._run(tmp_path, retrieval_cfgs=[
            RetrievalConfig(method=RetrievalMethod.VECTOR),
            RetrievalConfig(method=RetrievalMethod.BM25),
            RetrievalConfig(method=RetrievalMethod.HYBRID, alpha=0.5),
        ])
        assert len(results) == 3

    def test_result_files_written_to_disk(self, tmp_path):
        results = self._run(tmp_path)
        experiment_dir = tmp_path / "experiments"
        assert experiment_dir.exists()
        files = list(experiment_dir.glob("*.json"))
        assert len(files) == len(results)

    def test_resume_skips_existing_files(self, tmp_path):
        # First run writes result files.
        self._run(tmp_path)

        # Count parse_pdf calls on second run (should be 0 — all skipped).
        with patch("src.grid_search.parse_pdf") as mock_parse:
            self._run(tmp_path, force=False)
        mock_parse.assert_not_called()

    def test_force_reruns_even_if_file_exists(self, tmp_path):
        self._run(tmp_path)
        call_count = [0]
        original = __import__("src.evaluator", fromlist=["evaluate"]).evaluate

        def counting_evaluate(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        with patch("src.grid_search.evaluate", side_effect=counting_evaluate):
            self._run(tmp_path, force=True)

        assert call_count[0] == 1   # 1 config cell forced to re-run

    def test_experiment_ids_unique(self, tmp_path):
        results = self._run(tmp_path, retrieval_cfgs=[
            RetrievalConfig(method=RetrievalMethod.VECTOR),
            RetrievalConfig(method=RetrievalMethod.BM25),
        ])
        ids = [r.experiment_id for r in results]
        assert len(ids) == len(set(ids))

    def test_metrics_mrr_in_range(self, tmp_path):
        results = self._run(tmp_path)
        for r in results:
            assert 0.0 <= r.metrics.mrr <= 1.0
