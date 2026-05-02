"""Tests for evaluator — mocks pipeline.query, no disk I/O for IR logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_common.models import Chunk, RetrievalResult
from src.config import ChunkConfig, EmbedConfig, ExperimentConfig, RetrievalConfig
from src.evaluator import best_config, evaluate, filter_qrels_by_docs, load_qrels, save_qrels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_qrels(n: int = 5) -> dict[str, dict]:
    return {
        f"q{i:03d}": {
            "query": f"Test query {i}",
            "relevant_doc_ids": [f"doc_{i}", f"doc_{i+1}"],
        }
        for i in range(n)
    }


def _make_config(n_queries: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        chunk=ChunkConfig(),
        embed=EmbedConfig(),
        retrieval=RetrievalConfig(),
        n_queries=n_queries,
    )


def _make_retrieval_result(doc_id: str, score: float = 0.9) -> RetrievalResult:
    chunk = Chunk(content="test content", chunk_index=0, document_id=doc_id)
    return RetrievalResult(chunk=chunk, score=score, retriever_type="dense")


def _mock_pipeline(doc_ids: list[str]) -> MagicMock:
    pipeline = MagicMock()
    pipeline.query.return_value = [_make_retrieval_result(d) for d in doc_ids]
    return pipeline


# ---------------------------------------------------------------------------
# load_qrels / save_qrels
# ---------------------------------------------------------------------------

class TestQrelsIO:
    def test_load_roundtrip(self, tmp_path):
        qrels = _make_qrels(3)
        path = tmp_path / "qrels.json"
        save_qrels(qrels, path)
        loaded = load_qrels(path)
        assert loaded == qrels

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_qrels(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_experiment_result(self):
        qrels = _make_qrels(3)
        config = _make_config(n_queries=3)
        pipeline = _mock_pipeline(["doc_0", "doc_1"])
        result = evaluate(qrels, pipeline, config)
        assert result.experiment_id == config.experiment_id
        assert result.n_queries == 3

    def test_metrics_keys_present(self):
        qrels = _make_qrels(3)
        config = _make_config(n_queries=3)
        pipeline = _mock_pipeline(["doc_0", "doc_1"])
        result = evaluate(qrels, pipeline, config)
        for key in ("mrr", "map", "recall@5", "precision@5", "ndcg@5"):
            assert key in result.metrics, f"missing key: {key}"

    def test_metric_values_bounded(self):
        qrels = _make_qrels(5)
        config = _make_config(n_queries=5)
        pipeline = _mock_pipeline(["doc_0", "doc_1"])
        result = evaluate(qrels, pipeline, config)
        for v in result.metrics.values():
            assert 0.0 <= v <= 1.0 or v >= 0.0  # latency can exceed 1.0

    def test_perfect_retrieval_mrr_one(self):
        qrels = {"q001": {"query": "test", "relevant_doc_ids": ["doc_a"]}}
        config = _make_config(n_queries=1)
        pipeline = _mock_pipeline(["doc_a", "doc_b"])
        result = evaluate(qrels, pipeline, config)
        assert result.metrics["mrr"] == pytest.approx(1.0)

    def test_no_overlap_mrr_zero(self):
        qrels = {"q001": {"query": "test", "relevant_doc_ids": ["doc_z"]}}
        config = _make_config(n_queries=1)
        pipeline = _mock_pipeline(["doc_a", "doc_b"])
        result = evaluate(qrels, pipeline, config)
        assert result.metrics["mrr"] == pytest.approx(0.0)

    def test_query_results_count(self):
        qrels = _make_qrels(4)
        config = _make_config(n_queries=4)
        pipeline = _mock_pipeline(["doc_0"])
        result = evaluate(qrels, pipeline, config)
        assert len(result.query_results) == 4

    def test_n_queries_cap(self):
        qrels = _make_qrels(10)
        config = _make_config(n_queries=3)  # cap at 3
        pipeline = _mock_pipeline(["doc_0"])
        result = evaluate(qrels, pipeline, config)
        assert result.n_queries == 3

    def test_empty_qrels_raises(self):
        config = _make_config(n_queries=5)
        pipeline = _mock_pipeline([])
        with pytest.raises(ValueError, match="empty"):
            evaluate({}, pipeline, config)

    def test_deduplication_of_doc_ids(self):
        # Two chunks from same doc — should appear once in retrieved_ids.
        chunk1 = Chunk(content="c1", chunk_index=0, document_id="doc_a")
        chunk2 = Chunk(content="c2", chunk_index=1, document_id="doc_a")
        pipeline = MagicMock()
        pipeline.query.return_value = [
            RetrievalResult(chunk=chunk1, score=0.9, retriever_type="dense"),
            RetrievalResult(chunk=chunk2, score=0.8, retriever_type="dense"),
        ]
        qrels = {"q001": {"query": "test", "relevant_doc_ids": ["doc_a"]}}
        config = _make_config(n_queries=1)
        result = evaluate(qrels, pipeline, config)
        assert result.query_results[0].retrieved_ids == ["doc_a"]

    def test_timestamp_set(self):
        qrels = _make_qrels(1)
        config = _make_config(n_queries=1)
        pipeline = _mock_pipeline(["doc_0"])
        result = evaluate(qrels, pipeline, config)
        assert result.timestamp != ""


# ---------------------------------------------------------------------------
# filter_qrels_by_docs
# ---------------------------------------------------------------------------

class TestFilterQrelsByDocs:
    def test_keeps_matching_queries(self):
        qrels = {
            "q1": {"query": "a", "relevant_doc_ids": ["doc_a"]},
            "q2": {"query": "b", "relevant_doc_ids": ["doc_b"]},
        }
        result = filter_qrels_by_docs(qrels, {"doc_a"})
        assert set(result.keys()) == {"q1"}

    def test_empty_ingested_returns_empty(self):
        qrels = {"q1": {"query": "a", "relevant_doc_ids": ["doc_a"]}}
        assert filter_qrels_by_docs(qrels, set()) == {}

    def test_all_ingested_returns_all(self):
        qrels = _make_qrels(5)
        all_ids = {doc_id for e in qrels.values() for doc_id in e["relevant_doc_ids"]}
        assert filter_qrels_by_docs(qrels, all_ids) == qrels

    def test_partial_relevant_doc_match_included(self):
        qrels = {"q1": {"query": "a", "relevant_doc_ids": ["doc_a", "doc_b"]}}
        result = filter_qrels_by_docs(qrels, {"doc_b"})
        assert "q1" in result

    def test_missing_relevant_doc_ids_key(self):
        qrels = {"q1": {"query": "a"}}
        assert filter_qrels_by_docs(qrels, {"doc_a"}) == {}


# ---------------------------------------------------------------------------
# best_config
# ---------------------------------------------------------------------------

class TestBestConfig:
    def _make_result(self, exp_id: str, mrr: float):
        from src.models import ExperimentResult
        return ExperimentResult(
            experiment_id=exp_id,
            config={},
            metrics={"mrr": mrr, "map": 0.5},
            n_queries=5,
        )

    def test_returns_highest_mrr(self):
        results = [
            self._make_result("a", 0.4),
            self._make_result("b", 0.9),
            self._make_result("c", 0.6),
        ]
        winner = best_config(results)
        assert winner.experiment_id == "b"
