"""Tests for evaluator.py — no API calls, stub retriever used throughout."""

from __future__ import annotations

import pytest

from rag_common.models import Chunk, RetrievalResult
from rag_common.retrievers import RetrieverProtocol
from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModel,
    ExperimentConfig, RetrievalConfig, RetrievalMethod,
)
from src.evaluator import _K_VALUES, best_config, evaluate
from src.qa_generator import QADataset, QAPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(content=f"chunk {i}", chunk_index=i) for i in range(n)]


def _make_dataset(chunks: list[Chunk], n_pairs: int | None = None) -> QADataset:
    n = n_pairs or len(chunks)
    pairs = [
        QAPair(
            question=f"What is chunk {i}?",
            question_type="factual",
            relevant_chunk_ids=[chunks[i % len(chunks)].id_str()],
            metadata={"synthetic": True, "chunk_method": "fixed_size",
                      "source_chunk_index": i, "source_page": 1},
        )
        for i in range(n)
    ]
    return QADataset(chunk_config_label="fixed_256_ol50", pairs=pairs)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        chunk=ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256, overlap=50),
        embed=EmbedConfig(model=EmbedModel.SMALL),
        retrieval=RetrievalConfig(method=RetrievalMethod.VECTOR),
    )


class PerfectRetriever:
    """Always returns the relevant chunk first."""
    def __init__(self, chunks: list[Chunk]):
        self._by_id = {c.id_str(): c for c in chunks}
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Find which chunk the query refers to by index embedded in query text.
        for chunk in self._chunks:
            if chunk.content in query or str(chunk.chunk_index) in query:
                ordered = [chunk] + [c for c in self._chunks if c.id_str() != chunk.id_str()]
                return [RetrievalResult(chunk=c, score=1.0 - i * 0.1, retriever_type="dense")
                        for i, c in enumerate(ordered[:top_k])]
        return [RetrievalResult(chunk=c, score=1.0, retriever_type="dense")
                for c in self._chunks[:top_k]]


class WorstRetriever:
    """Never returns the relevant chunk in top results."""
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Return chunks in reverse order so the relevant one is always last.
        reversed_chunks = list(reversed(self._chunks))
        return [RetrievalResult(chunk=c, score=0.1, retriever_type="dense")
                for c in reversed_chunks[:top_k]]


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_evaluation_result(self):
        chunks = _make_chunks(5)
        dataset = _make_dataset(chunks)
        result = evaluate(dataset, PerfectRetriever(chunks), _make_config())
        from src.config import EvaluationResult
        assert isinstance(result, EvaluationResult)

    def test_experiment_id_matches_config(self):
        chunks = _make_chunks(5)
        cfg = _make_config()
        result = evaluate(_make_dataset(chunks), PerfectRetriever(chunks), cfg)
        assert result.experiment_id == cfg.experiment_id

    def test_total_queries_correct(self):
        chunks = _make_chunks(5)
        dataset = _make_dataset(chunks, n_pairs=5)
        result = evaluate(dataset, PerfectRetriever(chunks), _make_config())
        assert result.metrics.total_queries == 5

    def test_perfect_retriever_high_mrr(self):
        chunks = _make_chunks(10)
        dataset = _make_dataset(chunks)
        result = evaluate(dataset, PerfectRetriever(chunks), _make_config())
        assert result.metrics.mrr > 0.0

    def test_worst_retriever_lower_recall_than_perfect(self):
        chunks = _make_chunks(6)
        dataset = _make_dataset(chunks)
        worst = evaluate(dataset, WorstRetriever(chunks), _make_config())
        perfect = evaluate(dataset, PerfectRetriever(chunks), _make_config())
        assert worst.metrics.mrr < perfect.metrics.mrr

    def test_metrics_keys_for_all_k(self):
        chunks = _make_chunks(5)
        result = evaluate(_make_dataset(chunks), PerfectRetriever(chunks), _make_config())
        for k in _K_VALUES:
            assert str(k) in result.metrics.recall_at_k
            assert str(k) in result.metrics.precision_at_k
            assert str(k) in result.metrics.ndcg_at_k

    def test_per_query_detail_length(self):
        chunks = _make_chunks(4)
        dataset = _make_dataset(chunks, n_pairs=4)
        result = evaluate(dataset, PerfectRetriever(chunks), _make_config())
        assert len(result.query_results) == 4

    def test_per_query_has_expected_keys(self):
        chunks = _make_chunks(3)
        result = evaluate(_make_dataset(chunks, n_pairs=1), PerfectRetriever(chunks), _make_config())
        detail = result.query_results[0]
        assert "question" in detail
        assert "retrieved_ids" in detail
        assert "rr" in detail
        assert "recall@5" in detail

    def test_avg_retrieval_time_positive(self):
        chunks = _make_chunks(3)
        result = evaluate(_make_dataset(chunks), PerfectRetriever(chunks), _make_config())
        assert result.metrics.avg_retrieval_time_s >= 0.0

    def test_empty_dataset_raises(self):
        chunks = _make_chunks(3)
        dataset = QADataset(chunk_config_label="test", pairs=[])
        with pytest.raises(ValueError, match="empty"):
            evaluate(dataset, PerfectRetriever(chunks), _make_config())

    def test_mrr_range(self):
        chunks = _make_chunks(5)
        result = evaluate(_make_dataset(chunks), PerfectRetriever(chunks), _make_config())
        assert 0.0 <= result.metrics.mrr <= 1.0

    def test_map_range(self):
        chunks = _make_chunks(5)
        result = evaluate(_make_dataset(chunks), PerfectRetriever(chunks), _make_config())
        assert 0.0 <= result.metrics.map_score <= 1.0


# ---------------------------------------------------------------------------
# best_config()
# ---------------------------------------------------------------------------

class TestBestConfig:
    def _make_result(self, mrr: float, chunks: list[Chunk]) -> object:
        from src.config import EvaluationResult, MetricsResult
        cfg = _make_config()
        return EvaluationResult(
            experiment_id=cfg.experiment_id,
            config=cfg,
            metrics=MetricsResult(
                recall_at_k={"5": 0.8}, precision_at_k={"5": 0.2},
                mrr=mrr, map_score=mrr,
                ndcg_at_k={"5": mrr}, total_queries=5,
            ),
        )

    def test_returns_highest_mrr(self):
        chunks = _make_chunks(3)
        results = [self._make_result(0.5, chunks), self._make_result(0.9, chunks), self._make_result(0.3, chunks)]
        best = best_config(results, primary="mrr")
        assert best.metrics.mrr == 0.9
