"""
Retrieval evaluator for P3.

Takes a QADataset (tied to one chunking config) and a retriever, runs
retrieval for each question, then computes all IR metrics via rag_common.

Metrics are computed at K = 1, 3, 5, 10 so the results table can show
the full recall/precision/NDCG curves, not just a single cutoff.

Precision@K note (repeated from rag_common docstring for visibility)
----------------------------------------------------------------------
Each synthetic question maps to exactly ONE ground-truth chunk, so
Precision@K is capped at 1/K (max 0.20 at K=5). This is expected — use
MRR and Recall@K as primary quality signals, not Precision@K.
"""

from __future__ import annotations

import time

from rag_common import metrics
from rag_common.retrievers import RetrieverProtocol
from src.config import EvaluationResult, ExperimentConfig, MetricsResult
from src.qa_generator import QADataset

_K_VALUES = [1, 3, 5, 10]


def evaluate(
    dataset: QADataset,
    retriever: RetrieverProtocol,
    config: ExperimentConfig,
) -> EvaluationResult:
    """
    Run retrieval for every question in `dataset` and return an EvaluationResult.

    Args:
        dataset:   QADataset generated for the same chunking config as the retriever's index.
        retriever: any RetrieverProtocol implementation (BM25, Dense, or Hybrid).
        config:    ExperimentConfig for this grid cell (stored verbatim in the result).

    Returns:
        EvaluationResult with full IR metrics and per-query detail.
    """
    if not dataset.pairs:
        raise ValueError(f"QADataset '{dataset.chunk_config_label}' is empty — cannot evaluate.")

    max_k = max(_K_VALUES)
    query_results: list[tuple[list[str], set[str]]] = []
    per_query_detail: list[dict] = []
    retrieval_times: list[float] = []

    for pair in dataset.pairs:
        t0 = time.perf_counter()
        results = retriever.retrieve(pair.question, top_k=max_k)
        elapsed = time.perf_counter() - t0

        retrieved_ids = [r.chunk.id_str() for r in results]
        relevant_ids  = set(pair.relevant_chunk_ids)

        query_results.append((retrieved_ids, relevant_ids))
        retrieval_times.append(elapsed)

        per_query_detail.append({
            "question":       pair.question,
            "question_type":  pair.question_type,
            "retrieved_ids":  retrieved_ids[:5],   # top-5 for debugging
            "relevant_ids":   list(relevant_ids),
            "retrieval_time": round(elapsed, 4),
            **{f"recall@{k}":    metrics.recall_at_k(retrieved_ids, relevant_ids, k)   for k in _K_VALUES},
            **{f"precision@{k}": metrics.precision_at_k(retrieved_ids, relevant_ids, k) for k in _K_VALUES},
            **{f"ndcg@{k}":      metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)      for k in _K_VALUES},
            "rr":             metrics.reciprocal_rank(retrieved_ids, relevant_ids),
            "ap":             metrics.average_precision(retrieved_ids, relevant_ids),
        })

    # Aggregate across all queries.
    metrics_result = MetricsResult(
        recall_at_k    = {str(k): metrics.mean_recall_at_k(query_results, k)    for k in _K_VALUES},
        precision_at_k = {str(k): metrics.mean_precision_at_k(query_results, k) for k in _K_VALUES},
        mrr            = metrics.mrr(query_results),
        map_score      = metrics.map_score(query_results),
        ndcg_at_k      = {str(k): metrics.mean_ndcg_at_k(query_results, k)      for k in _K_VALUES},
        total_queries  = len(dataset.pairs),
        avg_retrieval_time_s = sum(retrieval_times) / len(retrieval_times),
    )

    return EvaluationResult(
        experiment_id = config.experiment_id,
        config        = config,
        metrics       = metrics_result,
        query_results = per_query_detail,
    )


def best_config(results: list[EvaluationResult], primary: str = "mrr") -> EvaluationResult:
    """Return the EvaluationResult with the highest primary metric."""
    return max(results, key=lambda r: getattr(r.metrics, primary, 0.0))
