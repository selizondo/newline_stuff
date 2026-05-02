"""
Document-level IR evaluator for P4.

P4 differs from P3 in two key ways:
  1. Ground truth is document-level (qrels.json) not chunk-level (synthetic QA).
  2. Multiple documents are ingested into one index.

Relevance mapping:
    retrieved_ids = ordered list of unique document_ids from top-K chunks
    relevant_ids  = set of doc IDs from qrels for that query

This lets us reuse rag_common metrics directly with document IDs.

qrels.json format:
    {
        "q001": {
            "query": "What methods are used for retrieval?",
            "relevant_doc_ids": ["paper_stem_a", "paper_stem_b"]
        },
        ...
    }

where each doc ID is the PDF filename stem (e.g. "paper" for "paper.pdf").
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_common import metrics

from src.config import ExperimentConfig
from src.models import ExperimentResult, QueryResult
from src.pipeline import RAGPipeline

_K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# qrels I/O
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> dict[str, dict]:
    """
    Load qrels.json → {query_id: {"query": str, "relevant_doc_ids": list[str]}}.

    Raises FileNotFoundError if path does not exist.
    """
    with open(path) as f:
        return json.load(f)


def save_qrels(qrels: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(qrels, f, indent=2)


def filter_qrels_by_docs(
    qrels: dict[str, dict],
    ingested_doc_ids: set[str],
) -> dict[str, dict]:
    """
    Return only qrels entries whose relevant docs are in `ingested_doc_ids`.

    Without this filter, queries for non-ingested papers always score MRR=0,
    making --limit N results uninterpretable when qrels covers more papers
    than were ingested.
    """
    return {
        qid: entry for qid, entry in qrels.items()
        if any(doc_id in ingested_doc_ids for doc_id in entry.get("relevant_doc_ids", []))
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    qrels: dict[str, dict],
    pipeline: RAGPipeline,
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    Run retrieval for each query in `qrels` (up to config.n_queries) and
    return an ExperimentResult with full IR metrics.

    Args:
        qrels:    Loaded qrels dict (query_id → entry).
        pipeline: Ingested RAGPipeline (must have been ingested before calling).
        config:   ExperimentConfig for this grid cell.

    Returns:
        ExperimentResult ready to write to disk.
    """
    query_items = list(qrels.items())[: config.n_queries]
    if not query_items:
        raise ValueError("qrels is empty — cannot evaluate.")

    top_k = max(_K_VALUES)
    paired: list[tuple[list[str], set[str]]] = []
    query_results: list[QueryResult] = []
    latencies: list[float] = []

    for query_id, entry in query_items:
        query = entry["query"]
        relevant_doc_ids: set[str] = set(entry.get("relevant_doc_ids", []))

        t0 = time.perf_counter()
        results = pipeline.query(query, top_k=top_k)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Map retrieved chunks → unique document IDs (preserve order).
        seen: dict[str, None] = {}
        for r in results:
            doc_id = r.chunk.document_id or ""
            if doc_id not in seen:
                seen[doc_id] = None
        retrieved_doc_ids: list[str] = list(seen.keys())

        paired.append((retrieved_doc_ids, relevant_doc_ids))
        query_results.append(QueryResult(
            query_id=query_id,
            query=query,
            retrieved_ids=retrieved_doc_ids,
            relevant_ids=list(relevant_doc_ids),
            retrieval_time_s=round(elapsed, 4),
        ))

    # Aggregate IR metrics across all queries.
    agg: dict[str, float] = {
        "mrr":      metrics.mrr(paired),
        "map":      metrics.map_score(paired),
        **{f"recall@{k}":    metrics.mean_recall_at_k(paired, k)    for k in _K_VALUES},
        **{f"precision@{k}": metrics.mean_precision_at_k(paired, k) for k in _K_VALUES},
        **{f"ndcg@{k}":      metrics.mean_ndcg_at_k(paired, k)      for k in _K_VALUES},
        "avg_retrieval_time_s": sum(latencies) / len(latencies),
    }

    return ExperimentResult(
        experiment_id=config.experiment_id,
        config=config.model_dump(mode="json"),
        metrics={k: round(v, 6) for k, v in agg.items()},
        query_results=query_results,
        avg_latency_s=round(sum(latencies) / len(latencies), 4),
        n_queries=len(query_results),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_result(result: ExperimentResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))


def load_result(path: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json(path.read_text())


def best_config(results: list[ExperimentResult], primary: str = "mrr") -> ExperimentResult:
    """Return the result with the highest primary metric."""
    return max(results, key=lambda r: r.metrics.get(primary, 0.0))
