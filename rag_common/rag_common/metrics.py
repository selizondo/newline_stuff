"""
IR evaluation metrics for RAG retrieval.

All per-query functions accept:
  retrieved_ids : ordered list of chunk ID strings (best → worst)
  relevant_ids  : set of ground-truth chunk ID strings

Aggregate functions accept:
  query_results : list of (retrieved_ids, relevant_ids) tuples, one per query

Precision@K note
----------------
When each query has exactly one ground-truth chunk (e.g. P3 synthetic QA),
Precision@K is mathematically capped at 1/K (e.g. 0.20 for K=5). This is
expected — use MRR and Recall@K as primary quality signals in that setup.

NDCG formula used
-----------------
  DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i + 1)   (binary rel: 0 or 1)
  IDCG@K = DCG of the perfect ranking (all relevant docs at the top)
  NDCG@K = DCG@K / IDCG@K
The +1 inside log2 is the standard TREC/BEIR convention so that rank-1 gets
full credit (log2(2) = 1) rather than undefined credit (log2(1) = 0).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks found in the top-k results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant result; 0.0 if none found."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Area under the precision-recall curve for a single query.

    Computed as:  Σ P@rank / |relevant|  for every rank where a hit occurs.
    Normalising by |relevant| (not by the number of hits) penalises systems
    that fail to retrieve all relevant documents.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    cumulative = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            cumulative += hits / rank
    return cumulative / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    total = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            total += 1.0 / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised DCG at k; IDCG is the DCG of the ideal (perfect) ranking."""
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / idcg


# ---------------------------------------------------------------------------
# Aggregate metrics across queries
# ---------------------------------------------------------------------------

# Type alias: each element is (retrieved_ids, relevant_ids) for one query.
QueryResults = list[tuple[list[str], set[str]]]


def mrr(query_results: QueryResults) -> float:
    """Mean Reciprocal Rank across all queries."""
    if not query_results:
        return 0.0
    return sum(reciprocal_rank(r, g) for r, g in query_results) / len(query_results)


def map_score(query_results: QueryResults) -> float:
    """Mean Average Precision across all queries."""
    if not query_results:
        return 0.0
    return sum(average_precision(r, g) for r, g in query_results) / len(query_results)


def mean_recall_at_k(query_results: QueryResults, k: int) -> float:
    if not query_results:
        return 0.0
    return sum(recall_at_k(r, g, k) for r, g in query_results) / len(query_results)


def mean_precision_at_k(query_results: QueryResults, k: int) -> float:
    if not query_results:
        return 0.0
    return sum(precision_at_k(r, g, k) for r, g in query_results) / len(query_results)


def mean_ndcg_at_k(query_results: QueryResults, k: int) -> float:
    if not query_results:
        return 0.0
    return sum(ndcg_at_k(r, g, k) for r, g in query_results) / len(query_results)


# ---------------------------------------------------------------------------
# Convenience wrapper: evaluate a full retrieval run
# ---------------------------------------------------------------------------

def evaluate(query_results: QueryResults, k: int = 5) -> dict[str, float]:
    """
    Run all standard IR metrics for a retrieval experiment and return a dict.

    Keys: recall@k, precision@k, mrr, map, ndcg@k

    Usage::

        from rag_common import metrics
        results = [(retrieved_ids, relevant_ids), ...]
        scores = metrics.evaluate(results, k=5)
    """
    return {
        f"recall@{k}": mean_recall_at_k(query_results, k),
        f"precision@{k}": mean_precision_at_k(query_results, k),
        "mrr": mrr(query_results),
        "map": map_score(query_results),
        f"ndcg@{k}": mean_ndcg_at_k(query_results, k),
    }
