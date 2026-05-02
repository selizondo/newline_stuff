"""CrossEncoderReranker — local cross-encoder reranking for retrieval results."""

from __future__ import annotations

from src.base import BaseReranker
from rag_common.models import RetrievalResult


class CrossEncoderReranker(BaseReranker):
    """Re-score retrieved chunks using a local cross-encoder model.

    The cross-encoder evaluates each (query, chunk) pair jointly, producing
    more accurate relevance scores than bi-encoder dot-product at the cost of
    O(N) forward passes per query. Use on a small candidate set (top-20 or
    fewer) and return the best top_k.

    Args:
        model_name: Any sentence-transformers cross-encoder model.
                    Default: "cross-encoder/ms-marco-MiniLM-L-6-v2" — fast,
                    strong MS MARCO–trained model suited for passage retrieval.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first rerank call

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Score all (query, chunk) pairs and return top_k by cross-encoder score.

        Args:
            query:   User question.
            results: Candidate RetrievalResults from first-stage retrieval.
            top_k:   Number of results to return after reranking.

        Returns:
            top_k RetrievalResults sorted by cross-encoder score (descending).
        """
        if not results:
            return results

        self._load()

        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)

        reranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        return [
            RetrievalResult(
                chunk=r.chunk,
                score=float(s),
                retriever_type="cross_encoder",
            )
            for s, r in reranked[:top_k]
        ]
