"""
Cross-encoder reranker for P3 RAG evaluation.

Uses a sentence-transformers CrossEncoder to re-score and reorder
candidates returned by a base retriever. Implements RetrieverProtocol
so RerankerRetriever can substitute any base retriever transparently.

Install: pip install "rag-pipeline-systematic-evals[reranking]"
Model:   cross-encoder/ms-marco-MiniLM-L-6-v2  (small, fast, strong)

The model is lazy-loaded on first use so importing this module does not
require sentence-transformers unless reranking is actually invoked.
"""

from __future__ import annotations

from rag_common.models import RetrievalResult
from rag_common.retrievers import RetrieverProtocol


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results:
            return results
        self._load()
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: float(x[0]), reverse=True)
        return [r for _, r in ranked[:top_k]]


class RerankerRetriever:
    """
    Wraps a base retriever with cross-encoder reranking.

    Fetches `top_k * fetch_multiplier` candidates from the base retriever,
    reranks with a CrossEncoderReranker, and returns the best `top_k`.
    The larger fetch pool (default 4×) ensures the cross-encoder sees
    enough diversity to meaningfully reorder results.
    """

    def __init__(
        self,
        base: RetrieverProtocol,
        reranker: CrossEncoderReranker,
        fetch_multiplier: int = 4,
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._fetch_multiplier = fetch_multiplier

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        candidates = self._base.retrieve(query, top_k=top_k * self._fetch_multiplier)
        return self._reranker.rerank(query, candidates, top_k)
