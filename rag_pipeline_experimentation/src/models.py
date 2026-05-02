"""
P4-specific data models.

rag_common.models.Chunk and RetrievalResult are reused directly — Chunk
already carries `document_id` and `source` fields for P4's multi-doc setup.
This module adds the higher-level models needed for answer generation,
judge scoring, and experiment tracking.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Source PDF metadata; created during ingestion, stored alongside the index."""
    id: str                              # paper_id from dataset (arXiv ID)
    title: str = ""
    source: str                          # filename (e.g. "2310.12345.pdf")
    page_count: int = 0
    char_count: int = 0
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    source: str                          # filename
    page_number: int | None = None
    text_snippet: str                    # first 200 chars of the chunk
    relevance_score: float | None = None


class QAResponse(BaseModel):
    query: str
    answer: str
    model: str = ""
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: list[dict] = Field(default_factory=list)
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0


class JudgeScore(BaseModel):
    """LLM-as-Judge scores on a 1–5 scale for one generated answer."""
    relevance: float        # Does the answer address the question?
    accuracy: float         # Is the information factually correct?
    completeness: float     # Is the answer sufficiently thorough?
    citation_quality: float # Are sources properly attributed?
    reasoning: str = ""     # Judge's brief rationale

    @property
    def average(self) -> float:
        return (self.relevance + self.accuracy + self.completeness + self.citation_quality) / 4


class QueryResult(BaseModel):
    """Per-query detail stored inside ExperimentResult."""
    query_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    retrieval_time_s: float = 0.0
    judge_score: JudgeScore | None = None


class ExperimentResult(BaseModel):
    """One completed experiment cell — written to experiments/results/{id}.json."""
    experiment_id: str
    config: dict                          # serialised ExperimentConfig
    metrics: dict[str, float]             # recall@5, precision@5, mrr, ndcg@5
    generation_metrics: dict[str, float] = Field(default_factory=dict)  # avg judge dims
    llm_model: str = ""                   # model used for answer generation + judging
    query_results: list[QueryResult] = Field(default_factory=list)
    avg_latency_s: float = 0.0
    n_queries: int = 0
    timestamp: str = ""
