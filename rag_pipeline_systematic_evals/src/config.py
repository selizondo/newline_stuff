"""
Configuration models for the systematic RAG evaluation pipeline (P3).

All config objects are Pydantic models so they validate at construction time,
serialise cleanly to JSON for experiment tracking, and can be reconstructed
from saved experiment files.

Grid search space
-----------------
    4 ChunkConfigs × 2 EmbedConfigs × 3 RetrievalMethods = 24 ExperimentConfigs

Each ExperimentConfig is the unit of work for grid_search.py and produces one
EvaluationResult written to experiments/.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE   = "sentence"
    SEMANTIC   = "semantic"


class EmbedModel(str, Enum):
    SMALL = "text-embedding-3-small"   # 1536-dim, fast, cheap
    LARGE = "text-embedding-3-large"   # 3072-dim, slower, higher quality


class RetrievalMethod(str, Enum):
    VECTOR = "vector"
    BM25   = "bm25"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

class ChunkConfig(BaseModel):
    """
    Parameters for one chunking strategy.

    `parser` is recorded here (not in a separate config) because switching PDF
    parsers changes the extracted text and therefore invalidates comparisons —
    keep it consistent across all runs in a single experiment grid.
    """
    strategy: ChunkStrategy
    parser: Literal["pdfplumber"] = "pdfplumber"

    # Fixed-size / sliding-window params
    chunk_size: int = 256            # characters
    overlap: int    = 50             # characters

    # Sentence-based params
    sentences_per_chunk: int = 5
    overlap_sentences: int   = 1

    # Semantic params
    breakpoint_threshold: float = 0.65
    max_sentences: int          = 10

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkConfig:
        if self.strategy == ChunkStrategy.FIXED_SIZE and self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if (self.strategy == ChunkStrategy.SENTENCE
                and self.overlap_sentences >= self.sentences_per_chunk):
            raise ValueError("overlap_sentences must be less than sentences_per_chunk")
        return self

    def label(self) -> str:
        """Short human-readable identifier used in experiment IDs and chart labels."""
        if self.strategy == ChunkStrategy.FIXED_SIZE:
            return f"fixed_{self.chunk_size}_ol{self.overlap}"
        if self.strategy == ChunkStrategy.SENTENCE:
            return f"sentence_{self.sentences_per_chunk}s_ol{self.overlap_sentences}"
        return f"semantic_t{self.breakpoint_threshold}_max{self.max_sentences}"


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------

class EmbedConfig(BaseModel):
    model: EmbedModel = EmbedModel.SMALL
    batch_size: int   = 100          # chunks per OpenAI API call
    # Embeddings are cached under cache_dir/{model}/{chunk_config_label}.pkl
    # to avoid re-embedding when only the retrieval method changes.
    cache_dir: Path   = Path("data/embed_cache")

    def label(self) -> str:
        return self.model.value.split("-")[-1]   # "small" or "large"


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

class RetrievalConfig(BaseModel):
    method: RetrievalMethod = RetrievalMethod.VECTOR
    top_k: int              = 5
    # Hybrid-only: weight for dense scores; (1 - alpha) for BM25.
    alpha: float            = 0.5

    def label(self) -> str:
        if self.method == RetrievalMethod.HYBRID:
            return f"hybrid_a{self.alpha}"
        return self.method.value


# ---------------------------------------------------------------------------
# Full experiment configuration (one cell in the grid)
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """
    One experiment = one cell in the chunking × embedding × retrieval grid.

    `experiment_id` is derived from the component labels so it is human-readable
    in saved JSON files and Rich table output:
        e.g. "fixed_256_ol50__small__vector"
        e.g. "fixed_256_ol50__small__vector__reranked"
    """
    chunk:     ChunkConfig
    embed:     EmbedConfig
    retrieval: RetrievalConfig

    # How many synthetic QA pairs to generate per chunking config.
    # The spec requires ≥ 20; default is 25 to have a small buffer.
    qa_pairs_per_config: int = 25
    use_reranking: bool = False

    @property
    def experiment_id(self) -> str:
        base = f"{self.chunk.label()}__{self.embed.label()}__{self.retrieval.label()}"
        return f"{base}__reranked" if self.use_reranking else base


# ---------------------------------------------------------------------------
# Evaluation result (written to experiments/ after each run)
# ---------------------------------------------------------------------------

class MetricsResult(BaseModel):
    recall_at_k:    dict[str, float]   # {"1": …, "3": …, "5": …, "10": …}
    precision_at_k: dict[str, float]
    mrr:            float
    map_score:      float
    ndcg_at_k:      dict[str, float]
    total_queries:  int
    avg_retrieval_time_s: float = 0.0


class EvaluationResult(BaseModel):
    experiment_id:  str
    config:         ExperimentConfig
    metrics:        MetricsResult
    # Per-query detail kept for debugging and future LLM-as-Judge scoring.
    query_results:  list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Default grid used by grid_search.py
# ---------------------------------------------------------------------------

def default_chunk_configs() -> list[ChunkConfig]:
    """Four chunking configurations covering the minimum grid requirement."""
    return [
        ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256,  overlap=50),
        ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=512,  overlap=100),
        ChunkConfig(strategy=ChunkStrategy.SENTENCE,   sentences_per_chunk=5, overlap_sentences=1),
        ChunkConfig(strategy=ChunkStrategy.SEMANTIC,   breakpoint_threshold=0.65, max_sentences=10),
    ]


def default_embed_configs() -> list[EmbedConfig]:
    return [
        EmbedConfig(model=EmbedModel.SMALL),
        EmbedConfig(model=EmbedModel.LARGE),
    ]


def default_retrieval_configs() -> list[RetrievalConfig]:
    return [
        RetrievalConfig(method=RetrievalMethod.VECTOR),
        RetrievalConfig(method=RetrievalMethod.BM25),
        RetrievalConfig(method=RetrievalMethod.HYBRID, alpha=0.5),
    ]


def build_experiment_grid(
    chunk_configs:     list[ChunkConfig]     | None = None,
    embed_configs:     list[EmbedConfig]     | None = None,
    retrieval_configs: list[RetrievalConfig] | None = None,
    qa_pairs_per_config: int = 25,
    use_reranking: bool = False,
) -> list[ExperimentConfig]:
    """
    Produce the full cross-product grid of experiments.

    Defaults give 4 × 2 × 3 = 24 experiments, meeting the spec minimum of 12.
    """
    chunks    = chunk_configs     or default_chunk_configs()
    embeds    = embed_configs     or default_embed_configs()
    retrievals = retrieval_configs or default_retrieval_configs()

    return [
        ExperimentConfig(
            chunk=c, embed=e, retrieval=r,
            qa_pairs_per_config=qa_pairs_per_config,
            use_reranking=use_reranking,
        )
        for c in chunks
        for e in embeds
        for r in retrievals
    ]
