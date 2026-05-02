"""
Experiment configuration for P4.

Supports both programmatic construction and YAML-driven grid loading.

YAML format (config/experiments/baseline.yaml):
    chunking_strategies:
      - strategy: fixed
        chunk_size: 512
        overlap: 64
    embedding_models:
      - model: all-MiniLM-L6-v2
    retrieval_methods:
      - method: dense
        top_k: 5
    n_queries: 50
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    FIXED          = "fixed"
    RECURSIVE      = "recursive"
    SLIDING_WINDOW = "sliding_window"
    SENTENCE       = "sentence"
    SEMANTIC       = "semantic"


class EmbedModelName(str, Enum):
    MINILM   = "all-MiniLM-L6-v2"       # 384d, fast
    MPNET    = "all-mpnet-base-v2"       # 768d, higher quality
    QA_MINI  = "multi-qa-MiniLM-L6-cos-v1"  # 384d, QA-optimised


class RetrievalMethod(str, Enum):
    DENSE  = "dense"
    BM25   = "bm25"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------

class ChunkConfig(BaseModel):
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    overlap: int = 100
    # Sentence-based params
    sentences_per_chunk: int = 5
    overlap_sentences: int = 1
    # Sliding-window params
    window_size: int = 10
    step: int = 5
    # Semantic params
    breakpoint_threshold: float = 0.65
    max_sentences: int = 10

    def label(self) -> str:
        s = self.strategy.value
        if self.strategy in (ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE):
            return f"{s}_{self.chunk_size}_ol{self.overlap}"
        if self.strategy == ChunkStrategy.SLIDING_WINDOW:
            return f"sliding_w{self.window_size}_s{self.step}"
        if self.strategy == ChunkStrategy.SENTENCE:
            return f"sentence_{self.sentences_per_chunk}s_ol{self.overlap_sentences}"
        return f"semantic_t{self.breakpoint_threshold}_max{self.max_sentences}"


class EmbedConfig(BaseModel):
    model: EmbedModelName = EmbedModelName.MINILM
    batch_size: int = 64
    cache_dir: Path = Path("data/embed_cache")

    def label(self) -> str:
        # "all-MiniLM-L6-v2" → "minilm-l6"
        return self.model.value.lower().replace("all-", "").split("-v")[0]


class RetrievalConfig(BaseModel):
    method: RetrievalMethod = RetrievalMethod.DENSE
    top_k: int = 5
    alpha: float = 0.6   # hybrid: weight for dense scores

    def label(self) -> str:
        if self.method == RetrievalMethod.HYBRID:
            return f"hybrid_a{self.alpha}"
        return self.method.value


# ---------------------------------------------------------------------------
# Full experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    chunk: ChunkConfig
    embed: EmbedConfig
    retrieval: RetrievalConfig
    n_queries: int = 50

    @property
    def experiment_id(self) -> str:
        return f"{self.chunk.label()}__{self.embed.label()}__{self.retrieval.label()}"


# ---------------------------------------------------------------------------
# YAML loader + grid builder
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_experiment_grid(
    chunk_configs: list[ChunkConfig] | None = None,
    embed_configs: list[EmbedConfig] | None = None,
    retrieval_configs: list[RetrievalConfig] | None = None,
    n_queries: int = 50,
) -> list[ExperimentConfig]:
    """Full cross-product grid from component lists."""
    chunks = chunk_configs or _default_chunk_configs()
    embeds = embed_configs or _default_embed_configs()
    retrievals = retrieval_configs or _default_retrieval_configs()

    return [
        ExperimentConfig(chunk=c, embed=e, retrieval=r, n_queries=n_queries)
        for c in chunks
        for e in embeds
        for r in retrievals
    ]


def build_grid_from_yaml(path: Path) -> list[ExperimentConfig]:
    """Load experiment grid from a YAML config file."""
    cfg = load_yaml(path)
    n_queries = cfg.get("n_queries", 50)

    chunks = [ChunkConfig(**c) for c in cfg.get("chunking_strategies", [])]
    embeds = [EmbedConfig(**e) for e in cfg.get("embedding_models", [])]
    retrievals = [RetrievalConfig(**r) for r in cfg.get("retrieval_methods", [])]

    if not chunks:
        chunks = _default_chunk_configs()
    if not embeds:
        embeds = _default_embed_configs()
    if not retrievals:
        retrievals = _default_retrieval_configs()

    return build_experiment_grid(chunks, embeds, retrievals, n_queries=n_queries)


def _default_chunk_configs() -> list[ChunkConfig]:
    return [
        ChunkConfig(strategy=ChunkStrategy.FIXED,          chunk_size=512, overlap=64),
        ChunkConfig(strategy=ChunkStrategy.RECURSIVE,      chunk_size=512, overlap=100),
        ChunkConfig(strategy=ChunkStrategy.SLIDING_WINDOW, window_size=10, step=5),
    ]


def _default_embed_configs() -> list[EmbedConfig]:
    return [
        EmbedConfig(model=EmbedModelName.MINILM),
        EmbedConfig(model=EmbedModelName.MPNET),
    ]


def _default_retrieval_configs() -> list[RetrievalConfig]:
    return [
        RetrievalConfig(method=RetrievalMethod.DENSE),
        RetrievalConfig(method=RetrievalMethod.HYBRID, alpha=0.6),
    ]
