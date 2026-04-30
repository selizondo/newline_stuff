"""
Grid-search orchestrator for P3.

Runs all 24 experiment cells (4 chunk × 2 embed × 3 retrieval) and writes one
EvaluationResult JSON per cell to `experiment_dir/`.

Execution order is chosen to minimise redundant work:
    parse PDF  (once)
    └─ for each chunk_config:         chunk + QA-generate (cached)
       └─ for each embed_config:      embed + build FAISS index (cached)
          └─ for each retrieval_config:  build retriever, evaluate, save

BM25 does not use embeddings but is still iterated inside the embed loop so
every (chunk, embed, retrieval) triple produces a file — makes the results
table uniform.

Resuming: if `experiment_dir/{experiment_id}.json` exists the cell is skipped
unless `force=True` is passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from rag_common.chunkers import FixedSizeChunker, SemanticChunker, SentenceBasedChunker
from rag_common.models import Chunk
from rag_common.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag_common.vector_store import FAISSVectorStore

from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EvaluationResult,
    ExperimentConfig, RetrievalConfig, RetrievalMethod,
    build_experiment_grid,
)
from src.embedders import embed_chunks, embed_texts
from src.evaluator import evaluate
from src.parsers import parse_pdf
from src.qa_generator import generate_qa_dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_grid_search(
    pdf_path: Path,
    experiment_dir: Path = Path("experiments"),
    qa_dir: Path = Path("data/qa_datasets"),
    n_pairs: int = 25,
    chunk_configs: list[ChunkConfig] | None = None,
    embed_configs: list[EmbedConfig] | None = None,
    retrieval_configs: list[RetrievalConfig] | None = None,
    force: bool = False,
) -> list[EvaluationResult]:
    """
    Run the full evaluation grid against `pdf_path`.

    Args:
        pdf_path:          Path to the PDF document to evaluate against.
        experiment_dir:    Directory to write one JSON per completed experiment.
        qa_dir:            Directory for cached QA datasets.
        n_pairs:           Synthetic QA pairs to generate per chunk config.
        chunk_configs:     Override the default 4 chunk configs.
        embed_configs:     Override the default 2 embed configs.
        retrieval_configs: Override the default 3 retrieval configs.
        force:             Re-run even if result file already exists.

    Returns:
        Ordered list of EvaluationResult, one per completed experiment.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(
        chunk_configs=chunk_configs,
        embed_configs=embed_configs,
        retrieval_configs=retrieval_configs,
        qa_pairs_per_config=n_pairs,
    )

    doc = parse_pdf(pdf_path)
    full_text = doc.full_text

    results: list[EvaluationResult] = []
    seen_chunk_labels: dict[str, list[Chunk]] = {}

    for config in grid:
        result_path = experiment_dir / f"{config.experiment_id}.json"
        if not force and result_path.exists():
            results.append(_load_result(result_path))
            continue

        chunk_label = config.chunk.label()

        if chunk_label not in seen_chunk_labels:
            seen_chunk_labels[chunk_label] = _chunk_document(
                full_text, config.chunk, config.embed
            )
        chunks = seen_chunk_labels[chunk_label]

        dataset = generate_qa_dataset(
            chunks,
            chunk_config_label=chunk_label,
            n_pairs=n_pairs,
            qa_dir=qa_dir,
        )

        embeddings = embed_chunks(chunks, config.embed, chunk_label)

        store = FAISSVectorStore()
        store.add(chunks, embeddings)

        embed_fn = _make_embed_fn(config.embed)
        retriever = _make_retriever(config.retrieval, store, chunks, embed_fn)

        result = evaluate(dataset, retriever, config)
        _save_result(result, result_path)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_document(
    text: str,
    chunk_config: ChunkConfig,
    embed_config: EmbedConfig,
) -> list[Chunk]:
    if chunk_config.strategy == ChunkStrategy.FIXED_SIZE:
        return FixedSizeChunker(
            chunk_size=chunk_config.chunk_size,
            overlap=chunk_config.overlap,
        ).chunk(text)

    if chunk_config.strategy == ChunkStrategy.SENTENCE:
        return SentenceBasedChunker(
            sentences_per_chunk=chunk_config.sentences_per_chunk,
            overlap_sentences=chunk_config.overlap_sentences,
        ).chunk(text)

    return SemanticChunker(
        embed_fn=_make_embed_fn(embed_config),
        breakpoint_threshold=chunk_config.breakpoint_threshold,
        max_sentences=chunk_config.max_sentences,
    ).chunk(text)


def _make_embed_fn(embed_config: EmbedConfig) -> Callable[[list[str]], np.ndarray]:
    """Return a batched embed function bound to one EmbedConfig."""
    return lambda texts: embed_texts(texts, embed_config)


def _make_retriever(
    retrieval_config: RetrievalConfig,
    store: FAISSVectorStore,
    chunks: list[Chunk],
    embed_fn: Callable[[list[str]], np.ndarray],
):
    if retrieval_config.method == RetrievalMethod.BM25:
        return BM25Retriever(chunks)

    dense = DenseRetriever(store, embed_fn)

    if retrieval_config.method == RetrievalMethod.VECTOR:
        return dense

    return HybridRetriever(
        dense=dense,
        bm25=BM25Retriever(chunks),
        alpha=retrieval_config.alpha,
    )


def _save_result(result: EvaluationResult, path: Path) -> None:
    path.write_text(result.model_dump_json(indent=2))


def _load_result(path: Path) -> EvaluationResult:
    return EvaluationResult.model_validate_json(path.read_text())
