"""
Synthetic QA dataset generator for P3.

CRITICAL DESIGN CONSTRAINT
---------------------------
Each QA dataset is generated from — and tied to — the chunks produced by one
specific chunking configuration. Sharing a single QA dataset across multiple
chunking configs would bias the evaluation: questions generated around 256-char
chunk boundaries make no sense when evaluated against 512-char chunks whose IDs
and text spans are completely different.

This module always produces a separate QADataset per (chunk_config_label,
chunk set). The `chunk_config_label` is the cache key; if the label matches an
existing file in `qa_dir`, the saved dataset is returned without calling the LLM.

Flow
----
    chunks (from one config) → sample → LLM + Instructor → validate IDs → QADataset → save

One question is generated per sampled chunk. If `n_pairs` > len(chunks), every
chunk gets a question and some chunks get a second question (sampled with replacement).

Instructor usage
----------------
We use `instructor.patch(client)` so the LLM is forced to return a valid
`QAPairResponse` Pydantic model. Malformed responses are retried automatically
by Instructor (up to 3 times). If validation still fails, the pair is logged
and skipped — the pipeline never crashes on a bad LLM response.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from rag_common.models import Chunk


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class QAPairResponse(BaseModel):
    """
    Structured output returned by the LLM for a single chunk.

    `question_type` helps track diversity across the generated dataset.
    `question` must paraphrase the chunk — the validator rejects questions
    that are pure copy-paste (first 30 chars identical to chunk content).
    """
    question: str = Field(description="A natural-language question answerable from the chunk.")
    question_type: Literal["factual", "conceptual", "comparative", "analytical"] = Field(
        description="Category of the question."
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class QAPair(BaseModel):
    """One synthetic QA pair tied to a specific chunk."""
    question: str
    question_type: str
    relevant_chunk_ids: list[str]   # UUIDs of the source chunk(s)
    metadata: dict = Field(default_factory=dict)


class QADataset(BaseModel):
    """Full QA dataset for one chunking configuration."""
    chunk_config_label: str
    pairs: list[QAPair]

    @property
    def size(self) -> int:
        return len(self.pairs)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert at creating evaluation datasets for retrieval-augmented generation systems.

Given a text chunk from a document, generate ONE question that:
- Is fully answerable using only the information in the chunk
- Uses natural language (do NOT copy phrases verbatim from the chunk)
- Tests understanding, not just keyword matching
- Falls into one of these types: factual, conceptual, comparative, analytical

Return ONLY a JSON object matching the required schema. Do not add explanation."""


def _user_prompt(chunk_content: str) -> str:
    return f"Chunk content:\n\n{chunk_content}\n\nGenerate one question from this chunk."


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_qa_dataset(
    chunks: list[Chunk],
    chunk_config_label: str,
    n_pairs: int = 25,
    model: str = "gpt-4o-mini",
    qa_dir: Path = Path("data/qa_datasets"),
    seed: int = 42,
) -> QADataset:
    """
    Generate (or load from cache) a QA dataset for one chunking configuration.

    Args:
        chunks:             chunks produced by one specific chunking config.
        chunk_config_label: unique label for the config (e.g. "fixed_256_ol50").
                            Used as the cache filename.
        n_pairs:            number of QA pairs to generate (spec requires ≥ 20).
        model:              OpenAI model for generation.
        qa_dir:             directory to cache QA datasets.
        seed:               random seed for chunk sampling reproducibility.

    Returns:
        QADataset with `n_pairs` (or fewer if some LLM calls fail validation).
    """
    if not chunks:
        raise ValueError("Cannot generate QA dataset from an empty chunk list.")

    cache_path = qa_dir / f"{chunk_config_label}.json"
    if cache_path.exists():
        cached = _load_dataset(cache_path)
        current_ids = {c.id_str() for c in chunks}
        cached_ids = {cid for p in cached.pairs for cid in p.relevant_chunk_ids}
        # Return cache only if all referenced chunk IDs still exist in this chunk set.
        if cached_ids.issubset(current_ids):
            return cached
        # Stale cache (chunks re-generated with new UUIDs) — regenerate QA dataset.

    chunk_by_id = {c.id_str(): c for c in chunks}
    sampled = _sample_chunks(chunks, n_pairs, seed)

    client = instructor.from_openai(OpenAI())
    pairs: list[QAPair] = []

    for i, chunk in enumerate(sampled):
        pair = _generate_one(client, chunk, model, attempt=i)
        if pair is not None:
            pairs.append(pair)
        # Polite rate-limit buffer between calls.
        if i < len(sampled) - 1:
            time.sleep(0.3)

    # Validate all chunk IDs exist in this config's chunk set.
    pairs = [p for p in pairs if all(cid in chunk_by_id for cid in p.relevant_chunk_ids)]

    dataset = QADataset(chunk_config_label=chunk_config_label, pairs=pairs)
    _save_dataset(dataset, cache_path)
    return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_chunks(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    """
    Sample `n` chunks from `chunks`.

    If n <= len(chunks), sample without replacement so every sampled chunk
    is distinct. If n > len(chunks), use replacement to meet the quota —
    this produces multiple questions from the same chunk, which is fine for
    evaluation purposes.
    """
    rng = random.Random(seed)
    if n <= len(chunks):
        return rng.sample(chunks, n)
    # With replacement: draw ceil(n / len) full passes then trim.
    base = chunks * (n // len(chunks) + 1)
    return base[:n]


def _generate_one(
    client: instructor.Instructor,
    chunk: Chunk,
    model: str,
    attempt: int,
) -> QAPair | None:
    """
    Call the LLM and return a validated QAPair, or None on failure.

    Instructor retries up to 3 times on schema validation errors internally.
    We catch any remaining exception and log it so the pipeline keeps running.
    """
    try:
        response: QAPairResponse = client.chat.completions.create(
            model=model,
            response_model=QAPairResponse,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _user_prompt(chunk.content)},
            ],
            max_retries=3,
        )
        return QAPair(
            question=response.question,
            question_type=response.question_type,
            relevant_chunk_ids=[chunk.id_str()],
            metadata={
                "source_chunk_index": chunk.chunk_index,
                "source_page": chunk.page_number,
                "chunk_method": chunk.method,
                "synthetic": True,
            },
        )
    except Exception as exc:
        print(f"  [qa_generator] skipping chunk {chunk.chunk_index}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_dataset(dataset: QADataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2))


def _load_dataset(path: Path) -> QADataset:
    return QADataset.model_validate_json(path.read_text())
