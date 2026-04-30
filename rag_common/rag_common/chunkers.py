"""
Text chunking strategies shared by P3 (rag_pipeline_systematic_evals)
and P4 (rag_pipeline_experimentation).

All chunkers implement the same interface:

    chunker.chunk(text, metadata={}) -> list[Chunk]

`metadata` is merged into every Chunk.metadata so callers can attach
provenance (source filename, document_id, parser used, etc.) without
each chunker needing to know about it.

Strategies provided here
------------------------
FixedSizeChunker   — character-window split; respects word boundaries
SentenceBasedChunker — groups N sentences with sentence-level overlap (NLTK)
SemanticChunker    — splits at low-similarity sentence boundaries (needs embed_fn)

Project-specific chunkers (RecursiveChunker, SlidingWindowChunker) live in
each project's own chunkers_ext.py and follow the same interface.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from rag_common.models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _split_sentences(text: str) -> list[str]:
    """
    Sentence tokeniser using NLTK punkt.

    Falls back to a simple regex split if NLTK data is unavailable so tests
    can run offline without downloading the punkt corpus.
    """
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception:
        # Regex fallback: split on '. ', '! ', '? '
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class FixedSizeChunker:
    """
    Splits text into fixed-size character windows with overlap.

    Word-boundary awareness: chunks always end at a space (or the string end)
    so words are never split mid-token. The actual chunk length may therefore
    be up to one word shorter than `chunk_size`.

    Args:
        chunk_size: target character length per chunk
        overlap:    character overlap between consecutive chunks; must be < chunk_size
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Walk back to the nearest word boundary so we don't cut mid-word.
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_index=idx,
                    method="fixed_size",
                    start_char=start,
                    end_char=end,
                    metadata={
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        **metadata,
                    },
                ))
                idx += 1

            # Advance by (chunk_size - overlap), but at least 1 char to avoid
            # an infinite loop when text is shorter than overlap.
            step = max(1, self.chunk_size - self.overlap)
            start += step

        return chunks


# ---------------------------------------------------------------------------
# SentenceBasedChunker
# ---------------------------------------------------------------------------

class SentenceBasedChunker:
    """
    Groups consecutive sentences into chunks with sentence-level overlap.

    Args:
        sentences_per_chunk: how many sentences form one chunk
        overlap_sentences:   how many trailing sentences carry over to the next chunk
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> None:
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be less than sentences_per_chunk")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = _split_sentences(text)
        chunks: list[Chunk] = []
        idx = 0
        step = self.sentences_per_chunk - self.overlap_sentences
        pos = 0

        while pos < len(sentences):
            window = sentences[pos : pos + self.sentences_per_chunk]
            content = " ".join(window).strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_index=idx,
                    method="sentence",
                    metadata={
                        "sentences_per_chunk": self.sentences_per_chunk,
                        "overlap_sentences": self.overlap_sentences,
                        "sentence_start": pos,
                        "sentence_end": pos + len(window),
                        **metadata,
                    },
                ))
                idx += 1
            pos += step

        return chunks


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Splits text at semantic boundaries detected by a drop in cosine similarity
    between adjacent sentence embeddings.

    Algorithm
    ---------
    1. Split text into sentences.
    2. Embed each sentence via `embed_fn`.
    3. Compute cosine similarity between consecutive sentence pairs.
    4. Mark a boundary wherever similarity < `breakpoint_threshold`.
    5. Collect sentences between boundaries into a chunk; if a segment exceeds
       `max_sentences`, force-split it to avoid oversized chunks.

    Args:
        embed_fn:             callable that accepts list[str] and returns
                              np.ndarray of shape (N, D); must handle batches.
        breakpoint_threshold: cosine similarity below this → new chunk boundary.
                              Typical range: 0.5 – 0.8; lower = fewer, larger chunks.
        max_sentences:        hard cap on sentences per chunk; prevents a run of
                              high-similarity sentences from creating a huge chunk.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        breakpoint_threshold: float = 0.65,
        max_sentences: int = 10,
    ) -> None:
        self.embed_fn = embed_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.max_sentences = max_sentences

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = _split_sentences(text)

        if len(sentences) <= 1:
            if sentences:
                return [Chunk(
                    content=sentences[0],
                    chunk_index=0,
                    method="semantic",
                    metadata={"breakpoint_threshold": self.breakpoint_threshold, **metadata},
                )]
            return []

        embeddings = self.embed_fn(sentences)  # shape (N, D)
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        # Build segment boundaries: indices where a new chunk starts.
        boundaries = [0]
        for i, sim in enumerate(similarities):
            since_last = i + 1 - boundaries[-1]
            if sim < self.breakpoint_threshold or since_last >= self.max_sentences:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        chunks: list[Chunk] = []
        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            content = " ".join(sentences[start:end]).strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_index=idx,
                    method="semantic",
                    metadata={
                        "breakpoint_threshold": self.breakpoint_threshold,
                        "sentence_start": start,
                        "sentence_end": end,
                        **metadata,
                    },
                ))

        return chunks
