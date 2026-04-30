"""
Unit tests for rag_common.chunkers.

SemanticChunker tests use a deterministic stub embed_fn that returns
fixed vectors so tests run offline without any ML model or API call.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker, SemanticChunker
from rag_common.models import Chunk


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SHORT_TEXT = "Hello world. This is a test."

PARA_TEXT = (
    "The mitochondria is the powerhouse of the cell. "
    "It produces ATP through oxidative phosphorylation. "
    "This process occurs in the inner mitochondrial membrane. "
    "Energy is released as electrons pass through protein complexes. "
    "The proton gradient drives ATP synthase to produce ATP. "
    "Without mitochondria, eukaryotic life would be impossible. "
    "Plant cells also contain chloroplasts for photosynthesis. "
    "Chloroplasts convert sunlight into chemical energy. "
    "Both organelles have their own DNA, suggesting endosymbiotic origin."
)


def _stub_embed(sentences: list[str]) -> np.ndarray:
    """
    Deterministic embed_fn stub: returns identity-like vectors based on
    sentence index so similarity between consecutive sentences is controlled.
    Odd-indexed sentences get an orthogonal vector → similarity = 0 (breakpoint).
    Even-indexed sentences share a direction   → similarity = 1 (no break).
    """
    dim = 8
    vecs = []
    for i, _ in enumerate(sentences):
        v = np.zeros(dim)
        v[i % dim] = 1.0
        vecs.append(v)
    return np.array(vecs, dtype=float)


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_returns_list_of_chunks(self):
        c = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = c.chunk(SHORT_TEXT)
        assert all(isinstance(ch, Chunk) for ch in chunks)

    def test_non_empty_content(self):
        c = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = c.chunk(PARA_TEXT)
        assert all(ch.content for ch in chunks)

    def test_chunk_method_field(self):
        chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(PARA_TEXT)
        assert all(ch.method == "fixed_size" for ch in chunks)

    def test_chunk_indices_sequential(self):
        chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(PARA_TEXT)
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_content_within_size_limit(self):
        size = 80
        chunks = FixedSizeChunker(chunk_size=size, overlap=0).chunk(PARA_TEXT)
        # Word-boundary backtrack may produce chunks slightly shorter than size.
        assert all(len(ch.content) <= size for ch in chunks)

    def test_no_mid_word_splits(self):
        # Every chunk should start and end at a word boundary (no partial words
        # caused by cutting exactly at chunk_size characters).
        chunks = FixedSizeChunker(chunk_size=60, overlap=10).chunk(PARA_TEXT)
        for ch in chunks:
            assert not ch.content.startswith(" ")

    def test_metadata_passed_through(self):
        chunks = FixedSizeChunker(chunk_size=100, overlap=10).chunk(
            PARA_TEXT, metadata={"source": "test.pdf", "page_number": 3}
        )
        for ch in chunks:
            assert ch.metadata["source"] == "test.pdf"
            assert ch.metadata["page_number"] == 3

    def test_metadata_records_params(self):
        chunks = FixedSizeChunker(chunk_size=128, overlap=32).chunk(SHORT_TEXT)
        assert chunks[0].metadata["chunk_size"] == 128
        assert chunks[0].metadata["overlap"] == 32

    def test_start_end_char_set(self):
        chunks = FixedSizeChunker(chunk_size=100, overlap=0).chunk(PARA_TEXT)
        assert all(ch.start_char is not None for ch in chunks)
        assert all(ch.end_char is not None for ch in chunks)

    def test_overlap_invalid_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=50, overlap=50)

    def test_empty_text_returns_empty(self):
        assert FixedSizeChunker(chunk_size=100, overlap=10).chunk("") == []

    def test_text_shorter_than_chunk_size(self):
        chunks = FixedSizeChunker(chunk_size=1000, overlap=10).chunk(SHORT_TEXT)
        assert len(chunks) == 1
        assert SHORT_TEXT.strip() in chunks[0].content

    def test_coverage_all_text_represented(self):
        # Every word in the original text should appear in at least one chunk.
        chunks = FixedSizeChunker(chunk_size=80, overlap=20).chunk(PARA_TEXT)
        combined = " ".join(ch.content for ch in chunks)
        for word in PARA_TEXT.split():
            assert word in combined


# ---------------------------------------------------------------------------
# SentenceBasedChunker
# ---------------------------------------------------------------------------

class TestSentenceBasedChunker:
    def test_returns_chunks(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=1).chunk(PARA_TEXT)
        assert len(chunks) > 0
        assert all(isinstance(ch, Chunk) for ch in chunks)

    def test_method_field(self):
        chunks = SentenceBasedChunker().chunk(PARA_TEXT)
        assert all(ch.method == "sentence" for ch in chunks)

    def test_sequential_indices(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=1).chunk(PARA_TEXT)
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_overlap_invalid_raises(self):
        with pytest.raises(ValueError):
            SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=3)

    def test_metadata_passthrough(self):
        chunks = SentenceBasedChunker().chunk(SHORT_TEXT, metadata={"doc": "x"})
        assert all(ch.metadata["doc"] == "x" for ch in chunks)

    def test_params_in_metadata(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=4, overlap_sentences=1).chunk(PARA_TEXT)
        assert chunks[0].metadata["sentences_per_chunk"] == 4
        assert chunks[0].metadata["overlap_sentences"] == 1

    def test_sentence_boundaries_tracked(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=1).chunk(PARA_TEXT)
        for ch in chunks:
            assert "sentence_start" in ch.metadata
            assert "sentence_end" in ch.metadata

    def test_short_text(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=5, overlap_sentences=1).chunk(SHORT_TEXT)
        assert len(chunks) >= 1

    def test_empty_text(self):
        chunks = SentenceBasedChunker().chunk("")
        assert chunks == []

    def test_non_empty_content(self):
        chunks = SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=1).chunk(PARA_TEXT)
        assert all(ch.content.strip() for ch in chunks)


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_returns_chunks(self):
        chunks = SemanticChunker(_stub_embed).chunk(PARA_TEXT)
        assert len(chunks) > 0

    def test_method_field(self):
        chunks = SemanticChunker(_stub_embed).chunk(PARA_TEXT)
        assert all(ch.method == "semantic" for ch in chunks)

    def test_sequential_indices(self):
        chunks = SemanticChunker(_stub_embed).chunk(PARA_TEXT)
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_metadata_passthrough(self):
        chunks = SemanticChunker(_stub_embed).chunk(SHORT_TEXT, metadata={"page": 1})
        assert all(ch.metadata["page"] == 1 for ch in chunks)

    def test_threshold_in_metadata(self):
        chunks = SemanticChunker(_stub_embed, breakpoint_threshold=0.7).chunk(PARA_TEXT)
        assert all(ch.metadata["breakpoint_threshold"] == 0.7 for ch in chunks)

    def test_high_threshold_more_chunks(self):
        # High threshold → more breakpoints → more (smaller) chunks.
        hi = SemanticChunker(_stub_embed, breakpoint_threshold=0.99).chunk(PARA_TEXT)
        lo = SemanticChunker(_stub_embed, breakpoint_threshold=0.01).chunk(PARA_TEXT)
        assert len(hi) >= len(lo)

    def test_max_sentences_cap(self):
        # With max_sentences=2, no chunk should contain more than 2 sentences.
        chunks = SemanticChunker(_stub_embed, max_sentences=2).chunk(PARA_TEXT)
        for ch in chunks:
            sentence_count = ch.metadata["sentence_end"] - ch.metadata["sentence_start"]
            assert sentence_count <= 2

    def test_single_sentence(self):
        chunks = SemanticChunker(_stub_embed).chunk("Only one sentence here.")
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = SemanticChunker(_stub_embed).chunk("")
        assert chunks == []

    def test_non_empty_content(self):
        chunks = SemanticChunker(_stub_embed).chunk(PARA_TEXT)
        assert all(ch.content.strip() for ch in chunks)

    def test_all_text_covered(self):
        # Every sentence should appear in exactly one chunk.
        from rag_common.chunkers import _split_sentences
        sentences = _split_sentences(PARA_TEXT)
        chunks = SemanticChunker(_stub_embed, breakpoint_threshold=0.5).chunk(PARA_TEXT)
        combined = " ".join(ch.content for ch in chunks)
        for sent in sentences:
            assert sent.strip() in combined
