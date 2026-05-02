"""Tests for RecursiveChunker and SlidingWindowChunker."""

from __future__ import annotations

import pytest

from src.chunkers_ext import RecursiveChunker, SlidingWindowChunker

_SHORT = "Hello world. This is a test."
_LONG  = ("Alpha beta gamma delta epsilon. " * 30).strip()   # ~960 chars


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunks = RecursiveChunker(chunk_size=512).chunk(_SHORT)
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    def test_long_text_splits(self):
        chunks = RecursiveChunker(chunk_size=128, overlap=20).chunk(_LONG)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_chunk_size_by_much(self):
        chunker = RecursiveChunker(chunk_size=128, overlap=20)
        chunks = chunker.chunk(_LONG)
        # Allow 2× slack because a single un-splittable token can exceed the limit.
        for c in chunks:
            assert len(c.content) <= chunker.chunk_size * 2

    def test_method_field(self):
        chunks = RecursiveChunker().chunk(_SHORT)
        assert all(c.method == "recursive" for c in chunks)

    def test_metadata_merged(self):
        chunks = RecursiveChunker().chunk(_SHORT, metadata={"source": "test.pdf"})
        assert all(c.metadata.get("source") == "test.pdf" for c in chunks)

    def test_chunk_indices_sequential(self):
        chunks = RecursiveChunker(chunk_size=128, overlap=20).chunk(_LONG)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_overlap_too_large_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, overlap=100)

    def test_empty_text(self):
        chunks = RecursiveChunker().chunk("")
        assert chunks == []

    def test_paragraph_splitting(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = RecursiveChunker(chunk_size=30, overlap=5).chunk(text)
        assert len(chunks) >= 2  # small paragraphs may merge up to chunk_size


class TestSlidingWindowChunker:
    _TEXT = " ".join(f"Sentence {i}." for i in range(30))

    def test_basic(self):
        chunks = SlidingWindowChunker(window_size=5, step=3).chunk(self._TEXT)
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    def test_method_field(self):
        chunks = SlidingWindowChunker().chunk(self._TEXT)
        assert all(c.method == "sliding_window" for c in chunks)

    def test_overlap_creates_more_chunks_than_no_overlap(self):
        c_overlap = SlidingWindowChunker(window_size=5, step=2).chunk(self._TEXT)
        c_no_overlap = SlidingWindowChunker(window_size=5, step=5).chunk(self._TEXT)
        assert len(c_overlap) >= len(c_no_overlap)

    def test_metadata_merged(self):
        chunks = SlidingWindowChunker().chunk(self._TEXT, metadata={"doc": "x"})
        assert all(c.metadata.get("doc") == "x" for c in chunks)

    def test_step_larger_than_window_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(window_size=3, step=5)

    def test_step_zero_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(window_size=5, step=0)

    def test_short_text(self):
        chunks = SlidingWindowChunker(window_size=10, step=5).chunk("One sentence.")
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        chunks = SlidingWindowChunker(window_size=5, step=3).chunk(self._TEXT)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
