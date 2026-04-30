"""
Tests for qa_generator.py.

All LLM calls are intercepted — no API key or network required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_common.models import Chunk
from src.qa_generator import (
    QADataset, QAPair, QAPairResponse,
    _generate_one, _load_dataset, _sample_chunks, _save_dataset,
    generate_qa_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(content=f"The mitochondria produces ATP. Fact number {i}.", chunk_index=i, method="fixed_size")
        for i in range(n)
    ]


def _fake_instructor_client(question: str = "What does mitochondria produce?", question_type: str = "factual"):
    """Returns a mock instructor client whose .chat.completions.create returns a fixed QAPairResponse."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = QAPairResponse(
        question=question,
        question_type=question_type,
    )
    return mock_client


# ---------------------------------------------------------------------------
# QAPairResponse validation
# ---------------------------------------------------------------------------

class TestQAPairResponse:
    def test_valid_response(self):
        r = QAPairResponse(question="What does ATP stand for?", question_type="factual")
        assert r.question == "What does ATP stand for?"

    def test_empty_question_raises(self):
        with pytest.raises(Exception):
            QAPairResponse(question="   ", question_type="factual")

    def test_strips_whitespace(self):
        r = QAPairResponse(question="  What is ATP?  ", question_type="factual")
        assert r.question == "What is ATP?"

    def test_invalid_question_type_raises(self):
        with pytest.raises(Exception):
            QAPairResponse(question="What?", question_type="unknown_type")


# ---------------------------------------------------------------------------
# _sample_chunks
# ---------------------------------------------------------------------------

class TestSampleChunks:
    def test_exact_count_no_replacement(self):
        chunks = _make_chunks(10)
        sample = _sample_chunks(chunks, 5, seed=42)
        assert len(sample) == 5
        assert len(set(c.id_str() for c in sample)) == 5   # no duplicates

    def test_all_chunks_when_n_equals_len(self):
        chunks = _make_chunks(5)
        sample = _sample_chunks(chunks, 5, seed=0)
        assert len(sample) == 5

    def test_with_replacement_when_n_exceeds_len(self):
        chunks = _make_chunks(3)
        sample = _sample_chunks(chunks, 7, seed=0)
        assert len(sample) == 7

    def test_reproducible_with_same_seed(self):
        chunks = _make_chunks(10)
        a = [c.id_str() for c in _sample_chunks(chunks, 5, seed=1)]
        b = [c.id_str() for c in _sample_chunks(chunks, 5, seed=1)]
        assert a == b

    def test_different_seeds_give_different_samples(self):
        chunks = _make_chunks(10)
        a = [c.id_str() for c in _sample_chunks(chunks, 5, seed=1)]
        b = [c.id_str() for c in _sample_chunks(chunks, 5, seed=2)]
        assert a != b


# ---------------------------------------------------------------------------
# _generate_one
# ---------------------------------------------------------------------------

class TestGenerateOne:
    def test_returns_qa_pair(self):
        client = _fake_instructor_client()
        chunk = _make_chunks(1)[0]
        pair = _generate_one(client, chunk, model="gpt-4o-mini", attempt=0)
        assert isinstance(pair, QAPair)
        assert pair.question == "What does mitochondria produce?"
        assert chunk.id_str() in pair.relevant_chunk_ids

    def test_returns_none_on_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        chunk = _make_chunks(1)[0]
        pair = _generate_one(client, chunk, model="gpt-4o-mini", attempt=0)
        assert pair is None

    def test_metadata_populated(self):
        client = _fake_instructor_client()
        chunk = _make_chunks(1)[0]
        pair = _generate_one(client, chunk, model="gpt-4o-mini", attempt=0)
        assert pair.metadata["synthetic"] is True
        assert pair.metadata["chunk_method"] == "fixed_size"
        assert "source_chunk_index" in pair.metadata


# ---------------------------------------------------------------------------
# _save_dataset / _load_dataset
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        chunks = _make_chunks(2)
        dataset = QADataset(
            chunk_config_label="fixed_256_ol50",
            pairs=[
                QAPair(question="Q1?", question_type="factual", relevant_chunk_ids=[chunks[0].id_str()]),
                QAPair(question="Q2?", question_type="conceptual", relevant_chunk_ids=[chunks[1].id_str()]),
            ],
        )
        path = tmp_path / "fixed_256_ol50.json"
        _save_dataset(dataset, path)
        loaded = _load_dataset(path)

        assert loaded.chunk_config_label == "fixed_256_ol50"
        assert loaded.size == 2
        assert loaded.pairs[0].question == "Q1?"

    def test_save_creates_parent_dirs(self, tmp_path):
        dataset = QADataset(chunk_config_label="test", pairs=[])
        path = tmp_path / "nested" / "dir" / "test.json"
        _save_dataset(dataset, path)
        assert path.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        dataset = QADataset(chunk_config_label="x", pairs=[])
        path = tmp_path / "x.json"
        _save_dataset(dataset, path)
        data = json.loads(path.read_text())
        assert data["chunk_config_label"] == "x"


# ---------------------------------------------------------------------------
# generate_qa_dataset
# ---------------------------------------------------------------------------

class TestGenerateQADataset:
    def _patched_generate(self, monkeypatch, chunks, n_pairs, tmp_path):
        """Run generate_qa_dataset with a mocked LLM."""
        call_count = [0]

        def fake_generate_one(client, chunk, model, attempt):
            call_count[0] += 1
            return QAPair(
                question=f"Question about chunk {chunk.chunk_index}?",
                question_type="factual",
                relevant_chunk_ids=[chunk.id_str()],
                metadata={"synthetic": True, "chunk_method": chunk.method,
                          "source_chunk_index": chunk.chunk_index, "source_page": None},
            )

        monkeypatch.setattr("src.qa_generator._generate_one", fake_generate_one)
        monkeypatch.setattr("src.qa_generator.time.sleep", lambda _: None)
        monkeypatch.setattr("src.qa_generator.instructor.from_openai", lambda c: MagicMock())

        with patch("src.qa_generator.OpenAI"):
            dataset = generate_qa_dataset(chunks, "fixed_256_ol50", n_pairs=n_pairs, qa_dir=tmp_path)
        return dataset, call_count[0]

    def test_returns_qa_dataset(self, monkeypatch, tmp_path):
        chunks = _make_chunks(10)
        dataset, _ = self._patched_generate(monkeypatch, chunks, n_pairs=5, tmp_path=tmp_path)
        assert isinstance(dataset, QADataset)

    def test_correct_number_of_pairs(self, monkeypatch, tmp_path):
        chunks = _make_chunks(10)
        dataset, _ = self._patched_generate(monkeypatch, chunks, n_pairs=7, tmp_path=tmp_path)
        assert dataset.size == 7

    def test_reads_from_cache_on_second_call(self, monkeypatch, tmp_path):
        chunks = _make_chunks(5)
        dataset1, count1 = self._patched_generate(monkeypatch, chunks, n_pairs=3, tmp_path=tmp_path)
        # Second call — cache file now exists, should not invoke _generate_one.
        dataset2, count2 = self._patched_generate(monkeypatch, chunks, n_pairs=3, tmp_path=tmp_path)
        assert count2 == 0   # no LLM calls on cache hit
        assert dataset2.size == dataset1.size

    def test_chunk_ids_valid(self, monkeypatch, tmp_path):
        chunks = _make_chunks(5)
        dataset, _ = self._patched_generate(monkeypatch, chunks, n_pairs=5, tmp_path=tmp_path)
        valid_ids = {c.id_str() for c in chunks}
        for pair in dataset.pairs:
            for cid in pair.relevant_chunk_ids:
                assert cid in valid_ids

    def test_empty_chunks_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            generate_qa_dataset([], "label", qa_dir=tmp_path)

    def test_label_in_dataset(self, monkeypatch, tmp_path):
        chunks = _make_chunks(3)
        dataset, _ = self._patched_generate(monkeypatch, chunks, n_pairs=2, tmp_path=tmp_path)
        assert dataset.chunk_config_label == "fixed_256_ol50"
