"""Tests for parsers.py — use a real tiny PDF written in-memory via fpdf2."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.parsers import ParsedDocument, ParsedPage, parse_pdf, _clean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(text_per_page: list[str]) -> Path:
    """Write a minimal multi-page PDF using fpdf2 and return its path."""
    try:
        from fpdf import FPDF
    except ImportError:
        pytest.skip("fpdf2 not installed — skipping PDF round-trip tests")

    pdf = FPDF()
    for text in text_per_page:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# _clean
# ---------------------------------------------------------------------------

class TestClean:
    def test_collapses_spaces(self):
        assert "a b" in _clean("a   b")

    def test_strips_lone_digits(self):
        result = _clean("page text\n42\nmore text")
        assert "42" not in result.split("\n")

    def test_normalises_crlf(self):
        assert "\r" not in _clean("line1\r\nline2")

    def test_preserves_content(self):
        text = "The mitochondria is the powerhouse of the cell."
        assert text in _clean(text)


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------

class TestParsedDocument:
    def _doc(self) -> ParsedDocument:
        pages = [
            ParsedPage(page_number=1, text="Hello world."),
            ParsedPage(page_number=2, text="Second page."),
        ]
        return ParsedDocument(source="test.pdf", pages=pages)

    def test_full_text_joins_pages(self):
        doc = self._doc()
        assert "Hello world." in doc.full_text
        assert "Second page." in doc.full_text

    def test_total_chars(self):
        doc = self._doc()
        assert doc.total_chars == len("Hello world.") + len("Second page.")

    def test_page_count(self):
        assert self._doc().page_count == 2

    def test_empty_pages_excluded_from_full_text(self):
        pages = [ParsedPage(page_number=1, text="Real content."),
                 ParsedPage(page_number=2, text="   ")]
        doc = ParsedDocument(source="x.pdf", pages=pages)
        assert "Real content." in doc.full_text
        assert doc.full_text.count("\n") == 0  # only one real page


# ---------------------------------------------------------------------------
# parse_pdf
# ---------------------------------------------------------------------------

class TestParsePdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_returns_parsed_document(self):
        path = _make_pdf(["The mitochondria is the powerhouse of the cell."])
        doc = parse_pdf(path)
        assert isinstance(doc, ParsedDocument)
        assert doc.page_count >= 1

    def test_source_field_set(self):
        path = _make_pdf(["Some text here."])
        doc = parse_pdf(path)
        assert str(path) in doc.source

    def test_text_extracted(self):
        path = _make_pdf(["Neural networks learn via backpropagation."])
        doc = parse_pdf(path)
        assert "Neural networks" in doc.full_text

    def test_multipage_pdf(self):
        path = _make_pdf(["Page one content.", "Page two content."])
        doc = parse_pdf(path)
        assert doc.page_count == 2
        assert "Page one" in doc.full_text
        assert "Page two" in doc.full_text

    def test_page_numbers_one_indexed(self):
        path = _make_pdf(["First.", "Second."])
        doc = parse_pdf(path)
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2
