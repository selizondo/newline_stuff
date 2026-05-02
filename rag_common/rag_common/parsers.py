"""
PDF text extraction shared by RAG projects.

Uses PyMuPDF (fitz) for fast, accurate text extraction with page-level
metadata. Falls back gracefully on corrupt or unreadable PDFs so one bad
file does not abort a batch ingestion.

Usage::

    from rag_common.parsers import parse_pdf

    text, page_count = parse_pdf(Path("paper.pdf"))
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(path: Path) -> tuple[str, int]:
    """
    Extract full text from a PDF using PyMuPDF.

    Args:
        path: Path to a PDF file.

    Returns:
        (text, page_count) — text is the concatenation of all pages joined
        by newlines; page_count is 0 if the file could not be parsed.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages), len(pages)
    except Exception:
        return "", 0
