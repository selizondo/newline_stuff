"""
PDF parsing for P3.

Extracts raw text from a PDF using pdfplumber and returns it as a single
string with page metadata preserved in the output structure.

Only pdfplumber is used here (the spec says "use a consistent PDF parser
across experiments" — switching parser between runs changes the extracted
text and invalidates comparisons). The abstraction is thin enough that
swapping to PyMuPDF or PyPDF2 later requires changing only this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class ParsedPage:
    page_number: int       # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ParsedDocument:
    source: str            # original file path
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        """Concatenated text across all pages, separated by newlines."""
        return "\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def parse_pdf(path: str | Path) -> ParsedDocument:
    """
    Extract text from a PDF using pdfplumber.

    Each page is extracted individually so page-level metadata (page_number)
    is preserved and can be stored in Chunk.page_number during chunking.

    Args:
        path: path to the PDF file.

    Returns:
        ParsedDocument with per-page text and aggregate properties.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if the PDF has no extractable text (e.g. scanned image PDF).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[ParsedPage] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # pdfplumber occasionally returns None for image-only pages.
            pages.append(ParsedPage(page_number=i, text=_clean(text)))

    doc = ParsedDocument(source=str(path), pages=pages)

    if not doc.full_text.strip():
        raise ValueError(
            f"No extractable text found in {path}. "
            "The PDF may be a scanned image — try OCR preprocessing first."
        )

    return doc


def _clean(text: str) -> str:
    """
    Light normalisation of raw PDF text.

    pdfplumber sometimes produces runs of spaces or ligature characters
    that hurt chunking quality. Strip those without altering content.
    """
    import re
    # Collapse multiple spaces / non-breaking spaces to a single space.
    text = re.sub(r"[ \t\xa0]+", " ", text)
    # Normalise line endings.
    text = re.sub(r"\r\n?", "\n", text)
    # Drop lines that are just whitespace or page numbers (lone integers).
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().isdigit()]
    return "\n".join(lines)
