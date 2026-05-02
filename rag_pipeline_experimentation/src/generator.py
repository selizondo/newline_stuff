"""LLM answer generator with indexed citation extraction."""

from __future__ import annotations

import re
import time

from dotenv import load_dotenv

from rag_common.models import Chunk, RetrievalResult
from src.models import Citation, QAResponse

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about scientific papers. "
    "Answer using only the provided context chunks, numbered [1], [2], etc. "
    "Cite sources inline with [N] notation. "
    "If the context does not contain the answer, say so — do not hallucinate."
)


def _format_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.source or chunk.document_id or "unknown"
        page = f", p.{chunk.page_number}" if chunk.page_number else ""
        parts.append(f"[{i}] (Source: {src}{page})\n{chunk.content}")
    return "\n\n".join(parts)


def _extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            chunk = chunks[idx]
            citations.append(Citation(
                chunk_id=chunk.id_str(),
                source=chunk.source or chunk.document_id or "unknown",
                page_number=chunk.page_number,
                text_snippet=chunk.content[:200],
            ))
    return citations


def generate_answer(
    query: str,
    retrieval_results: list[RetrievalResult],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> QAResponse:
    """Generate an answer for `query` given retrieved chunks.

    Uses indexed [N] references so citations can be extracted by regex.

    Args:
        query:             User question.
        retrieval_results: Retrieved chunks, best first.
        model:             LLM model identifier.
        temperature:       Generation temperature.
        max_tokens:        Max answer tokens.

    Returns:
        QAResponse with answer text, parsed citations, and timing.
    """
    from llm_utils.client import chat_complete

    chunks = [r.chunk for r in retrieval_results]
    context = _format_context(chunks)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using the context above. "
        "Cite your sources using [N] notation inline."
    )

    t0 = time.perf_counter()
    answer = chat_complete(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    gen_time = time.perf_counter() - t0

    citations = _extract_citations(answer, chunks)

    return QAResponse(
        query=query,
        answer=answer,
        citations=citations,
        chunks_used=[c.model_dump(exclude={"embedding"}) for c in chunks],
        generation_time_s=round(gen_time, 4),
    )
