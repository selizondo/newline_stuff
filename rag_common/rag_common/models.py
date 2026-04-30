"""
Shared data models used across rag_common, rag_pipeline_systematic_evals (P3),
and rag_pipeline_experimentation (P4).

Design notes
------------
- `content` is used instead of `text` so both projects share one field name.
- `embedding` is optional here; the vector store owns the numpy matrix and
  only populates this field when a chunk needs to be serialised with its vector
  (e.g. for caching). Retrieval code should read from the FAISS index, not
  from Chunk.embedding, to avoid duplicating large float arrays in memory.
- All IDs are UUIDs so chunks from different chunking runs never collide even
  when chunk_index values overlap across configurations.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    chunk_index: int
    method: str | None = None        # "fixed_size" | "sentence" | "semantic" | …
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    document_id: str | None = None   # ties chunk to a source document (P4)
    source: str | None = None        # filename / paper ID (P4)
    metadata: dict = Field(default_factory=dict)
    # Populated only when serialising chunks with their vectors for disk caching.
    # Do not use for in-memory retrieval — read from FAISSVectorStore instead.
    embedding: list[float] | None = None

    def id_str(self) -> str:
        return str(self.id)


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    # "dense" | "bm25" | "hybrid" — preserved so callers can filter or log by
    # retriever type without needing to re-run retrieval.
    retriever_type: str
