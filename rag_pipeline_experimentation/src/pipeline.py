"""
RAGPipeline — ingestion and query orchestrator for P4.

Ingestion flow:
    PDF paths → parse (PyMuPDF) → chunk → embed → FAISS index → save to disk

Query flow:
    question → embed → retrieve → (optional rerank) → list[RetrievalResult]

The pipeline holds one FAISS index that spans all ingested documents.
`document_id` and `source` are written into every Chunk so the evaluator
can map retrieved chunks back to their source papers for qrels evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rag_common.models import Chunk, RetrievalResult
from rag_common.parsers import parse_pdf as _parse_pdf
from rag_common.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag_common.vector_store import FAISSVectorStore

from src.base import BaseChunker, BaseEmbedder, BaseReranker
from src.models import Document


class RAGPipeline:
    """
    Wraps a chunker, embedder, vector store, and retriever into one object.

    Designed to be constructed once per experiment config and reused for
    all queries in that experiment.

    Args:
        chunker:   BaseChunker implementation (fixed, recursive, sliding, …)
        embedder:  SentenceTransformersEmbedder (or any BaseEmbedder)
        retrieval_method: "dense" | "bm25" | "hybrid"
        alpha:     hybrid fusion weight for dense scores (ignored if not hybrid)
        reranker:  optional BaseReranker applied after retrieval
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retrieval_method: str = "dense",
        alpha: float = 0.6,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.retrieval_method = retrieval_method
        self.alpha = alpha
        self.reranker = reranker

        self._store = FAISSVectorStore()
        self._all_chunks: list[Chunk] = []
        self._retriever = None   # built after ingestion
        self._documents: list[Document] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_paths: list[Path],
        index_dir: Path,
        chunk_label: str = "default",
    ) -> list[Chunk]:
        """
        Parse, chunk, embed, and index all PDFs.

        Saves the FAISS index and document metadata to `index_dir`.
        Embedding cache is keyed by `chunk_label` so configs with the same
        chunking strategy share cached embeddings.

        Returns:
            Flat list of all chunks across all documents.
        """
        index_dir.mkdir(parents=True, exist_ok=True)

        all_chunks: list[Chunk] = []
        documents: list[Document] = []

        for pdf_path in pdf_paths:
            paper_id = pdf_path.stem
            text, page_count = _parse_pdf(pdf_path)
            if not text.strip():
                continue

            doc = Document(
                id=paper_id,
                source=pdf_path.name,
                page_count=page_count,
                char_count=len(text),
            )
            documents.append(doc)

            chunks = self.chunker.chunk(
                text,
                metadata={"document_id": paper_id, "source": pdf_path.name},
            )
            # Write provenance into each Chunk's top-level fields.
            for c in chunks:
                c.document_id = paper_id
                c.source = pdf_path.name

            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks produced — check PDF paths and chunker config.")

        embeddings = self.embedder.embed_chunks(all_chunks, chunk_label)
        self._store.add(all_chunks, embeddings)
        self._store.save(str(index_dir / "faiss_index"))

        self._all_chunks = all_chunks
        self._documents = documents
        self._retriever = self._build_retriever(embeddings)

        # Persist document metadata for later loading.
        meta_path = index_dir / "documents.json"
        meta_path.write_text(
            json.dumps([d.model_dump() for d in documents], indent=2)
        )

        return all_chunks

    def load(self, index_dir: Path) -> None:
        """
        Load a previously saved index from disk.

        Call this instead of `ingest` when the index already exists.
        """
        self._store.load(str(index_dir / "faiss_index"))
        self._all_chunks = list(self._store._chunks)

        meta_path = index_dir / "documents.json"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            self._documents = [Document(**d) for d in raw]

        # Embeddings are not cached after load; DenseRetriever uses the FAISS
        # index directly, so no embeddings matrix is needed for retrieval.
        self._retriever = self._build_retriever(embeddings=None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for `question`."""
        if self._retriever is None:
            raise RuntimeError("Call ingest() or load() before query().")
        results = self._retriever.retrieve(question, top_k=top_k)
        if self.reranker is not None:
            results = self.reranker.rerank(question, results, top_k)
        return results

    def query_timed(self, question: str, top_k: int = 5) -> tuple[list[RetrievalResult], float]:
        """Returns (results, elapsed_seconds)."""
        t0 = time.perf_counter()
        results = self.query(question, top_k)
        return results, time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_retriever(self, embeddings: np.ndarray | None):
        embed_fn = lambda texts: self.embedder.embed(texts)  # noqa: E731

        if self.retrieval_method == "bm25":
            return BM25Retriever(self._all_chunks)

        dense = DenseRetriever(self._store, embed_fn)

        if self.retrieval_method == "dense":
            return dense

        return HybridRetriever(
            dense=dense,
            bm25=BM25Retriever(self._all_chunks),
            alpha=self.alpha,
        )

    @property
    def documents(self) -> list[Document]:
        return self._documents

    @property
    def chunks(self) -> list[Chunk]:
        return self._all_chunks
