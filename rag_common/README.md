# rag_common

Shared library for the RAG pipeline mini-projects.

Both downstream projects depend on this package as an editable path install:

- **rag_pipeline_systematic_evals** (P3) — single-PDF grid search with synthetic QA evaluation
- **rag_pipeline_experimentation** (P4) — multi-paper QA assistant with real ground truth (qrels.json), citations, and a Streamlit UI

Centralising these modules means IR metric bugs are fixed once, chunk IDs are always UUID-based, and score normalisation logic never drifts between projects.

---

## Modules

### `rag_common.models`

Core Pydantic data types shared across both projects.

| Class | Purpose |
|---|---|
| `Chunk` | A piece of text extracted from a PDF, with provenance metadata |
| `RetrievalResult` | A retrieved chunk paired with its score and retriever type |

**Key design decisions:**
- Field is named `content` (not `text`) so both projects use the same field name.
- `Chunk.id` is a UUID so chunk IDs from different chunking configs never collide even when `chunk_index` values overlap.
- `Chunk.embedding` is `None` by default. Only populated when serialising chunks to disk for caching; in-memory retrieval reads from the FAISS index to avoid duplicating large float arrays.

```python
from rag_common.models import Chunk, RetrievalResult

chunk = Chunk(content="Neural networks learn via backprop.", chunk_index=0, method="fixed_size")
print(chunk.id_str())   # "3f2a1b…"
```

---

### `rag_common.metrics`

Standard IR evaluation metrics operating on chunk ID strings, independent of any embedding library.

| Function | Description |
|---|---|
| `recall_at_k(retrieved, relevant, k)` | Fraction of relevant chunks in top-k |
| `precision_at_k(retrieved, relevant, k)` | Fraction of top-k that are relevant |
| `reciprocal_rank(retrieved, relevant)` | 1 / rank of first relevant result |
| `average_precision(retrieved, relevant)` | Area under precision-recall curve |
| `ndcg_at_k(retrieved, relevant, k)` | Normalised Discounted Cumulative Gain |
| `mrr(query_results)` | Mean Reciprocal Rank across all queries |
| `map_score(query_results)` | Mean Average Precision across all queries |
| `evaluate(query_results, k)` | All metrics in one dict |

**Precision@K note:** When each query has exactly one ground-truth chunk (P3 synthetic QA), Precision@K is capped at `1/K` — e.g. max 0.20 at K=5. Expected behaviour, not a bug. Use MRR and Recall@K as primary signals.

**NDCG formula:** Standard TREC/BEIR convention — `1 / log2(rank + 1)` — so rank-1 gets full credit and the denominator is never zero.

```python
from rag_common import metrics

query_results = [
    (["chunk-a", "chunk-b", "chunk-c"], {"chunk-a"}),
    (["chunk-x", "chunk-a", "chunk-b"], {"chunk-a"}),
]
scores = metrics.evaluate(query_results, k=5)
# {"recall@5": 1.0, "precision@5": 0.2, "mrr": 0.75, "map": 0.75, "ndcg@5": 1.0}
```

---

### `rag_common.chunkers`

Three text splitting strategies, all exposing `.chunk(text, metadata={}) -> list[Chunk]`.

| Class | Strategy | Key params |
|---|---|---|
| `FixedSizeChunker` | Character window, word-boundary aware | `chunk_size`, `overlap` |
| `SentenceBasedChunker` | Groups N sentences with sentence-level overlap (NLTK) | `sentences_per_chunk`, `overlap_sentences` |
| `SemanticChunker` | Splits at cosine-similarity drops between adjacent sentences | `embed_fn`, `breakpoint_threshold`, `max_sentences` |

Project-specific chunkers (`RecursiveChunker`, `SlidingWindowChunker`) live in each project's `chunkers_ext.py` and follow the same interface.

- **`FixedSizeChunker`** walks back to the nearest space before cutting — no mid-word splits. Actual chunk length may be up to one word shorter than `chunk_size`.
- **`SentenceBasedChunker`** uses NLTK `punkt`; downloads corpus automatically on first use; falls back to a regex splitter if NLTK is unavailable.
- **`SemanticChunker`** accepts any `embed_fn: Callable[[list[str]], np.ndarray]` — OpenAI, SentenceTransformers, or a test stub. `max_sentences` caps chunk size even when similarity never drops below the threshold.

```python
from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker, SemanticChunker

chunks = FixedSizeChunker(chunk_size=512, overlap=64).chunk(text, metadata={"source": "paper.pdf"})
chunks = SentenceBasedChunker(sentences_per_chunk=5, overlap_sentences=1).chunk(text)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = SemanticChunker(embed_fn=model.encode, breakpoint_threshold=0.65).chunk(text)
```

---

### `rag_common.vector_store`

Protocol-based adapter — swap vector backends in one line.

| Name | Role |
|---|---|
| `VectorStoreProtocol` | Structural Protocol (`@runtime_checkable`) — type-hint against this |
| `FAISSVectorStore` | Production; IndexFlatIP + L2-normalised embeddings (cosine via inner product) |
| `InMemoryVectorStore` | Tests / prototypes; brute-force cosine, no FAISS dependency |

Any class implementing `add`, `search`, `save`, `load`, and `__len__` satisfies the Protocol — no inheritance required. Adding a Qdrant or Pinecone backend is a new class, zero changes to retrieval code.

**Why IndexFlatIP?** After L2-normalisation, inner product equals cosine similarity and scores land in `[-1, 1]`, matching the scale `HybridRetriever` expects on the dense side.

```python
from rag_common.vector_store import FAISSVectorStore, InMemoryVectorStore, VectorStoreProtocol

store: VectorStoreProtocol = FAISSVectorStore()       # prod
store: VectorStoreProtocol = InMemoryVectorStore()    # tests
store: VectorStoreProtocol = QdrantVectorStore(...)   # future — just implement the protocol

store.add(chunks, embeddings)   # embeddings: np.ndarray (N, D)
results = store.search(query_emb, top_k=5)
store.save("data/indices/my_index")
store.load("data/indices/my_index")
```

---

### `rag_common.retrievers`

All retrievers satisfy `RetrieverProtocol`:

```python
results: list[RetrievalResult] = retriever.retrieve(query: str, top_k: int)
```

| Class | Strategy | Key params |
|---|---|---|
| `BM25Retriever` | Sparse keyword matching (rank-bm25) | `chunks` |
| `DenseRetriever` | Embedding similarity via any `VectorStoreProtocol` | `store`, `embed_fn` |
| `HybridRetriever` | Min-max normalised score fusion of dense + BM25 | `dense`, `bm25`, `alpha`, `fetch_k` |

**Score fusion:** BM25 scores are unbounded positive floats; dense scores are cosine similarities in `[-1, 1]`. Fusing without normalisation lets BM25 dominate. `HybridRetriever` fetches `fetch_k = max(top_k * 3, 20)` candidates from each, min-max normalises both to `[0, 1]` independently, then fuses: `score = alpha * dense_norm + (1 - alpha) * bm25_norm`. Fused scores guaranteed in `[0, 1]`.

```python
from rag_common.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag_common.vector_store import FAISSVectorStore

store = FAISSVectorStore()
store.add(chunks, embeddings)

bm25   = BM25Retriever(chunks)
dense  = DenseRetriever(store, embed_fn=model.encode)
hybrid = HybridRetriever(dense, bm25, alpha=0.6)   # 60% dense, 40% BM25

results = hybrid.retrieve("what is backpropagation?", top_k=5)
```

---

## Installation

```bash
pip install -e ../rag_common
```

Or via uv `pyproject.toml`:

```toml
[tool.uv.sources]
rag-common = { path = "../rag_common", editable = true }

[project]
dependencies = ["rag-common", ...]
```

---

## Running tests

```bash
cd rag_common
pip install -e ".[dev]"
pytest tests/ -v
```

143 tests: metrics (hand-verified math), chunkers (deterministic stub embedder), vector store (parametrised against both FAISS and InMemory), retrievers (Protocol checks + alpha boundary tests).

---

## Status

`rag_common` is complete. Project-specific logic lives in each downstream project:

| Project | Adds |
|---|---|
| `rag_pipeline_systematic_evals` (P3) | OpenAI embedder, pdfplumber parser, synthetic QA generator, grid search, visualiser |
| `rag_pipeline_experimentation` (P4) | SentenceTransformers embedder, PyMuPDF parser, recursive/sliding-window chunkers, LLM generator + citations, qrels evaluator, Streamlit UI |
