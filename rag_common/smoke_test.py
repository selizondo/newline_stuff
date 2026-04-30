"""
Smoke test for rag_common.

Runs the full pipeline end-to-end using only stdlib + installed deps:
  text → chunk → embed → store → retrieve (BM25 / dense / hybrid) → metrics

No API calls, no mocking. Uses numpy random vectors as the embedder so
the test is fast and offline. Fails loudly if any module is broken.
"""

from __future__ import annotations

import tempfile
import textwrap

import numpy as np

from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker, SemanticChunker
from rag_common.metrics import evaluate
from rag_common.models import Chunk
from rag_common.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from rag_common.vector_store import FAISSVectorStore, InMemoryVectorStore, VectorStoreProtocol

# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------

TEXT = textwrap.dedent("""\
    The mitochondria is the powerhouse of the cell.
    It produces ATP through oxidative phosphorylation in the inner membrane.
    This process requires oxygen and releases carbon dioxide as a byproduct.
    Photosynthesis is the reverse: plants convert CO2 and water into glucose.
    Chloroplasts contain chlorophyll that absorbs sunlight for this reaction.
    Neural networks learn by adjusting weights through backpropagation.
    Gradient descent minimises the loss function iteratively over many epochs.
    DNA replication is semi-conservative: each strand serves as a template.
    Primase synthesises a short RNA primer before DNA polymerase continues.
    Quantum entanglement links particle states regardless of distance.
""")

DIM = 32
RNG = np.random.default_rng(42)


def _embed(texts: list[str]) -> np.ndarray:
    """Deterministic stub: hash-seeded random unit vectors."""
    vecs = []
    for t in texts:
        seed = abs(hash(t[:30])) % (2**31)
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "✓" if condition else "✗ FAIL"
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"SMOKE TEST FAILED: {label}")


# ---------------------------------------------------------------------------
# 1. Chunkers
# ---------------------------------------------------------------------------

_banner("1. Chunkers")

fixed_chunks = FixedSizeChunker(chunk_size=200, overlap=40).chunk(TEXT, metadata={"source": "sample.txt"})
_check("FixedSizeChunker produces chunks", len(fixed_chunks) > 0, f"{len(fixed_chunks)} chunks")
_check("FixedSize method field", all(c.method == "fixed_size" for c in fixed_chunks))
_check("FixedSize metadata passthrough", fixed_chunks[0].metadata["source"] == "sample.txt")
_check("FixedSize no empty content", all(c.content.strip() for c in fixed_chunks))

sent_chunks = SentenceBasedChunker(sentences_per_chunk=3, overlap_sentences=1).chunk(TEXT)
_check("SentenceBasedChunker produces chunks", len(sent_chunks) > 0, f"{len(sent_chunks)} chunks")
_check("Sentence method field", all(c.method == "sentence" for c in sent_chunks))

sem_chunks = SemanticChunker(_embed, breakpoint_threshold=0.5, max_sentences=4).chunk(TEXT)
_check("SemanticChunker produces chunks", len(sem_chunks) > 0, f"{len(sem_chunks)} chunks")
_check("Semantic method field", all(c.method == "semantic" for c in sem_chunks))

# ---------------------------------------------------------------------------
# 2. Vector store — FAISS round-trip
# ---------------------------------------------------------------------------

_banner("2. Vector store (FAISS)")

chunks = fixed_chunks
embeddings = _embed([c.content for c in chunks])

faiss_store: VectorStoreProtocol = FAISSVectorStore()
faiss_store.add(chunks, embeddings)
_check("FAISS add + len", len(faiss_store) == len(chunks), f"{len(faiss_store)} indexed")

results = faiss_store.search(embeddings[0], top_k=3)
_check("FAISS search returns results", len(results) == 3)
_check("FAISS top result is exact match", results[0].chunk.content == chunks[0].content)
_check("FAISS scores descending", results[0].score >= results[1].score >= results[2].score)

with tempfile.TemporaryDirectory() as tmp:
    faiss_store.save(f"{tmp}/idx")
    loaded = FAISSVectorStore()
    loaded.load(f"{tmp}/idx")
    _check("FAISS save/load round-trip len", len(loaded) == len(chunks))
    rt_results = loaded.search(embeddings[0], top_k=1)
    _check("FAISS save/load top result intact", rt_results[0].chunk.content == chunks[0].content)

# ---------------------------------------------------------------------------
# 3. Vector store — InMemory agrees with FAISS
# ---------------------------------------------------------------------------

_banner("3. Vector store (InMemory vs FAISS)")

mem_store: VectorStoreProtocol = InMemoryVectorStore()
mem_store.add(chunks, embeddings.astype(np.float64))

faiss_top = faiss_store.search(embeddings[2], top_k=1)[0].chunk.content
mem_top   = mem_store.search(embeddings[2], top_k=1)[0].chunk.content
_check("InMemory top-1 agrees with FAISS", faiss_top == mem_top, f"both: '{faiss_top[:40]}…'")

# ---------------------------------------------------------------------------
# 4. Retrievers
# ---------------------------------------------------------------------------

_banner("4. Retrievers")

bm25  = BM25Retriever(chunks)
dense = DenseRetriever(faiss_store, _embed)
hybrid = HybridRetriever(dense, bm25, alpha=0.6)

bm25_results = bm25.retrieve("mitochondria ATP oxidative", top_k=3)
_check("BM25 returns results", len(bm25_results) > 0)
_check("BM25 retriever_type", all(r.retriever_type == "bm25" for r in bm25_results))
_check("BM25 relevant chunk ranks first", "mitochondria" in bm25_results[0].chunk.content.lower())

dense_results = dense.retrieve("photosynthesis chloroplasts sunlight", top_k=3)
_check("Dense returns results", len(dense_results) > 0)
_check("Dense retriever_type", all(r.retriever_type == "dense" for r in dense_results))
_check("Dense scores descending", dense_results[0].score >= dense_results[-1].score)

hybrid_results = hybrid.retrieve("neural network backpropagation gradient", top_k=3)
_check("Hybrid returns results", len(hybrid_results) > 0)
_check("Hybrid retriever_type", all(r.retriever_type == "hybrid" for r in hybrid_results))
_check("Hybrid scores in [0, 1]", all(0.0 <= r.score <= 1.0 + 1e-9 for r in hybrid_results))
_check("Hybrid scores descending", hybrid_results[0].score >= hybrid_results[-1].score)

# ---------------------------------------------------------------------------
# 5. Metrics — end-to-end
# ---------------------------------------------------------------------------

_banner("5. Metrics (end-to-end evaluation)")

# Build ground truth: for each chunk, it is its own correct answer.
# Query each chunk's content, check the chunk itself ranks in top-5.
query_results = []
for i, chunk in enumerate(chunks):
    retrieved = [r.chunk.id_str() for r in dense.retrieve(chunk.content, top_k=5)]
    relevant  = {chunk.id_str()}
    query_results.append((retrieved, relevant))

scores = evaluate(query_results, k=5)
_check("recall@5 > 0", scores["recall@5"] > 0, f"{scores['recall@5']:.3f}")
_check("mrr > 0", scores["mrr"] > 0, f"{scores['mrr']:.3f}")
_check("ndcg@5 > 0", scores["ndcg@5"] > 0, f"{scores['ndcg@5']:.3f}")
_check("evaluate() returns all keys",
       {"recall@5", "precision@5", "mrr", "map", "ndcg@5"} == set(scores))

print(f"\n  Scores: recall@5={scores['recall@5']:.3f}  mrr={scores['mrr']:.3f}"
      f"  ndcg@5={scores['ndcg@5']:.3f}  map={scores['map']:.3f}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_banner("SMOKE TEST PASSED")
print("  All rag_common modules wired correctly.\n")
