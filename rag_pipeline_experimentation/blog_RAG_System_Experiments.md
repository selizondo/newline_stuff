# Building a RAG System That Can Actually Experiment: What We Learned from 12 Configurations and 1,000 arXiv Papers

Most RAG systems get built once. You pick a chunk size, grab an embedding model, wire up FAISS, and ship it. That's fine — until someone asks why the assistant missed a question it should have answered, and you realize you have no way to answer that question because you never built a way to measure it.

This post documents the design of a RAG pipeline built for *experimentation first*: a system where you can swap chunking strategies, embedding models, and retrieval methods via config file and get a comparable evaluation back in under 30 minutes. We ran it on the Open RAG Benchmark — 1,000 real arXiv papers with 3,045 human-authored questions — and the first 12-cell grid on a 5-paper corpus exposed something worth knowing about how POC results can mislead you.

---

## The Problem: Researchers Drowning in Papers

Picture a PhD student surveying a subfield before writing a literature review. They have 200 papers to cover. They know the answer to their question is in *one of them* — but reading 200 papers to find it isn't research, it's archaeology.

A RAG system built for this should answer: *"Which paper introduces the attention sink token technique for KV-cache compression?"* in under a second, with a citation. That's tractable. What isn't tractable — without measurement — is knowing whether the system will also answer the subtler version: *"What methods address the memory bottleneck in long-context inference?"*, which requires semantic understanding, not keyword matching.

The configuration choices — chunk strategy, embedding model, retrieval method — determine whether the system handles the second question or only the first. Most teams pick one configuration and never find out.

---

## What Is a Configurable RAG Pipeline?

A standard RAG pipeline looks like this:

```
PDF
 └─ Chunking   (strategy + size + overlap)
     └─ Embedding  (model + dimensions)
         └─ Vector Store → Retrieval  (method + top-k)
             └─ LLM  (generates answer from retrieved chunks)
```

Each stage is a hyperparameter. And the interactions between stages matter more than any individual choice:

- A chunking strategy that destroys sentence boundaries makes BM25 worse (no complete phrases to match) and vector retrieval worse (semantic meaning is split mid-sentence).
- A smaller embedding model might catch up to a larger one if the chunking strategy produces cleaner, more self-contained chunks.
- Hybrid retrieval (dense + BM25) only beats pure vector search if the query distribution has enough keyword-specific questions to benefit from the lexical signal.

You can reason about these interactions all day. Or you can run a grid and measure them.

---

## The Dataset: Open RAG Benchmark

The [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) is 1,000 arXiv research papers with 3,045 question-answer pairs and ground-truth relevance labels, assembled by Vectara. It's one of the few publicly available RAG evaluation datasets built from real documents with real questions — not synthetic QA pairs generated from the documents themselves.

**Why this matters:** most RAG benchmarks generate synthetic questions *from* the chunks they're evaluating against. The question is worded to match the chunk. That inflates BM25 scores artificially and makes the evaluation measure "can you find the chunk this question was written from?" instead of "can you answer a question a real user would ask?"

Open RAG Benchmark questions were written by humans reading the papers. They're paraphrases. The vocabulary in the question often doesn't appear in the relevant chunk. That's a harder, more realistic evaluation.

### The qrels Format Surprise

The benchmark uses a non-standard relevance format. Standard BEIR format maps query IDs to document scores:

```json
{ "query_id": { "doc_id": 1 } }
```

Open RAG Benchmark format maps query IDs to a dict with named fields:

```json
{ "query_id": { "doc_id": "2404.18884v2", "section_id": 3 } }
```

This broke the standard BEIR loading code silently — the qrels appeared to load but no queries matched any documents, scoring everything MRR = 0.0 with no error. The fix required reading `rel.get("doc_id")` instead of iterating over the dict as document-score pairs. **Silent zero scores are the worst kind of bug** — the pipeline runs cleanly, you just never know the evaluation was broken.

### Only 13% of Downloaded Papers Have qrels

The benchmark covers 1,000 papers. Downloading 41 of them at random means most won't have any matching questions — the qrels exist for the full 1,000, not a random 41. On the initial download, only 13 of 41 papers (32%) had matching queries after filtering.

The solution: filter qrels at download time to only include queries whose relevant paper was actually downloaded. This produces `qrels_filtered.json` — a smaller but correct evaluation set — rather than a full qrels file full of queries you can never score.

---

## The Architecture: Built for Swapping

### Abstract Base Classes for Every Stage

Every swappable component implements an abstract base class:

```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]: ...

class BaseEmbedder(ABC):
    @property @abstractmethod
    def model_name(self) -> str: ...

    @property @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]: ...
```

The `RAGPipeline` takes these interfaces, not concrete classes. Switching from dense to hybrid retrieval in the experiment loop is one line:

```python
# dense + MiniLM — fast baseline
pipeline = RAGPipeline(
    chunker=FixedSizeChunker(chunk_size=512, overlap=64),
    embedder=SentenceTransformersEmbedder("all-MiniLM-L6-v2"),
    retrieval_method="dense",
)

# hybrid + mpnet — same API, different config
pipeline = RAGPipeline(
    chunker=SlidingWindowChunker(window_size=10, step=5),
    embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
    retrieval_method="hybrid",
    alpha=0.6,
)
```

This isn't just cleanliness. It means the experiment driver doesn't need to know what retrieval method is in use — it calls `pipeline.query(text, top_k=5)` and gets back a ranked list of `RetrievalResult` objects regardless.

### Local Models Instead of OpenAI Embeddings

The embedding layer uses [SentenceTransformers](https://sbert.net/) (`all-MiniLM-L6-v2` and `all-mpnet-base-v2`) rather than the OpenAI API. Three reasons:

**Cost.** A 12-cell grid over 50 queries with ~500 chunks per paper generates thousands of embed calls. At OpenAI API rates, this adds up fast on a project that may run 50 grid iterations while tuning.

**Reproducibility.** Local models don't change between runs. OpenAI model updates can shift embeddings — your cached results from last week and your new results may not be directly comparable.

**Meaningful variation.** `all-MiniLM-L6-v2` (22M params, 384 dims) vs `all-mpnet-base-v2` (110M params, 768 dims) span a realistic performance/speed tradeoff. The dimension gap (384 vs 768) means the FAISS index for mpnet-base is exactly twice the memory footprint. That's a real engineering tradeoff worth measuring.

The local models will lose to `text-embedding-3-large` on semantic benchmarks. For a system where you're measuring *relative differences between configurations*, what matters is that both models are consistent and that the gap between them is real and reproducible.

### Document-Level Ground Truth

The previous RAG pipeline (P3) used chunk-level ground truth: each question pointed to a specific chunk UUID. That works for synthetic evaluation but creates a tight coupling between the question and the chunking configuration.

P4 uses document-level ground truth: each question points to a paper ID (`2404.18884v2`), not a specific chunk. The evaluation asks: *did you surface a chunk from the right paper in the top-K results?*

```
Query:     "How does the presence of reputation affect equilibrium outcomes
            in repeated regime change scenarios?"
Relevant:  2404.18884v2
Retrieved: [chunk_7 (2404.18884v2), chunk_3 (2402.12350v3), ...]
→ Rank 1 hit → RR = 1.0
```

This is more realistic — a user asking about a paper doesn't care which specific chunk you return, as long as it's from the right paper. It also makes the evaluation configuration-agnostic: the same qrels file works for fixed-size, recursive, and sliding-window chunking without regenerating QA pairs per config.

The tradeoff: document-level evaluation can't tell you whether retrieval is finding the most relevant section within a paper. That's a meaningful gap — a paper with 50 chunks might have the correct chunk at rank 1 and 49 irrelevant chunks at ranks 2–50. Document-level evaluation scores this as perfect. Chunk-level would penalize it.

### Two-Level Caching: Don't Recompute What You Already Know

Embedding 50 papers' worth of chunks takes minutes and should never happen twice. The pipeline caches at two levels:

1. **Embedding cache** — keyed by `(model_label, chunk_label).pkl`. Same chunking strategy + same model = load from disk, not recomputed.
2. **Result resume** — if `experiments/results/{experiment_id}.json` exists, that cell is skipped. Interrupted runs pick up where they left off. Pass `--force` to redo.

Without caching, every code change to the evaluator means re-embedding from scratch. With caching, evaluation iteration is seconds.

---

## The Experiment Grid

The baseline grid is **3 × 2 × 2 = 12 configurations**, sweeping chunking strategy, embedding model, and retrieval method:

| Dimension | Options |
|---|---|
| **Chunking** | Fixed-512 (512 chars, 64 overlap) · Recursive-512 (paragraph → line → sentence hierarchy) · Sliding-window (10 sentences, step 5) |
| **Embedding** | `all-MiniLM-L6-v2` (384d) · `all-mpnet-base-v2` (768d) |
| **Retrieval** | Dense (FAISS cosine) · Hybrid (dense + BM25, α=0.6) |

Each configuration ingests all papers, builds a shared FAISS index, evaluates against the filtered qrels, and writes a result file with MRR, MAP, Recall@K, NDCG@K, and per-query retrieval times.

### The Three Chunking Strategies

**Fixed-size** (`FixedSizeChunker`) — character-window split with word-boundary awareness. Never cuts mid-word. The most predictable chunk size distribution and the easiest to reason about. The starting point for most RAG systems.

**Recursive** (`RecursiveChunker`) — hierarchical separator hierarchy: paragraph breaks first (`\n\n`), then line breaks (`\n`), then sentence ends (`. `), then spaces. Splits at the coarsest available boundary before falling back to finer granularity. On well-structured academic PDFs with clear paragraph breaks, this preserves document structure better than fixed-size.

**Sliding window** (`SlidingWindowChunker`) — sentence-level window of 10 sentences advancing by 5. Every sentence appears in two consecutive chunks. This maximizes recall at the cost of index size — if the relevant sentence is near a boundary, it's guaranteed to appear fully in at least one chunk.

---

## POC Results: 5 Papers, 9 Queries

The first run was a 5-paper POC on the alphabetically-first papers from the downloaded corpus with 9 matching qrels queries.

| Rank | Configuration | MRR | R@5 | NDCG@5 | Latency |
|---|---|---|---|---|---|
| 1–12 (tie) | All 12 configurations | 1.000 | 1.000 | 1.000 | 18–99ms |

Every single cell hit MRR = 1.000. Perfect score across the board.

**This is the expected result — and the correct one.**

---

### Finding 1: MRR = 1.0 on a 5-Paper Corpus Is Not a Signal

On a corpus with 5 papers and 9 queries, retrieval is easy. Each query has exactly one relevant paper. The FAISS index holds chunks from 5 papers. For a query about algebraic tori and invariant ideals, there is one paper about algebraic tori. Every reasonable embedding model will place that paper's chunks closer to the query than the other four. You can't fail this.

**The POC is not telling you which configuration is best. It's confirming the pipeline is wired correctly end-to-end.** That's exactly what a POC should do. The 12-cell grid at 50 papers and 50 queries is where configurations will actually diverge — when the index has enough noise that embedding model quality and chunking strategy start to matter.

A POC that produces MRR = 0.0 tells you the pipeline is broken. A POC that produces MRR = 1.0 tells you it's not. That's the only question a 5-paper run can answer.

---

### Finding 2: Latency Is the Only Differentiator at Small Scale

With retrieval quality saturated at 1.0, latency is the one dimension where configurations differ. And the pattern is stark:

| Model | Dense latency | Hybrid latency |
|---|---|---|
| `all-MiniLM-L6-v2` | 19–43ms | 18–61ms |
| `all-mpnet-base-v2` | 82–91ms | 91–99ms |

**MiniLM is 3–5× faster than mpnet-base at identical quality on this corpus.** The difference is parameter count: 22M vs 110M, and embedding dimension: 384 vs 768. Smaller model, smaller vectors, faster FAISS search.

At 5 papers, neither model needs 768 dimensions to discriminate between papers. MiniLM's 384 dimensions are sufficient to capture the semantic difference between "algebraic tori" and "reputation in repeated games." The extra dimensions in mpnet-base aren't providing additional signal — they're just adding overhead.

**Practical implication:** run your iterative experiment grid with MiniLM. It's 4× faster per cell, so a 12-cell grid takes 7 minutes instead of 30. Switch to mpnet-base for the final comparison run to verify the quality gap is real at your actual corpus size.

---

### Finding 3: Hybrid Retrieval Adds Nothing at Small Scale

Dense retrieval and hybrid retrieval (dense + BM25, α=0.6) both score MRR = 1.000 on this corpus. But dense is faster — MiniLM dense at 19ms vs MiniLM hybrid at 18–61ms (the hybrid range reflects BM25 initialization variance).

This is the expected pattern. Hybrid retrieval helps when BM25 has signal to contribute — lexically specific queries, author names, arXiv IDs, method acronyms that appear verbatim in the relevant chunk. On 5 papers with 9 questions, the FAISS search is already returning the right paper at rank 1. There is no rank-1 ambiguity for BM25 to resolve.

The hybrid advantage will emerge at scale. When the index contains 50 papers covering overlapping topics — multiple papers about attention mechanisms, multiple papers about transformer training — dense retrieval starts to confuse semantically similar but non-relevant papers. BM25's exact-match behavior becomes a useful tie-breaker for queries that use specific technical vocabulary from one paper.

**The rule:** treat hybrid retrieval as an optimization, not a default. Start with dense, add BM25 when you see dense retrieval confusing semantically similar papers.

---

## The Smoke Test: 16 Seconds to Catch a Broken Pipeline

Running the full 12-cell grid takes 25–30 minutes. A pre-run smoke test — 1 cell, 3 queries, 2 papers — catches wiring bugs in 16 seconds:

```yaml
# config/experiments/smoke.yaml
n_queries: 3
chunking_strategies:
  - strategy: fixed
    chunk_size: 512
    overlap: 64
embedding_models:
  - model: all-MiniLM-L6-v2
retrieval_methods:
  - method: dense
    top_k: 5
```

The tricky part is picking the right 2 papers. If you pick the alphabetically-first 2 papers in the download directory, they may not have any qrels entries — leaving you with a smoke test that produces MRR = 0.0 not because retrieval is broken but because there are no scored queries. Always select smoke test papers from the subset that *has* matching qrels entries.

---

## What Moved Into the Shared Library

Building this system revealed which components belong to a specific pipeline and which belong to all RAG pipelines. The shared `rag_common` library grew by four components during P4 development:

| Component | What it does | Why it's shared |
|---|---|---|
| `RecursiveChunker` | Hierarchical separator splitting | Any pipeline that processes structured documents benefits from this |
| `SlidingWindowChunker` | Sentence-window with configurable step | High-recall chunking strategy useful in any retrieval context |
| `parse_pdf()` | PyMuPDF text extraction | Every pipeline extracts PDFs; they all had the same function |
| `BaseChunker`, `BaseEmbedder`, `BaseRetriever`, `BaseReranker`, `BaseLLM` | Abstract base classes | The ABCs define the contract that makes components swappable across projects |

The original implementations lived in P4's `src/` directory. Moving them to `rag_common` means P3 and P4 share a single implementation — a bug fix in `RecursiveChunker` propagates to both. P4's local files (`src/base.py`, `src/chunkers_ext.py`) became thin re-export shims:

```python
# src/chunkers_ext.py — kept for backwards compatibility
from rag_common.chunkers import RecursiveChunker, SlidingWindowChunker
__all__ = ["RecursiveChunker", "SlidingWindowChunker"]
```

The criterion for promotion: *would this component be useful in a third RAG pipeline without modification?* The ABCs and chunkers passed immediately. The pipeline orchestration (FAISS multi-doc indexing, qrels conversion, experiment resuming) stayed in P4's `src/` — it's specific to this project's structure.

---

## What Comes Next

The 5-paper POC answered: *is the pipeline wired correctly?* The answer is yes.

The 50-paper grid answers: *which configurations actually matter when retrieval gets hard?* This is where the blog gets more interesting. Predictions before running it:

**Recursive chunking should outperform fixed-size on academic papers.** Research papers have clear paragraph structure. Recursive chunking respects those boundaries; fixed-size splits at character 512 regardless of where the paragraph ends. On P3's government statistical document, semantic chunking won for exactly this reason — fixed-size boundaries destroyed multi-sentence claims. Academic PDFs have the same problem with multi-sentence theorem statements and experimental results.

**Hybrid retrieval should start pulling ahead of pure dense.** At 50 papers, the index will contain multiple papers on overlapping topics. A query about "attention sink tokens" may semantically resemble chunks from several transformer papers. BM25's exact-match on "attention sink" will help discriminate the specific paper that introduced the term.

**mpnet-base's 768-dimensional embeddings may start earning their latency cost.** At small scale, the extra dimensions provide no signal. At 50 papers with subtle semantic distinctions, the higher-dimensional space may capture differences that 384-dimensional MiniLM collapses.

These are predictions. Run the grid and find out.

---

## Takeaways

**1. Build for measurement from the start.** A RAG pipeline without evaluation infrastructure is a black box. The experiment grid, caching, resume logic, and per-config result files are the system — the retrieval pipeline is just the thing they measure.

**2. POC results should answer one question: is it broken?** MRR = 1.0 on 5 papers is not a signal about which configuration is best. It's a green light to run the real grid. Don't misread a clean POC as a finding.

**3. Latency is the useful signal at small scale.** When quality is saturated, latency shows you which model to use for iteration. MiniLM at 4× the speed of mpnet-base means 4× more experiment iterations per hour. Save the bigger model for the final evaluation.

**4. Hybrid retrieval is a tuning problem, not a default.** Equal weights (α = 0.5) assumes BM25 and dense are equally useful for your query distribution. They're not. Start with dense. Add hybrid when you see dense retrieval failing on lexically-specific queries, and tune alpha explicitly.

**5. Ground truth format is load-bearing.** Open RAG Benchmark's qrels format is not what standard BEIR loaders expect. Silent MRR = 0.0 is the failure mode — the pipeline runs fine, the evaluation is just wrong. Verify your qrels format before trusting any results.

**6. Shared components compound.** Every chunker added to `rag_common` is available to every future RAG project. The cost of generalizing is one extra layer of abstraction. The benefit is no duplicated bugs.

---

## Run It Yourself

```bash
git clone git@github.com:selizondo/newline_stuff.git
cd newline_stuff/projects

pip install -e rag_common
pip install -e rag_pipeline_experimentation

cd rag_pipeline_experimentation

# Download 5 papers for POC (downloads 50 in background)
python scripts/download_dataset.py --limit 5

# Smoke test — 16 seconds
python scripts/evaluate.py data/smoke_papers/ data/qrels_smoke.json \
  --config config/experiments/smoke.yaml

# Full 12-cell baseline grid
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --top-k 5
```

Results land in `experiments/results/` as one JSON file per configuration. Interrupted runs resume automatically. Add `--force` to rerun all cells.
