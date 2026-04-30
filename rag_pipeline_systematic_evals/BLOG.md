# Systematic RAG Evaluation: What Actually Matters When You Measure It

Most RAG pipelines are built with defaults. Chunk size 512, `text-embedding-3-small`, cosine similarity, done. That works — until it doesn't, and you don't know why.

This post documents a systematic evaluation across 24 configurations of a RAG pipeline on a real document (the US DoJ FY2010 Immigration Statistical Yearbook). The goal: replace intuition with measurement. The surprise: the configuration the spec predicted would win didn't.

---

## Architecture

The pipeline has five stages:

```
PDF → Chunks → Embeddings → Vector Store → Retrieval → Evaluation
                                              ↑
                                    Synthetic QA (per config)
```

The evaluation grid is a full cross-product:

| Dimension | Options |
|---|---|
| **Chunking** | Fixed-256, Fixed-512, Sentence-5s, Semantic |
| **Embedding** | `text-embedding-3-small`, `text-embedding-3-large` |
| **Retrieval** | Vector (FAISS), BM25, Hybrid (α=0.5) |

4 × 2 × 3 = **24 experiments**, each producing Recall@K, Precision@K, MRR, MAP, and NDCG@K.

---

## Design Decision 1: Per-Configuration QA Datasets

This is the most important architectural decision in the whole project.

The naive approach is to generate one QA dataset and evaluate all 24 configurations against it. That's wrong, and here's why: a question generated from chunk `abc-123` in a 256-character configuration references that chunk by UUID. In a 512-character configuration, chunk `abc-123` doesn't exist — the same text lives in a larger chunk with a different ID. The retriever finds the right text but the evaluator scores it as a miss.

The fix: generate a separate QA dataset for every chunking configuration, tied to the chunk IDs from that specific chunking run.

```python
# qa_generator.py — cache invalidation by chunk ID subset check
if cache_path.exists():
    cached = _load_dataset(cache_path)
    current_ids = {c.id_str() for c in chunks}
    cached_ids  = {cid for p in cached.pairs for cid in p.relevant_chunk_ids}
    if cached_ids.issubset(current_ids):
        return cached
    # stale: chunk UUIDs changed, regenerate
```

This cache validation caught a real bug during development. The first run crashed mid-way (OpenAI token limit — a 10-sentence semantic chunk hit 8,192 tokens). The second run produced new chunks with new UUIDs via `uuid4()`, but the cached QA dataset still referenced the old UUIDs. Without the subset check, sentence-based configs silently scored MRR=0.0000 — technically correct, completely misleading.

The lesson: evaluation correctness is as important as pipeline correctness.

---

## Design Decision 2: Truncation Before Embedding

The OpenAI embedding API has a hard 8,192-token limit. Semantic chunking with `max_sentences=10` on dense government prose can produce chunks well above that. The fix is token-aware truncation before any API call:

```python
# embedders.py
_MAX_TOKENS = 8191

def _truncate(text: str) -> str:
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return enc.decode(tokens[:_MAX_TOKENS])
```

The tokenizer (`tiktoken`, cl100k_base) is lazy-loaded — `tiktoken.get_encoding()` costs ~200ms and would be paid at import time on every worker.

---

## Design Decision 3: Caching at Two Layers

Embedding 1,000+ chunks via OpenAI takes 30–60 seconds and costs money. The pipeline caches at two layers:

1. **Embedding cache** — keyed by `(model, chunk_config_label)`. Invalidated by chunk ID mismatch (same text, different UUIDs from a re-parse → stale cache detected).
2. **QA dataset cache** — keyed by `chunk_config_label`. Same invalidation logic.
3. **Experiment resume** — if `experiments/{experiment_id}.json` exists, the cell is skipped. `--force` overrides.

This means the second run of a 24-experiment grid takes seconds, not minutes — which matters when iterating on evaluation code.

---

## Design Decision 4: Batched Embedding with Parallelism

Embedding chunks one-at-a-time is 10–50x slower than batching. The implementation uses `ThreadPoolExecutor` to send multiple batches concurrently:

```python
# embedders.py
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(_embed_batch, enumerate(batches)))
```

OpenAI's embedding endpoint is safe for concurrent requests. For 500 chunks at batch size 100, this sends 5 batches in parallel rather than sequentially — wall time drops from ~25s to ~8s.

The embed function exposed to the retriever is a simple closure over the config:

```python
# grid_search.py
def _make_embed_fn(embed_config: EmbedConfig):
    return lambda texts: embed_texts(texts, embed_config)
```

This keeps `SemanticChunker` and `DenseRetriever` agnostic to the embedding config — they receive a `(list[str]) → ndarray` callable and don't need to know which model or batch size is in use.

---

## Design Decision 5: Execution Order Minimizes Redundant Work

The grid loop runs in chunk → embed → retrieval order. This means:

- **PDF parsed once** — not 24 times
- **Chunking runs 4 times** — once per unique chunk config, cached for all 6 cells that share that config
- **Embedding runs 8 times** — once per (chunk, embed) pair, shared across the 3 retrieval methods

Without this ordering, the naive grid would re-chunk and re-embed 24 times each.

---

## Design Decision 6: Instructor for Structured QA Generation

LLMs don't reliably produce valid JSON. The pipeline uses `instructor` to wrap the OpenAI client and enforce a Pydantic schema:

```python
client = instructor.from_openai(OpenAI())
response: QAPairResponse = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=QAPairResponse,
    messages=[...],
    max_retries=3,
)
```

Instructor handles retries on schema validation failures automatically. Any pair that still fails after 3 retries is logged and skipped — the pipeline never crashes on a bad LLM response. The prompt explicitly forbids copy-paste from the chunk:

> Use natural language (do NOT copy phrases verbatim from the chunk)

This matters for evaluation quality. If questions are just paraphrased excerpts, BM25 will artificially score high (it finds exact keywords), masking real differences between retrieval methods.

---

## Results: Full 24-Experiment Grid

All 24 experiments on `fy10syb.pdf` (119 pages, 190K characters):

| Rank | Configuration | MRR | Recall@5 | NDCG@5 | Latency |
|---|---|---|---|---|---|
| 1 | semantic + large + vector | **0.9280** | **1.000** | 0.9459 | 287ms |
| 2 | semantic + small + vector | 0.9100 | **1.000** | 0.9329 | 327ms |
| 3 | sentence + small + vector | 0.8600 | **1.000** | 0.8941 | 207ms |
| 4 | sentence + large + hybrid | 0.7907 | 0.920 | 0.8209 | 243ms |
| 5 | sentence + small + hybrid | 0.7873 | 0.920 | 0.8182 | 260ms |
| 6 | semantic + large + hybrid | 0.7784 | 0.880 | 0.7934 | 239ms |
| 7 | semantic + small + hybrid | 0.7715 | 0.800 | 0.7607 | 197ms |
| 8 | sentence + large + vector | 0.7607 | 0.960 | 0.8087 | 302ms |
| 9 | sentence + bm25 (both) | 0.6350 | 0.840 | 0.678 | 1-2ms |
| 10 | fixed-512 + large + vector | 0.5698 | 0.600 | 0.5555 | 245ms |
| … | fixed-256 + vector | 0.49-0.51 | 0.60-0.72 | 0.49-0.55 | 214-327ms |
| 23-24 | fixed-256 + bm25 | 0.3223 | 0.360 | 0.325 | 3-4ms |

---

## Finding 1: Semantic Chunking Won — Against the Reference Prediction

The reference implementation (on an unspecified document) reports `fixed_256 + small + vector` as best (MRR=0.963). On `fy10syb.pdf`, the same configuration scores MRR=0.507.

**Why?** `fy10syb.pdf` is a US government statistical yearbook: dense tables, defined terms, statistical categories that span multiple sentences. Fixed-size chunking at 256 characters mid-sentences in the middle of statistical tables — "Of the 1,130,818 persons granted lawful permanent resident status in FY 2010" gets split before the qualifying clause that gives it meaning.

Semantic chunking groups text by meaning, using embedding-based similarity to detect topic boundaries. For this document type — where the "concept" is often a multi-sentence statistical claim — semantic chunks preserve the unit of information that retrieval needs to surface. The questions generated from semantic chunks are also aligned with those boundaries, which reinforces the advantage.

**The broader lesson:** chunk size is a hyperparameter, not a constant. The "256 chars is usually best" rule of thumb comes from documents with short, self-contained paragraphs. For long-form statistical text, the optimal granularity is larger.

---

## Finding 2: Vector Retrieval Dominates

Average MRR by retrieval method across all 24 experiments:

| Method | Avg MRR | Max MRR |
|---|---|---|
| Vector | 0.683 | 0.928 |
| Hybrid | 0.612 | 0.791 |
| BM25 | 0.466 | 0.635 |

Vector search wins because the synthetic questions are paraphrases, not exact excerpts. BM25 needs the query to share vocabulary with the document. When the question is "What criteria governed who qualified for adjustment of status?" and the chunk says "Persons who met the requirements for lawful permanent resident designation," BM25 scores low — no exact term overlap. The embedding captures the semantic equivalence.

BM25's best showing is on sentence-based chunks (MRR=0.635) — likely because sentence boundaries preserve complete syntactic units with consistent vocabulary. Fixed-size chunks disrupt term context, hurting BM25 more than vector search.

---

## Finding 3: Hybrid Underperforms Vector

Hybrid retrieval (α=0.5, equal weight to vector and BM25) consistently scores below vector-only. This is a known failure mode: BM25 scores are unbounded positive numbers; cosine similarity is 0–1. Before combining, both must be min-max normalized to the same range.

The `rag_common` HybridRetriever does normalize scores per query. The issue is that BM25 performance is weak on this document, so giving it 50% weight drags down the combined score regardless of normalization correctness. The optimal `alpha` here would be closer to 0.8–0.9 (vector-dominant) rather than 0.5. Tuning alpha is itself a hyperparameter search.

---

## Finding 4: Large Embedding Model Beats Small — at the Top

Averaged across all experiments, `text-embedding-3-large` (0.589 avg MRR) barely edges out `text-embedding-3-small` (0.585). The gap is negligible in the aggregate.

But at the top of the leaderboard, large wins: semantic + large = 0.928 vs semantic + small = 0.910. For the best configuration, large produces a 2% absolute MRR improvement. At 3072 dimensions vs 1536, it captures finer semantic distinctions — which matters for a document with closely related statistical categories that need precise differentiation.

Cost tradeoff: `text-embedding-3-large` costs roughly 3× more and is marginally slower (287ms vs 327ms — not consistently faster in our runs due to FAISS Flat index being compute-bound, not dimension-bound at this dataset size). Whether 2% MRR justifies 3× embedding cost depends on the application.

---

## Finding 5: MRR = MAP When Ground Truth is 1:1

Every row has MRR = MAP to 4 decimal places. This is not a bug. Average Precision for a single relevant document is `1 / rank_of_first_relevant_result` — which is exactly Reciprocal Rank. When every query has exactly one relevant chunk, MAP = mean(AP) = mean(RR) = MRR.

The spec notes Precision@K is capped at `1/K` for the same reason — one relevant chunk across K retrieved results. This means MRR and Recall@K are the meaningful metrics for this evaluation setup. Precision@K and MAP are mathematically redundant given the 1:1 ground truth structure.

For richer evaluation, consider multi-chunk questions that span 2–3 semantically related chunks. This breaks the 1:1 constraint and makes Precision@K and MAP informative.

---

## Finding 6: BM25 is Fast Enough to Be the Latency Baseline

BM25 retrieval: 1–4ms. Vector retrieval with FAISS Flat index: 200–330ms. The latency is dominated by the embedding call for the query, not the index search.

For a document under 10K chunks (we had ~500–2,000 depending on config), FAISS Flat is fine — it does exhaustive search but the dataset is small. At 100K+ chunks, switching to FAISS IVF or HNSW would drop search time to sub-millisecond even with embeddings pre-cached.

The latency-quality Pareto frontier in our experiments: BM25 on sentence chunks (MRR=0.635, 1ms) vs vector on semantic chunks (MRR=0.928, 287ms). For latency-sensitive production use, BM25 with sentence chunking gives reasonable quality at 200× lower latency. For quality-maximizing applications, vector + semantic is the winner.

---

## Spec Alignment Notes

This implementation covers all required deliverables. Items not implemented or diverging from spec:

| Item | Status | Rationale |
|---|---|---|
| Multiple PDF parsers compared | Not done | One parser used consistently across all experiments — switching parsers changes text, invalidating comparisons |
| Cohere reranker | Not implemented | Optional enhancement; requires Cohere API key |
| Recall@K vs Precision@K scatter | Not implemented | Generated `recall_at_k_curves` by retrieval method instead |
| Metric correlation heatmap | Not implemented | Generated embedding comparison bar chart instead |
| Response time vs quality scatter | Not implemented | Latency data captured in JSON; chart not generated |
| MRR heatmap axis | Diverges | Spec: chunk × embed. Implemented: chunk × retrieval |

The visualization set covers the retrieval-method and chunking-strategy dimensions that explain most of the variance. Extending to include the spec's exact chart set (scatter plots, correlation matrix) is straightforward from the existing DataFrame in `visualizer.py`.

---

## Iteration Log

### Iteration 1: Baseline — fixed_256 + small + vector
- **Hypothesis**: Reference implementation identifies this as best configuration.
- **Result**: MRR=0.507. Well below the 0.963 reference.
- **Decision**: Continue — document is different from reference, investigate all configurations.

### Iteration 2: Token limit crash on semantic chunks
- **Change**: First run crashed mid-way with `openai.BadRequestError: maximum input length is 8192 tokens`.
- **Hypothesis**: Semantic chunks grouping 10 sentences of dense government prose exceed the API limit.
- **Result**: Added `_truncate()` via tiktoken. Pipeline completes.
- **Decision**: Keep. Truncation at 8191 tokens is the correct boundary.

### Iteration 3: Stale QA cache produces MRR=0.0000
- **Change**: Second run after crash used new chunk UUIDs (uuid4()) but loaded cached QA dataset with old UUIDs.
- **Hypothesis**: QA cache needs UUID subset validation, not just label match.
- **Result**: Added subset check in `generate_qa_dataset()`. Re-ran sentence configs. MRR recovered to 0.635–0.860.
- **Decision**: Keep. Cache invalidation by label alone is insufficient when UUIDs are random.

### Iteration 4: Semantic chunking outperforms fixed-size
- **Change**: Ran all 24 configs. Semantic configs sweep the top 2 positions.
- **Hypothesis**: fy10syb.pdf is statistical text; semantic chunking preserves multi-sentence claims better than fixed-size splitting.
- **Result**: semantic + large + vector = MRR 0.928, Recall@5 = 1.000.
- **Decision**: Keep. The document type drives chunking strategy selection — not a universal rule.

---

## Code Architecture

```
src/
├── config.py          # Pydantic models: ChunkConfig, EmbedConfig, RetrievalConfig,
│                      # ExperimentConfig, MetricsResult, EvaluationResult
│                      # build_experiment_grid() → 24 ExperimentConfigs
├── parsers.py         # PDF → ParsedDocument via pdfplumber
├── embedders.py       # Batch embedding with disk cache; lazy tiktoken tokenizer
├── qa_generator.py    # Instructor + OpenAI → per-config QADataset with cache validation
├── evaluator.py       # QADataset + Retriever → EvaluationResult (5 IR metrics)
├── grid_search.py     # Orchestrator: chunk → embed → retrieve → evaluate × 24
├── visualizer.py      # 6 charts from results DataFrame
└── main.py            # CLI: Rich progress, tables, panels
```

All config objects use Pydantic — they validate at construction, serialize to JSON for experiment tracking, and reconstruct cleanly from saved files. The `experiment_id` property on `ExperimentConfig` is derived from config labels (e.g., `fixed_256_ol50__small__vector`), making experiment files human-readable without a lookup table.

---

## Takeaways

1. **Chunk strategy is document-dependent.** For statistical government text, semantic chunking outperforms fixed-size by 80% MRR. For clean prose, the relationship likely reverses. Measure, don't assume.

2. **Evaluation methodology is the hard part.** Getting per-config QA datasets right — with UUID validation, stale cache detection, and prompt engineering to prevent trivially easy questions — took more care than the retrieval implementation itself.

3. **Hybrid retrieval needs alpha tuning.** Equal weighting (α=0.5) consistently underperformed pure vector search on this dataset because BM25 is weak on paraphrase-style questions. Alpha is a hyperparameter; treat it as one.

4. **Cache everything expensive.** Embeddings, QA datasets, experiment results. Systematic evaluation means running the pipeline many times. Without caching, the second run costs as much as the first.

5. **Embedding model size has diminishing returns at this scale.** `text-embedding-3-large` gives 2% MRR improvement at 3× cost for the best configuration. For most production use cases at <10K chunks, `text-embedding-3-small` is the right default unless quality benchmarks demand more.
