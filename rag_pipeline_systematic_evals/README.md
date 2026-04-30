# RAG Pipeline — Systematic Evaluation (P3)

Systematic evaluation of a RAG pipeline across a 4×2×3 configuration grid (24 experiments).
See [rag_pipeline_systematic_evals.md](rag_pipeline_systematic_evals.md) for the full project spec.

---

## Setup

```bash
pip install -e ../rag_common       # shared chunkers, retrievers, metrics, vector store
pip install -e .
cp .env.example .env               # set OPENAI_API_KEY
```

## Run

```bash
# Full grid search against a PDF (24 experiments, 25 QA pairs each)
python -m src.main data/fy10syb.pdf

# Skip chart generation
python -m src.main data/fy10syb.pdf --no-charts

# Re-run even if experiment files already exist
python -m src.main data/fy10syb.pdf --force

# Custom QA pairs and output paths
python -m src.main data/fy10syb.pdf --n-pairs 30 --out-dir runs/ --viz-dir charts/

# Show top 10 configs in summary table
python -m src.main data/fy10syb.pdf --top-n 10
```

## Grid search space

```
4 ChunkConfigs × 2 EmbedModels × 3 RetrievalMethods = 24 Experiments
```

| Dimension | Options |
|---|---|
| **Chunking** | `fixed_256_ol50`, `fixed_512_ol100`, `sentence_5s_ol1`, `semantic_t0.65_max10` |
| **Embedding** | `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d) |
| **Retrieval** | `vector` (FAISS cosine), `bm25` (lexical), `hybrid` (α=0.5) |

## Resume logic

Completed experiments are written to `experiments/{experiment_id}.json`. On re-run,
existing files are loaded and skipped — only missing cells execute. Use `--force` to
re-run all 24 regardless.

## Caching

- **Embeddings** — `data/embed_cache/{model}/{chunk_label}.pkl`. Invalidated when chunk
  UUIDs change (re-parse or re-chunk).
- **QA datasets** — `data/qa_datasets/{chunk_label}.json`. Invalidated when cached chunk
  IDs are no longer a subset of the current chunk set.

## Output

| Path | Contents |
|---|---|
| `experiments/{id}.json` | `EvaluationResult` with full metrics + per-query detail |
| `visualizations/mrr_leaderboard.png` | All 24 configs ranked by MRR |
| `visualizations/recall_at_k_curves.png` | Recall@K curves by retrieval method |
| `visualizations/chunking_comparison.png` | Grouped bar: chunk strategy × metric |
| `visualizations/embedding_comparison.png` | Grouped bar: embed model × metric |
| `visualizations/retrieval_comparison.png` | Grouped bar: retrieval method × metric |
| `visualizations/mrr_heatmap.png` | Heatmap: chunk config × retrieval method |

## Module layout

```
src/
├── config.py        # Pydantic models: ChunkConfig, EmbedConfig, RetrievalConfig,
│                    # ExperimentConfig, MetricsResult, EvaluationResult
│                    # build_experiment_grid() → 24 ExperimentConfigs
├── parsers.py       # PDF → ParsedDocument via pdfplumber
├── embedders.py     # Batch embedding (ThreadPoolExecutor) with tiktoken truncation
│                    # and two-level disk cache; embed_chunks / embed_texts
├── qa_generator.py  # Instructor + gpt-4o-mini → QADataset per chunk config
│                    # Cache validated by chunk UUID subset check
├── evaluator.py     # QADataset + Retriever → EvaluationResult
│                    # Metrics: Recall@K, Precision@K, MRR, MAP, NDCG@K at K=1,3,5,10
├── grid_search.py   # Orchestrator: chunk → embed → retrieve → evaluate × 24
│                    # Execution order minimises redundant work; resume + force flags
├── visualizer.py    # 6 Matplotlib/Seaborn charts from results DataFrame
└── main.py          # CLI: argparse + Rich progress bar, panels, tables

tests/               # pytest — all tests mock OpenAI calls; no API key required
data/
├── fy10syb.pdf      # US DoJ FY2010 Immigration Statistical Yearbook (119 pages)
├── embed_cache/     # Pickled embedding arrays keyed by (model, chunk_label)
└── qa_datasets/     # JSON QADataset per chunk config
experiments/         # One EvaluationResult JSON per completed experiment cell
visualizations/      # PNG charts
```

## Key design notes

- **Per-config QA datasets** — each chunking config generates its own QA dataset tied to
  chunk UUIDs from that specific run. Sharing a dataset across configs produces invalid
  evaluation (chunk IDs won't match).
- **Token truncation** — tiktoken caps chunks at 8,191 tokens before any OpenAI embedding
  call. Semantic chunks on dense text can exceed the API limit without truncation.
- **Batched embedding** — `ThreadPoolExecutor(max_workers=4)` parallelises OpenAI batch
  requests. ~5–8× faster than sequential embedding.
- **Execution order** — grid iterates chunk → embed → retrieval so PDF is parsed once,
  chunking runs 4×, embedding runs 8×, retrieval/eval runs 24×.
- **MRR = MAP** — with 1:1 ground truth (one relevant chunk per question), Average
  Precision = Reciprocal Rank, so MAP = MRR exactly. Focus on MRR and Recall@K.
