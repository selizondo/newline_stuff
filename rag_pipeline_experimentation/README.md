# RAG Pipeline — Experimentation (P4)

Modular RAG system for research papers with a configurable experiment grid: swap chunking strategy, embedding model, and retrieval method via YAML config and get a scored evaluation back in under 30 minutes.

Dataset: [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) — 1,000 arXiv PDFs, 3,045 human-authored QA pairs.
Spec: [rag_pipeline_experimentation.md](rag_pipeline_experimentation.md)

---

## Setup

```bash
pip install -e ../rag_common   # shared library (chunkers, metrics, vector store)
pip install -e .
```

---

## Dataset

```bash
# Download 50 PDFs (default)
python scripts/download_dataset.py

# Download N PDFs
python scripts/download_dataset.py --limit 100

# Metadata only, skip PDFs
python scripts/download_dataset.py --no-pdfs

# Re-download already-existing files
python scripts/download_dataset.py --limit 100 --force
```

Output: `data/papers/*.pdf`, `data/qrels_filtered.json` (queries filtered to downloaded papers only).

---

## Evaluate

```bash
# Smoke test — 1 cell, 2 PDFs, 3 queries (~16s)
python scripts/evaluate.py data/smoke_papers/ data/qrels_smoke.json \
  --config config/experiments/smoke.yaml

# 12-cell baseline grid — 20 PDFs (good for iteration / debugging)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --limit 20

# 12-cell baseline grid — full 100 PDFs, 281 queries
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml

# Re-run all cells (ignore cached results)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --force

# Show top-10 configs in summary table (default is top-3)
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --top-k 10

# Write results to a custom directory
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/baseline.yaml --out-dir experiments/results_v2
```

Results land in `experiments/results/{experiment_id}.json`. Interrupted runs resume automatically — already-completed cells are skipped.

---

## Experiment Configs

| Config | Cells | Description |
|---|---|---|
| `config/experiments/smoke.yaml` | 1 | 1 chunker × 1 embedder × 1 retriever, 3 queries — wiring check |
| `config/experiments/baseline.yaml` | 12 | 3 chunkers × 2 embedders × 2 retrievers, 50 queries |

```bash
# Inspect a config
cat config/experiments/baseline.yaml

# Point at a custom config
python scripts/evaluate.py data/papers/ data/qrels_filtered.json \
  --config config/experiments/my_experiment.yaml
```

---

## Tests

```bash
# Run full test suite (56 tests)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_pipeline.py -v

# Run a single test
python -m pytest tests/test_evaluator.py::TestEvaluate::test_perfect_retrieval_mrr_one -v
```

---

## Results

```bash
# Inspect a result file
python3 -c "
import json
with open('experiments/results/fixed_512_ol64__minilm-l6__dense.json') as f:
    d = json.load(f)
print('metrics:', json.dumps(d['metrics'], indent=2))
print('n_queries:', d['n_queries'])
"

# List all results sorted by MRR
python3 -c "
import json, glob, os
results = []
for f in glob.glob('experiments/results/*.json'):
    d = json.load(open(f))
    results.append((d['metrics']['mrr'], os.path.basename(f)))
for mrr, name in sorted(results, reverse=True):
    print(f'{mrr:.4f}  {name}')
"
```

---

## Project Layout

```
rag_pipeline_experimentation/
├── config/experiments/          # YAML experiment grids
│   ├── smoke.yaml               # 1-cell smoke test
│   └── baseline.yaml            # 12-cell baseline grid
├── data/
│   ├── papers/                  # downloaded arXiv PDFs (100 currently)
│   ├── smoke_papers/            # 2 PDFs for smoke test
│   ├── qrels_filtered.json      # 281 queries → 100 papers
│   ├── qrels_smoke.json         # 3 queries → 2 papers
│   ├── embed_cache/             # cached embeddings keyed by (model, chunk_label)
│   └── indices/                 # FAISS indices per experiment cell
├── experiments/results/         # one JSON per completed experiment cell
├── scripts/
│   ├── download_dataset.py      # fetch PDFs + build qrels_filtered.json
│   ├── evaluate.py              # run experiment grid, write results
│   └── ingest.py                # ingest a single PDF directory
├── src/
│   ├── config.py                # YAML parsing, EmbedModelName enum
│   ├── embedders.py             # SentenceTransformersEmbedder (+ disk cache)
│   ├── evaluator.py             # qrels loading, per-query scoring, result schema
│   ├── experiment.py            # grid runner, resume logic, chunker/embedder factory
│   ├── models.py                # Document dataclass
│   ├── pipeline.py              # RAGPipeline (ingest + query orchestrator)
│   ├── base.py                  # re-exports from rag_common.base
│   └── chunkers_ext.py          # re-exports from rag_common.chunkers
├── tests/                       # 56 tests
├── blog_RAG_System_Experiments.md
└── pyproject.toml
```

---

## Shared Library

Components promoted to `rag_common` (available to all RAG projects):

| Component | Import |
|---|---|
| `RecursiveChunker`, `SlidingWindowChunker` | `from rag_common.chunkers import ...` |
| `parse_pdf()` | `from rag_common.parsers import parse_pdf` |
| `BaseChunker`, `BaseEmbedder`, `BaseRetriever`, `BaseReranker`, `BaseLLM` | `from rag_common.base import ...` |
