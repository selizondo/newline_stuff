# Mini-Project 3. RAG Pipeline for PDF Documents: Systematic Retrieval Optimization

## 🎯 Project Goal

Build a **Retrieval-Augmented Generation (RAG) pipeline** that ingests PDF documents, splits them into chunks using multiple strategies, embeds them with different models, stores them in vector databases, and retrieves relevant chunks using various retrieval methods. The system must include an **automated evaluation framework** that systematically compares configurations across a grid of chunking x embedding x retrieval combinations to find the optimal setup.

**Core Challenge**: Design a fair evaluation methodology where synthetic QA datasets are generated per chunking configuration (not shared across configurations), enabling apples-to-apples comparison of RAG pipeline hyperparameters.

***

## 🧠 The Problem Context

RAG systems are the backbone of modern AI applications that need to answer questions from proprietary documents. But building a RAG pipeline involves dozens of design decisions: chunk size, overlap, embedding model, retrieval method, each affecting quality in non-obvious ways. Most teams pick defaults and never measure the impact.

### Dataset

Use one of the following PDF documents as your data source:

* `any.pdf**` - Download from: [https://www.kaggle.com/datasets/rrr3try/enterprise-rag-markdown](https://www.kaggle.com/datasets/rrr3try/enterprise-rag-markdown) **OR**
* `fy10syb.pdf**` - Download from: [https://drive.google.com/file/d/1FlzJqXRzjnRVVE7x6viBUJcGl58p5czA/view?usp=sharing](https://drive.google.com/file/d/1FlzJqXRzjnRVVE7x6viBUJcGl58p5czA/view?usp=sharing)

Either PDF is suitable. Pick one and use it consistently across all experiments.

Your system must:

* **Parse PDFs** using multiple extraction libraries and compare their output quality
* **Chunk text** using at least 3 different strategies (fixed-size, sentence-based, semantic)
* **Embed chunks** using multiple OpenAI embedding models
* **Store vectors** in at least one vector database (FAISS for local, optionally Turbopuffer for cloud)
* **Retrieve** using BM25 (lexical), vector search (semantic), and hybrid (combined) methods
* **Generate synthetic QA datasets** per chunking configuration using an LLM
* **Evaluate** retrieval quality with standard IR metrics across all configurations
* **Identify the best configuration** through systematic grid search

**Real-world motivation:** In production RAG systems, the difference between a 60% MRR and a 96% MRR is the difference between a frustrating chatbot and a useful assistant. Systematic evaluation is how you find the configuration that actually works.

***

## 🏗 System Architecture Overview

Your pipeline should follow this high-level flow:

**Ingestion Phase**: PDF Parser (pdfplumber, PyPDF2, PyMuPDF) → Chunking Strategies (fixed-size, sentence-based, semantic) → Chunk Storage with metadata

**Embedding Phase**: Chunks → Embedding Models (text-embedding-3-small, text-embedding-3-large) → Vector Database (FAISS, optionally Turbopuffer)

**QA Generation Phase**: Chunks → LLM-powered Synthetic QA Dataset Generation (per chunking configuration) → QA pairs with ground-truth chunk IDs

**Retrieval Phase**: QA Questions → Retrieval Methods (BM25 lexical, Vector semantic, Hybrid combined) → Ranked Results

**Evaluation Phase**: Retrieved Results + Ground Truth → Evaluation Engine (Recall\@K, Precision\@K, MRR, MAP, NDCG\@K) → Results and Best Configuration

### Grid Search Space

The system evaluates configurations across a combinatorial grid:

```
4 Chunking Configs × 2 Embedding Models × 3 Retrieval Methods = 24 Experiments
```

Each experiment produces a full set of IR metrics, enabling systematic comparison.

***

## 📊 Success Metrics

### Primary Metrics (Must Achieve)

| ​ | Metric                           | Target | Description                                              |
| - | -------------------------------- | ------ | -------------------------------------------------------- |
| ​ | **MRR (Mean Reciprocal Rank)**   | ≥ 0.85 | The relevant chunk should appear near the top of results |
| ​ | **Recall\@5**                    | ≥ 0.90 | At least 90% of relevant chunks found in top 5 results   |
| ​ | **MAP (Mean Average Precision)** | ≥ 0.80 | Overall ranking quality across all queries               |
| ​ | **NDCG\@5**                      | ≥ 0.85 | Normalized ranking quality accounting for position       |

### Reference Best Results (Achieved in Reference Implementation)

| ​ | Metric    | Best Score | Configuration                                                                |
| - | --------- | ---------- | ---------------------------------------------------------------------------- |
| ​ | MRR       | 0.963      | fixed\_size (256 chars, 50 overlap) + text-embedding-3-small + vector search |
| ​ | Recall\@5 | 1.000      | Same configuration                                                           |
| ​ | MAP       | 0.963      | Same configuration                                                           |
| ​ | NDCG\@5   | 0.975      | Same configuration                                                           |

### Secondary Metrics

| ​ | Metric                    | Target          | Description                                                      |
| - | ------------------------- | --------------- | ---------------------------------------------------------------- |
| ​ | **Configurations tested** | ≥ 12            | At least 12 unique chunking × embedding × retrieval combinations |
| ​ | **Chunking strategies**   | ≥ 3             | Must compare at least 3 different chunking approaches            |
| ​ | **Embedding models**      | ≥ 2             | Must compare at least 2 embedding models                         |
| ​ | **QA dataset size**       | ≥ 20 per config | At least 20 synthetic questions per chunking configuration       |

> **Note on Precision\@K:** Because each synthetic question maps to exactly 1 ground-truth chunk, Precision\@K is mathematically capped at `1/K` (e.g., Precision\@5 max = 0.20). This is expected behavior, not a system failure. Focus on MRR and Recall\@K as your primary quality indicators.

***

## 🛠 Technical Requirements

### Required Technology Stack

| ​ | Component               | Options                                  | Purpose                                                                 |
| - | ----------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| ​ | **Language**            | Python 3.10+                             | Core implementation                                                     |
| ​ | **PDF Parsing**         | pdfplumber, PyPDF2, PyMuPDF (any or all) | Text extraction from PDFs                                               |
| ​ | **Text Chunking**       | Custom implementations                   | Fixed-size, sentence-based, semantic splitting                          |
| ​ | **Embeddings**          | OpenAI Embedding API                     | text-embedding-3-small, text-embedding-3-large                          |
| ​ | **Structured Output**   | Instructor                               | Generate structured QA aligned with chunk IDs                           |
| ​ | **Vector Database**     | FAISS (required), Turbopuffer (optional) | Vector storage and similarity search                                    |
| ​ | **Lexical Search**      | rank-bm25                                | BM25 keyword-based retrieval                                            |
| ​ | **NLP**                 | NLTK, spaCy                              | Sentence tokenization, semantic boundary detection                      |
| ​ | **LLM Provider**        | OpenRouter or OpenAI                     | LLM generations (e.g., `deepseek/deepseek-r1-0528:free` via OpenRouter) |
| ​ | **Embedding Switching** | LiteLLM or OpenAI                        | Switch between different embedding model providers                      |
| ​ | **Data Validation**     | Pydantic                                 | Configuration and data model validation                                 |
| ​ | **Output**              | Rich                                     | Formatted console output with tables and progress bars                  |
| ​ | **Batch Processing**    | ThreadPoolExecutor                       | Parallel batch embedding for performance                                |

### Enhancements

| ​                                         | Enhancement                                   | Purpose                                                                   |
| ----------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| ​optional can use any opensource reranker | **Cohere Reranker**                           | Rerank top-k results and compare with/without reranking                   |
| ​                                         | **Turbopuffer**                               | Cloud vector database alongside local FAISS                               |
| ​optional                                 | **Neo4j / Memgraph**                          | Knowledge graph for storing relationships between nodes                   |
| ​                                         | **Logfire**                                   | Tracing, logging, debugging, and inspecting LLM inputs/outputs at runtime |
| ​                                         | **Braintrust**                                | Logging, monitoring, comprehensive debugging and evaluation               |
| ​optional                                 | **judgy, can use your own llm-as-judge code** | LLM-as-Judge for evaluating synthetic QA dataset accuracy                 |
| ​                                         | **SentenceTransformers**                      | Alternative embeddings with MiniLM / Instructor-XL                        |

> **JSON Validation Warning:** LLMs do not always return valid JSON. Use **Pydantic** models with **Instructor** to enforce structured output from LLM calls. Always validate generated QA pairs and chunk metadata against your Pydantic schemas before using them in evaluation. Log and skip any records that fail validation rather than crashing the pipeline.

***

## 📋 Data Models & Interfaces

### Chunk

Represents a piece of text extracted and split from the PDF:

```
{
  "id": "uuid-string",
  "text": "The extracted chunk content...",
  "page_number": 5,
  "chunk_index": 42,
  "start_char": 1200,
  "end_char": 1456,
  "method": "fixed_size",
  "metadata": {
    "chunk_size": 256,
    "overlap": 50,
    "parser": "pdfplumber"
  }
}
```

### QA Example (for evaluation)

Each synthetic question maps to the chunk(s) it was generated from:

```
{
  "question": "What is the role of activation functions in neural networks?",
  "relevant_chunk_ids": ["chunk-uuid-123"],
  "metadata": {
    "source_page": 12,
    "chunk_method": "fixed_size",
    "synthetic": true,
    "generation_method": "openai_gpt35"
  }
}
```

### Evaluation Metrics Result

```
{
  "experiment_id": "text-embedding-3-small_fixed_size_256_vector",
  "embedding_model": "text-embedding-3-small",
  "chunking_method": "pdfplumber_fixed_size_chunk_size256_overlap50",
  "retrieval_method": "vector_text-embedding-3-small",
  "use_reranking": false,
  "metrics": {
    "recall_at_k": { "1": 0.85, "3": 0.95, "5": 1.0, "10": 1.0 },
    "precision_at_k": { "1": 0.85, "3": 0.317, "5": 0.20, "10": 0.10 },
    "mrr": 0.963,
    "map_score": 0.963,
    "ndcg_at_k": { "1": 0.85, "3": 0.933, "5": 0.975, "10": 0.975 },
    "total_queries": 20,
    "avg_retrieval_time": 0.045
  }
}
```

### Chunking Configuration

```
{
  "parser": "pdfplumber",
  "chunker": "fixed_size",
  "chunk_size": 256,
  "overlap": 50
}
```

***

## 🔧 Key Implementation Challenges

### Challenge 1: Fair Evaluation Across Chunking Methods

**The problem:** If you generate one QA dataset from chunks produced by method A, then evaluate method B using the same questions, the evaluation is biased. The questions were designed around method A's chunk boundaries.

**What to think about:** Each chunking configuration produces different chunk IDs and different text boundaries. A question generated from chunk X in a 256-char configuration won't have a matching chunk ID in a 512-char configuration.

**Why it matters:** Without per-configuration QA datasets, your evaluation results are meaningless. You'd be comparing apples to oranges, concluding that one chunking method is "better" when really the QA dataset just happened to align with its boundaries.

### Challenge 2: Chunking Strategy Design

**The problem:** Different chunking strategies make fundamentally different tradeoffs:

| ​ | Strategy           | Pros                   | Cons                                        |
| - | ------------------ | ---------------------- | ------------------------------------------- |
| ​ | **Fixed-size**     | Predictable, simple    | May split mid-sentence                      |
| ​ | **Sentence-based** | Preserves meaning      | Variable size, some sentences too short     |
| ​ | **Semantic**       | Groups related content | Computationally expensive, needs NLP models |

**What to think about:** Fixed-size chunking needs word-boundary awareness (don't split "neural" from "network"). Sentence chunking needs a reliable sentence tokenizer. Semantic chunking needs spaCy or similar for boundary detection.

**Why it matters:** Chunk quality directly determines retrieval quality. A perfectly-split chunk that captures a complete concept will always retrieve better than a chunk that cuts a sentence in half.

### Challenge 3: Embedding Model Comparison

**The problem:** Different embedding models produce vectors of different dimensions and capture different semantic relationships. You need to compare them fairly.

**What to think about:**

* `text-embedding-3-small` (1536 dims): fast, cheap, good for most use cases
* `text-embedding-3-large` (3072 dims): slower, more expensive, captures finer distinctions
* Batch processing is essential. Embedding chunks one-by-one is 10-50x slower than batching

**Why it matters:** Embedding model choice affects both quality and cost. In production, a 2% quality improvement might not justify 3x the cost and latency.

### Challenge 4: Retrieval Method Design

**The problem:** Three fundamentally different retrieval approaches need to work together:

* **BM25 (lexical):** Exact keyword matching. Great when the query uses the same terms as the document
* **Vector search (semantic):** Meaning-based matching. Finds paraphrased or conceptually related content
* **Hybrid:** Weighted combination of both, but how do you normalize and combine scores from completely different scoring systems?

**What to think about:** BM25 scores are unbounded positive numbers. Vector similarity scores are typically 0-1 cosine distances. You must normalize both to the same range before combining. The `alpha` weight parameter (vector vs BM25 contribution) is another hyperparameter to tune.

**Why it matters:** Real-world queries mix exact terminology with conceptual questions. Hybrid retrieval covers both cases, but only if the score combination is done correctly.

### Challenge 5: Synthetic QA Generation Quality

**The problem:** The quality of your evaluation depends entirely on the quality of your synthetic questions. LLM-generated questions tend to be too easy (they use exact phrases from the chunk) or too hard (they ask about information not in the chunk).

**What to think about:** Good evaluation questions should:

* Be answerable from the source chunk
* Use natural language (not copy-paste from the chunk)
* Cover different question types (factual, conceptual, comparative)
* Have clear, unambiguous answers

**Why it matters:** If your synthetic questions are trivially easy (they contain unique keywords from the chunk), even BM25 will score 100%. If they're too hard or poorly generated, no method will score well. Neither extreme tells you anything useful about your system.

**Better approaches to explore:**

* **Multi-chunk questions**: Instead of 1 question to 1 chunk, create questions spanning 2-3 semantically similar chunks with multiple `relevant_chunk_ids`
* **Question type diversity**: Generate different types: factual ("What is X?"), comparative ("How does X differ from Y?"), analytical ("Why does X happen?"), summarization, multi-hop
* **Hierarchical questions**: Page-level questions (multiple chunks), section-level (few chunks), paragraph-level (single chunk)
* **Real-world question patterns**: Domain-specific templates like "Define \{concept} and explain its significance" or "Compare and contrast \{concept\_a} with \{concept\_b}"
* **LLM-generated question chains**: Prompt the LLM to generate a basic factual question, a deeper analytical question, and a cross-concept question from each chunk

### Challenge 6: Cohere Reranking Comparison

**The problem:** After initial retrieval, a reranker can re-score the top-K results using a cross-encoder model, potentially improving ranking quality.

**What to think about:** Compare your best configuration **with and without** Cohere reranking. Measure the delta in MRR, Recall\@K, and MAP.

**Why it matters:** Reranking adds latency and cost. You need to measure whether the quality improvement justifies it for your specific dataset and use case.

***

## 📦 Deliverables

* **PDF Processing Module**: Parser abstraction supporting multiple PDF libraries, with pluggable chunking strategies (fixed-size, sentence, semantic)
* **Embedding System**: Batch embedding with multiple OpenAI models, caching/saving of embeddings to avoid re-computation
* **Vector Database Integration**: FAISS local index with full CRUD operations; optional Turbopuffer cloud integration
* **Retrieval System**: BM25, vector search, and hybrid retrieval methods with configurable parameters
* **Synthetic QA Generator**: LLM-powered question generation from chunks, producing per-configuration QA datasets
* **Evaluation Engine**: Calculation of Recall\@K, Precision\@K, MRR, MAP, NDCG\@K with formatted results display
* **Grid Search Runner**: Orchestration script that runs all configuration combinations, collects results, identifies best configuration
* **Results Output**: JSON files with all experiment results and best configuration summary; Rich-formatted console tables

***

## 🧪 Evaluation Approach

Follow these steps in order. Record every result in your iteration log.

### Metric Definitions

* **Recall\@K**: Of all relevant chunks, what fraction appears in the top K results? `Recall@K = |relevant ∩ retrieved_top_k| / |relevant|`
* **Precision\@K**: Of the top K results, what fraction is relevant? `Precision@K = |relevant ∩ retrieved_top_k| / K`
* **MRR (Mean Reciprocal Rank)**: How high does the first relevant chunk appear? `MRR = (1/Q) × Σ (1 / rank_of_first_relevant_chunk)`
* **MAP (Mean Average Precision)**: Average precision calculated at each relevant result position, averaged across queries.
* **NDCG\@K**: Ranking quality that gives more credit to relevant results appearing earlier. `DCG@K = Σ (relevance_i / log2(i + 1))`, `NDCG@K = DCG@K / IDCG@K`

### Known Limitation: 1-to-1 Ground Truth

Each synthetic question maps to exactly 1 ground truth chunk. This means **Precision\@K** is mathematically capped at `1/K` (e.g., Precision\@5 max = 0.20). This is not a system failure, it's a limitation of the evaluation methodology. MRR and Recall\@K are more meaningful metrics in this setup.

### Step 1: Validate PDF Parsing and Chunking

Parse the PDF with your chosen library. Verify text extraction quality and implement at least 3 chunking strategies. Spot-check chunk boundaries for completeness.

Example output:

| Chunking Method | Chunk Size | Overlap  | Total Chunks | Avg Chunk Length | Boundary Quality           |
| --------------- | ---------- | -------- | ------------ | ---------------- | -------------------------- |
| fixed\_size     | 256        | 50       | 342          | 248 chars        | Some mid-sentence splits   |
| fixed\_size     | 512        | 100      | 178          | 495 chars        | Fewer splits, more context |
| sentence        | 5 sent     | 1 sent   | 215          | 380 chars        | Clean sentence boundaries  |
| semantic        | 300 tok    | adaptive | 195          | 410 chars        | Groups related content     |

**If total chunks \< 100**, the PDF may be too short or parsing failed. Check extraction output. **If boundary quality is poor for fixed-size**, implement word-boundary awareness to avoid splitting mid-word.

### Step 2: Validate Synthetic QA Generation

Generate at least 20 QA pairs per chunking configuration. Verify that each question maps to a valid chunk ID from that specific configuration. Spot-check 5 questions per config for quality.

Example output:

| Chunking Config        | QA Pairs Generated | Valid Chunk IDs | Avg Question Length | Quality (spot-check)   |
| ---------------------- | ------------------ | --------------- | ------------------- | ---------------------- |
| fixed\_256\_overlap50  | 25                 | 25/25 (100%)    | 12 words            | Good, natural phrasing |
| fixed\_512\_overlap100 | 22                 | 22/22 (100%)    | 14 words            | Good, some too easy    |
| sentence\_5sent        | 20                 | 20/20 (100%)    | 11 words            | Good                   |
| semantic\_300tok       | 21                 | 21/21 (100%)    | 13 words            | Good                   |

**If any QA pair references a non-existent chunk ID**, the generation is not aligned with the chunking output. Re-generate with correct chunk references. **If questions are too easy** (contain exact phrases from the chunk), adjust the prompt to require paraphrasing.

### Step 3: Verify Retrieval Method Correctness

Test each retrieval method (BM25, vector, hybrid) on a small set of queries. Verify that results are ranked and scored correctly.

Example output:

| Retrieval Method   | Avg Recall\@5 | Avg MRR | Avg Response Time | Score Range |
| ------------------ | ------------- | ------- | ----------------- | ----------- |
| BM25               | 0.85          | 0.68    | 0.02s             | 2.1 - 18.7  |
| Vector (3-small)   | 1.00          | 0.96    | 0.45s             | 0.72 - 0.99 |
| Vector (3-large)   | 0.95          | 0.92    | 0.38s             | 0.68 - 0.97 |
| Hybrid (alpha=0.7) | 0.90          | 0.05    | 0.47s             | 0.0 - 1.0   |

**If hybrid MRR is near zero**, score normalization is broken. BM25 scores and cosine similarities must be normalized to the same range (e.g., min-max to 0-1) before combining. **If BM25 returns empty results**, check tokenization and ensure the BM25 index was built from the correct chunks.

### Step 4: Run Full Grid Search

Execute all configuration combinations. Target at least 12 unique experiments (4 chunking x 2 embedding x 3 retrieval = 24 possible).

Example output:

| Experiment                  | Recall\@5 | Precision\@5 | MRR   | MAP   | NDCG\@5 | Avg Time |
| --------------------------- | --------- | ------------ | ----- | ----- | ------- | -------- |
| 3-small\_fixed\_256\_vector | 1.000     | 0.200        | 0.963 | 0.963 | 0.975   | 0.470s   |
| 3-large\_fixed\_256\_vector | 0.950     | 0.190        | 0.919 | 0.919 | 0.925   | 0.380s   |
| 3-small\_fixed\_512\_vector | 1.000     | 0.200        | 0.904 | 0.904 | 0.957   | 0.369s   |
| 3-small\_fixed\_256\_bm25   | 0.800     | 0.160        | 0.665 | 0.665 | 0.720   | 0.020s   |

**If best MRR \< 0.85**, review chunking quality (are concepts split across chunks?) and QA quality (are questions too hard or ambiguous?). **If fewer than 12 experiments completed**, check for crashes in specific configurations and add error handling.

### Step 5: Validate Results Against Reference Insights

Compare your findings to these reference patterns:

| Finding                 | Reference Evidence                          | Your Result Matches? |
| ----------------------- | ------------------------------------------- | -------------------- |
| Smaller chunks win      | 256-char (96.3% MRR) > 512-char (90.4% MRR) | Check your data      |
| 3-small beats 3-large   | 3-small (96.3%) > 3-large (91.9%)           | Check your data      |
| Vector >> BM25          | Vector (96.3%) >> BM25 (66.5-70%)           | Check your data      |
| Hybrid underperforms    | Hybrid (0-6% MRR) vs Vector (96.3%)         | Check your data      |
| Perfect recall possible | 100% Recall\@5 achieved                     | Check your data      |

**If your results diverge significantly from the reference**, investigate whether the divergence is due to a bug (e.g., broken score normalization in hybrid) or a legitimate difference (e.g., different PDF content). Document the divergence in your iteration log.

### Step 6: Self-Evaluation Questions

After completing steps 1-5, answer these questions honestly:

* Does my best configuration achieve MRR >= 0.85 on the synthetic QA dataset?
* Have I tested at least 12 different configurations in a systematic grid search?
* Are my synthetic QA datasets generated separately for each chunking configuration with matching chunk IDs?
* Do my evaluation metrics match the expected patterns (high recall, low precision due to 1:1 ground truth)?
* Can I explain why one configuration outperforms another based on the metrics?
* Does my hybrid retrieval correctly normalize and combine BM25 and vector scores?
* Have I compared Cohere reranking with and without to measure its impact?

### Expected CLI Output (Reference)

Your final output should display a Rich-formatted table similar to this:

```
Step 5: Analyzing results
                                RAG Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Experiment                ┃ Recall@5 ┃ Precision@5 ┃   MRR ┃   MAP ┃ NDCG@5 ┃ Avg Time (s) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ embed-3-small_fixed_256…  │    1.000 │       0.200 │ 0.963 │ 0.963 │  0.975 │        0.470 │
│ embed-3-large_fixed_256…  │    0.950 │       0.190 │ 0.919 │ 0.919 │  0.925 │        0.380 │
│ embed-3-small_fixed_512…  │    1.000 │       0.200 │ 0.904 │ 0.904 │  0.957 │        0.369 │
│ ...                       │      ... │         ... │   ... │   ... │    ... │          ... │
└───────────────────────────┴──────────┴─────────────┴───────┴───────┴────────┴──────────────┘

Best Configurations:
Best configuration by MRR:
  Experiment: text-embedding-3-small_pdfplumber_fixed_size_chunk_size256_overlap50_vector
  Score: 0.963
Total experiments: 24
```

***

## 🧭 First Principles

### Why Chunk Size Matters

| ​ | Chunk Size                 | Effect                                                 |
| - | -------------------------- | ------------------------------------------------------ |
| ​ | Too small (\< 128 chars)   | Loses context, fragments concepts across chunks        |
| ​ | Sweet spot (256-512 chars) | Captures complete ideas, good retrieval granularity    |
| ​ | Too large (> 1024 chars)   | Dilutes signal, embeds too many concepts in one vector |

Overlap ensures continuity. Concepts at chunk boundaries aren't lost. 10-20% overlap is typical.

### Chunking Design Guidelines

Choosing `chunk_size` and `overlap` is **not arbitrary** . Treat them as hyperparameters tuned by retrieval metrics.

| ​ | Chunking Goal           | Chunk Size (tokens) | Overlap (tokens) | Why                                                   |
| - | ----------------------- | ------------------- | ---------------- | ----------------------------------------------------- |
| ​ | Preserve semantic units | 100-300             | 20-50            | Avoid breaking sentences, definitions, tables mid-way |
| ​ | Enable dense search     | 256-512             | 64-128           | Enough context for embeddings                         |
| ​ | Support answer tracing  | ≤512                | 64-128           | Needed for citations and span evaluation              |
| ​ | Maximize retrievability | 128-256             | High (\~50%)     | Improves recall, increases redundancy                 |
| ​ | Support long reasoning  | 512-1024            | Low or adaptive  | Rare, used with summarizers or rerankers              |

**Analogy:** Choosing chunk size is like choosing puzzle piece size. Too big loses granularity, too small creates meaningless fragments, overlap provides shared edges for context.

### Model Context Window Constraint

Your retriever and generator have a context limit. Rule of thumb:

```
max_context ≈ k × chunk_size
```

Example: Top-4 retrieval with `chunk_size = 1024` → \~4096 tokens per call.

If chunk size is too large:

* Retrieve fewer documents → **lower recall**
* Burn token budget faster → **higher cost, less grounding**

### Layout-Aware Chunking

PDFs are **visually formatted**: paragraphs, sections, bullet lists. Do **not** blindly split every N tokens. Instead:

* Use **layout-aware chunking** (tools: `pdfplumber` , `PyMuPDF` , `unstructured` )
* Chunk by **text structure** (headings, paragraphs, lists)
* Then tokenize and truncate or pad

This often outperforms fixed-size chunking for contracts, reports, academic papers, and manuals.

### Why Separate QA Datasets Per Chunking Method

Think of it this way: if you generate a question from chunk #42 in a 256-char configuration, that question's ground truth is `chunk_42_id` . But in a 512-char configuration, chunk #42 doesn't exist. The same text might be in chunk #21. The question's ground truth chunk ID is meaningless for the other configuration.

**Solution:** Generate separate QA datasets for each chunking configuration. Each question is tied to a specific chunk from that specific configuration.

### Why Hybrid Retrieval

BM25 excels at exact term matching ("What is backpropagation?"). Vector search excels at semantic matching ("How do neural networks learn?"). Real-world queries span both types. Hybrid retrieval with proper score normalization covers the full spectrum.

### Why MRR is the Primary Metric

In a RAG pipeline, the LLM typically uses the top 1-3 retrieved chunks to generate an answer. If the relevant chunk is at position 1, the answer will be great. Position 5, maybe. Position 50, it's useless. MRR captures this: it heavily penalizes relevant results that appear too far down the list.

***

## 💡 Getting Started Hints

### Recommended Development Order

* **Start with PDF parsing**: Get text extraction working with at least one PDF library. Verify the output makes sense.
* **Implement fixed-size chunking first**: It's the simplest strategy. Get the full pipeline working end-to-end with this one method before adding sentence and semantic chunking.
* **Build the embedding pipeline**: Start with one model ( `text-embedding-3-small` ). Implement batch processing and saving/loading to avoid re-embedding.
* **Set up FAISS**: Get vector storage and retrieval working locally before attempting cloud databases.
* **Implement BM25**: It's the simplest retrieval method and doesn't need embeddings. Good baseline.
* **Build the evaluation framework**: Implement metrics calculation and QA dataset generation. This is the core differentiator of the project.
* **Scale to grid search**: Add more chunking configs, embedding models, and retrieval methods. Run the full evaluation grid.

### Common Pitfalls

* **Don't share QA datasets across chunking configs.** This is the #1 mistake. Each chunking method needs its own QA dataset with matching chunk IDs.
* **Don't forget to normalize scores in hybrid retrieval.** BM25 scores and cosine distances are on completely different scales.
* **Don't embed chunks one at a time.** Use batch embedding, it's 10-50x faster.
* **Save intermediate results.** Embeddings are expensive to compute. Save them to disk so you don't re-compute when iterating on evaluation code.
* **Use a consistent PDF parser across experiments.** Switching parsers changes the text, which changes the chunks, which invalidates comparisons.

### Chunking Guidelines Reference

| ​ | Parameter             | Range to Explore          | Guidance                        |
| - | --------------------- | ------------------------- | ------------------------------- |
| ​ | Chunk size            | 128, 256, 512, 1024 chars | Start with 256-512, then widen  |
| ​ | Overlap               | 0, 50, 100, 200 chars     | 10-20% of chunk size works well |
| ​ | Sentences per chunk   | 3, 5, 7                   | For sentence-based chunking     |
| ​ | Max tokens (semantic) | 200, 300, 500             | For semantic chunking           |

***

## 📚 Key Concepts to Understand

### BM25 (Best Matching 25)

A probabilistic ranking function for text retrieval. It scores documents based on query term frequency, adjusted by document length. No embeddings needed, pure keyword matching.

```
BM25(q, d) = Σ IDF(qi) × (tf(qi, d) × (k1 + 1)) / (tf(qi, d) + k1 × (1 - b + b × |d|/avgdl))
```

* **IDF:** Rare terms get more weight
* **TF saturation:** Diminishing returns for repeated terms
* **Length normalization:** Longer documents don't automatically score higher

### Cosine Similarity vs L2 Distance

Vector databases use distance metrics to find similar embeddings:

* **Cosine similarity:** Measures angle between vectors (0 = orthogonal, 1 = identical). Direction matters, magnitude doesn't.
* **L2 (Euclidean) distance:** Measures straight-line distance. Lower = more similar.

FAISS uses L2 by default. Turbopuffer uses cosine distance. Both work; just be consistent.

### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality by assigning decreasing importance to results lower in the list:

```
Position 1: full credit (÷ log2(2) = ÷1)
Position 2: 63% credit (÷ log2(3))
Position 3: 50% credit (÷ log2(4))
Position 5: 39% credit (÷ log2(6))
```

A perfect NDCG\@K of 1.0 means all relevant documents appear at the very top of the ranked list.

### Vector Index Types

* **Flat (exhaustive):** Compares query against every vector. Perfect recall, O(n) time. Use for \< 100K vectors.
* **IVF (Inverted File):** Clusters vectors, only searches relevant clusters. Approximate, but much faster for large collections.
* **HNSW:** Graph-based approximate search. Best speed/accuracy tradeoff for most use cases.

For this project, **Flat index is sufficient** given the dataset size (\< 10K chunks).

***

## 📊 Visualization Requirements

All charts must be generated using **Matplotlib**, **Seaborn**, or **Plotly**. Save each chart as a PNG file in a `visualizations/` directory.

* **MRR Comparison Bar Chart**: Bar chart comparing MRR across all experiment configurations (x-axis: experiment name, y-axis: MRR score). Use color coding by retrieval method (BM25, Vector, Hybrid).
* **Recall\@K vs Precision\@K Scatter Plot**: Scatter plot with Recall\@5 on x-axis and Precision\@5 on y-axis. Each point represents one experiment configuration. Label the top 5 configurations.
* **Chunking Strategy Heatmap**: Heatmap showing average MRR for each combination of chunking strategy and embedding model (rows: chunking configs, columns: embedding models).
* **Retrieval Method Comparison Grouped Bar Chart**: Grouped bar chart comparing BM25, Vector, and Hybrid retrieval across all chunking configurations. Show MRR as the primary metric.
* **Metric Correlation Matrix**: Correlation heatmap of all IR metrics (Recall\@K, Precision\@K, MRR, MAP, NDCG\@K) across all experiments.
* **Response Time vs Quality Trade-off**: Scatter plot with average retrieval time (x-axis) and MRR (y-axis) for each configuration. Annotate the Pareto-optimal configurations.

***

## 📋 Iteration Logs

Maintain a structured log for each major experiment or pipeline change. Use this format:

```
### Iteration N: [Brief title]
- **Date**: YYYY-MM-DD
- **Change**: What you changed from the previous iteration
- **Hypothesis**: Why you expected this change to help
- **Result**: Quantitative outcome (MRR, Recall@K, etc.)
- **Decision**: Keep / revert / modify further
- **Next step**: What to try next based on this result
```

Example entries:

| Iteration | Change                                      | MRR Before | MRR After | Decision                      |
| --------- | ------------------------------------------- | ---------- | --------- | ----------------------------- |
| 1         | Baseline: fixed\_256, embed-3-small, vector | N/A        | 0.963     | Keep as baseline              |
| 2         | Switch to embed-3-large                     | 0.963      | 0.919     | Revert, 3-small is better     |
| 3         | Add sentence-based chunking                 | 0.963      | 0.871     | Keep for comparison, not best |
| 4         | Fix hybrid score normalization              | 0.050      | 0.820     | Keep, still below vector-only |

***

## ✅ Final Checklist

### PDF Parsing and Chunking

* \[ ] PDF dataset downloaded (`any.pdf` or `fy10syb.pdf`) and used consistently
* \[ ] System parses PDFs and extracts text using at least one PDF library (pdfplumber, PyPDF2, or PyMuPDF)
* \[ ] At least 3 chunking strategies implemented (fixed-size, sentence-based, semantic)
* \[ ] At least 4 chunking configurations tested (varying chunk size and overlap)
* \[ ] Chunks include metadata (page number, method, size parameters, unique ID)
* \[ ] Chunk boundaries are reasonable (no mid-word or mid-sentence splits for sentence/semantic)

### Embedding and Vector Storage

* \[ ] At least 2 embedding models compared (text-embedding-3-small, text-embedding-3-large)
* \[ ] Embeddings computed in batches (not one-by-one)
* \[ ] Embeddings cached to disk to avoid re-computation
* \[ ] FAISS vector database stores and retrieves vectors correctly
* \[ ] Vector index supports top-K retrieval with configurable K

### Retrieval Methods

* \[ ] BM25 lexical retrieval implemented and functional
* \[ ] Vector (semantic) retrieval implemented and functional
* \[ ] Hybrid retrieval with proper score normalization implemented
* \[ ] Score normalization tested (BM25 and cosine similarity on same scale)
* \[ ] All retrieval methods return ranked results with scores

### Synthetic QA Generation

* \[ ] QA dataset generated **separately for each chunking configuration**
* \[ ] Structured QA generation using Instructor aligned with chunk IDs
* \[ ] At least 20 QA pairs per chunking configuration
* \[ ] Each QA pair references a valid chunk ID from its specific configuration
* \[ ] QA quality spot-checked (natural phrasing, correct ground truth)

### Evaluation Engine

* \[ ] Recall\@K implemented correctly
* \[ ] Precision\@K implemented correctly
* \[ ] MRR implemented correctly
* \[ ] MAP implemented correctly
* \[ ] NDCG\@K implemented correctly
* \[ ] Metrics validated against manual calculations for a few queries

### Grid Search and Results

* \[ ] At least 12 experiment configurations evaluated in a systematic grid search
* \[ ] Best configuration identified with MRR >= 0.85
* \[ ] Best configuration achieves Recall\@5 >= 0.90
* \[ ] Results displayed in formatted tables (Rich or equivalent)
* \[ ] All experiment results saved to JSON files (validated with Pydantic)
* \[ ] Best configuration summary saved separately

### Reranking

* \[ ] Cohere reranking integrated
* \[ ] Comparison run with and without reranking on best configuration
* \[ ] Delta in MRR, Recall\@K, and MAP measured and documented

### Visualizations

* \[ ] MRR comparison bar chart generated and saved
* \[ ] Recall vs Precision scatter plot generated and saved
* \[ ] Chunking strategy heatmap generated and saved
* \[ ] Retrieval method comparison chart generated and saved
* \[ ] Metric correlation matrix generated and saved
* \[ ] Response time vs quality trade-off chart generated and saved
* \[ ] All charts use Matplotlib, Seaborn, or Plotly

### Iteration Logs

* \[ ] Iteration log maintained with structured entries
* \[ ] Each major change documented with hypothesis, result, and decision
* \[ ] At least 4 iteration entries recorded

### Code Quality and Testing

* \[ ] Code is modular with clear separation: parsing, chunking, embedding, retrieval, evaluation
* \[ ] Configuration management uses Pydantic models
* \[ ] Error handling for API calls (rate limits, timeouts)
* \[ ] Unit tests for metric calculations
* \[ ] Integration test for full pipeline (single configuration end-to-end)

**Remember:** This project is about systematic evaluation, not just building a RAG pipeline. The value is in discovering which configurations work best through rigorous measurement, and being able to explain _why_. Your evaluation methodology is as important as your implementation. Good luck!
