# Home DIY Repair Q&A — Synthetic Data Pipeline

A 7-phase pipeline that generates, validates, evaluates, and iteratively improves
synthetic Q&A pairs for home DIY repair tasks. The pipeline demonstrates the full
lifecycle of LLM-generated training data: from raw generation through structured
validation, judge calibration against a real-world benchmark, LLM-as-Judge quality
scoring, failure analysis, and data-driven iterative prompt correction.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env

# Run the full pipeline (50 samples, zero_shot strategy)
python main.py --batch-label my-run-1

# Run with separate generation and judge models
python main.py --generation-model gpt-4o-mini --judge-model gpt-4o --batch-label my-run-1

# Run only specific phases
python main.py --phase 1-6 --batch-label my-run-1
python main.py --phase 7   --batch-label my-run-1

# Correction with up to 5 iterations
python main.py --phase 7 --max-iterations 5 --batch-label my-run-1

# Status table — phase completion + key metrics across all runs
python main.py stats

# Full JSON report for a specific run
python main.py stats --batch-label my-run-1

# Cross-strategy comparison charts (after running multiple batches)
python main.py compare

# Human/LLM agreement analysis for a run (requires human_labels.json)
python main.py agreement --batch-label my-run-1

# Mock pipeline — no API credentials required, seeds from HF benchmark
python main.py mock
python main.py mock --batch-label baseline-mock --num-samples 50 --seed 42
python main.py mock --skip-human-labels
```

## Configuration

Settings are loaded from environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | *(required)* | API key for the LLM provider |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (works with Ollama) |
| `LLM_MODEL` | `gpt-4o-mini` | Model used for data generation (Phase 1) |
| `LLM_JUDGE_MODEL` | same as `LLM_MODEL` | Model used for LLM-as-Judge evaluation (Phases 3, 4, 5) |
| `LLM_JUDGE_BASE_URL` | same as `LLM_BASE_URL` | Endpoint for the judge model (e.g. local Ollama) |
| `LLM_JUDGE_API_KEY` | same as `LLM_API_KEY` | API key for the judge endpoint |
| `LLM_RATE_LIMIT_DELAY` | `2.0` | Seconds to sleep after each generation LLM call |
| `LLM_JUDGE_RATE_LIMIT_DELAY` | `0.0` | Seconds to sleep after each judge LLM call |

All settings can also be overridden at the CLI level via `--generation-model` and `--judge-model`.

## Prompt Strategies

Phase 1 generation supports four strategies, selected via `--prompt-strategy`:

| Strategy | Description |
|---|---|
| `zero_shot` | Minimal instructions, no examples (default) |
| `few_shot` | Detailed instructions with one worked example per category |
| `chain_of_thought` | Explicit reasoning steps before generating output |
| `human_feedback` | Corrected prompts targeting observed failure modes; used internally by Phase 7 |

Templates live in `prompts/<strategy>/` — one YAML per repair category. Add a new
strategy by creating a subdirectory with 5 category YAMLs; no code changes needed.

## Output

Each run writes to its own isolated subdirectory: `output/<batch-label>/`. Re-running
with the same label overwrites that run; use a different label to keep runs side by side.

---

## Pipeline Phases

### Phase 1 — Generation

**What it does:** Calls the LLM using Instructor to generate structured `QAPair` objects
for each of the five repair categories (`appliance_repair`, `plumbing_repair`,
`electrical_repair`, `hvac_maintenance`, `general_home_repair`). Samples are distributed
across categories using stratified sampling so every category is represented even at
small batch sizes.

**Rationale:** Instructor wraps the OpenAI API with Pydantic-backed retries, so the
model is coerced into emitting valid JSON that matches the `QAPair` schema before the
result is accepted. This separates *schema compliance* (does the output parse correctly?)
from *semantic quality* (is the content actually good?), which are evaluated independently
in later phases. Generation failures are recorded rather than silently dropped, giving
visibility into how often the model struggles with the schema.

---

### Phase 2 — Structural Validation

**What it does:** Re-validates each raw generation result against the `QAPair` Pydantic
schema and applies additional checks (minimum step count, non-empty fields, tool list
presence). Produces a `ValidationSummary` and a clean list of `ValidatedResult` objects
that downstream phases can trust.

**Rationale:** Instructor's internal retries catch most schema errors during generation,
but a second validation pass is cheap insurance and provides a clear audit record of
which samples passed structural requirements. It also decouples the validation contract
from the generation implementation — if the generation strategy changes, Phase 2 remains
the authoritative gate.

---

### Phase 3 — Benchmark Calibration (Judge Verification)

**What it does:** Loads 50 samples from the `dipenbhuva/home-diy-repair-qa` HuggingFace
dataset, maps them to the `QAPair` schema, and runs the Phase 5 quality judge on them.
Reports per-dimension pass rates and a calibration pass/fail (≥ 95% required). Saves
`benchmark_eval.csv` and `benchmark_report.json`.

**Rationale:** Running calibration *before* Phases 4 and 5 means any systematic judge
miscalibration is caught before it silently distorts all downstream quality metrics and
correction targets. A judge that fails calibration is untrustworthy; continuing with it
would invalidate the entire feedback loop. Phase 6 (analysis) auto-loads `benchmark_eval.csv`
for an apples-to-apples quality gap comparison.

---

### Phase 4 — Failure Labeling (LLM-as-Judge)

**What it does:** A separate LLM judge evaluates each validated Q&A pair against six
binary failure modes: `incomplete_answer`, `safety_violations`, `unrealistic_tools`,
`overcomplicated_solution`, `missing_context`, and `poor_quality_tips`. Each mode is
defined in its own YAML file under `failure_modes/`.

**Rationale:** Failure labeling answers the question *"what is wrong with this data?"*
rather than just *"is this data good?"*. Binary per-mode labels are easier for a judge
to produce reliably than a holistic score, and they give actionable signal: if
`poor_quality_tips` is the dominant failure mode, the fix is targeted prompt engineering
for that specific problem rather than a wholesale rewrite. Storing mode labels per sample
also enables correlation analysis (Phase 6) to surface which failure modes tend to
co-occur.

---

### Phase 5 — Quality Evaluation (LLM-as-Judge)

**What it does:** A second LLM judge scores each sample across nine quality dimensions:
`answer_coherence`, `answer_completeness`, `step_actionability`, `tool_realism`,
`safety_specificity`, `tip_usefulness`, `problem_answer_alignment`, `appropriate_scope`,
and `category_accuracy`. Each dimension has a pass/fail threshold defined in its YAML
config under `quality_dimensions/`. A sample passes overall only if it passes all nine.

**Rationale:** Where Phase 4 looks for specific defects, Phase 5 evaluates positive
quality attributes. Using a dedicated judge model (configurable via `LLM_JUDGE_MODEL`)
separates the roles: a cheaper, faster model can generate data while a more capable model
acts as the quality gate. Defining dimensions and thresholds in YAML makes it easy to
adjust criteria or add new dimensions without touching Python code.

---

### Phase 6 — Analysis & Visualizations

**What it does:** Joins the Phase 4 and Phase 5 outputs and produces six charts plus a
structured `AnalysisSummary` JSON report. Auto-loads `benchmark_eval.csv` from Phase 3
to compute an apples-to-apples quality gap (both sides measured on `overall_quality_pass`):

1. Failure mode heatmap (samples × modes)
2. Failure rates by repair category
3. Before vs. after failure mode trends (populated after Phase 7)
4. Quality dimension pass rates vs. thresholds
5. Generated vs. benchmark quality comparison
6. Failure mode correlation heatmap

**Rationale:** Raw numbers from the judge phases are hard to act on without context.
Visualizations make it immediately obvious which categories are most problematic, which
failure modes cluster together, and which quality dimensions are furthest below threshold.
The benchmark gap uses the same `overall_quality_pass` metric on both sides so the
comparison is methodologically valid (not a conjunctive pass rate vs. an arithmetic mean).
Phase 6 is re-run after Phase 7 so the same charts reflect the latest state.

---

### Phase 7 — Prompt Correction & Re-evaluation

**What it does:** Re-runs Phases 1, 2, 4, 5 using the `human_feedback` prompt strategy,
which contains revised prompts targeting the failure modes identified in Phase 4. Key
improvements over a static correction approach:

- **Data-driven:** failure rates from Phase 4 are injected into generation prompts so
  the corrected run targets the specific modes that actually failed, not a generic
  "write better" instruction.
- **Iterative:** re-runs up to `--max-iterations` (default 3) until all three absolute
  targets are simultaneously met: failure ≤ 15%, quality pass ≥ 80%, improvement ≥ 80%.
- **Diversity check:** Jaccard similarity guard flags if corrected answers are near-copies
  of the baseline.

Produces a `ComparisonReport` with before/after failure rates, quality pass rates,
improvement percentage, iterations run, and diversity score.

**Rationale:** The failure labels from Phase 4 tell you *what* is failing; Phase 7 closes
the loop by testing whether targeted prompt changes actually fix those failures. Running
the corrected prompts through the full evaluation stack (not just eyeballing outputs)
gives an objective, quantified answer. Using the same judge infrastructure as the
baseline ensures the comparison is apples-to-apples.

---

## Cost Analysis

The pipeline makes two distinct types of LLM calls with different cost profiles:

- **Generation** (Phase 1, Phase 7): long outputs (~500 tokens), quality-sensitive — benefits from a capable model
- **Judging** (Phases 3, 4, 5): single-token outputs (`0` or `1`), high volume — benefits from a fast, cheap model

### Per-run cost estimate (50 samples)

Assumptions: ~300 input / ~500 output tokens per generation call; ~200 input / ~1 output token per judge call (8 dimensions × 50 samples = 400 judge calls, plus 6 failure modes × 50 = 300, plus 50 benchmark calls).

| Role | Model | Provider | Input | Output | Est. cost / run |
|---|---|---|---|---|---|
| Generation | `llama-3.1-8b-instant` | Groq | $0.05/1M | $0.08/1M | ~$0.003 |
| Generation | `llama-3.3-70b-versatile` | Groq | $0.59/1M | $0.79/1M | ~$0.030 |
| Generation | `gpt-4o-mini` | OpenAI | $0.15/1M | $0.60/1M | ~$0.008 |
| Judge | `qwen2.5:3b` (Ollama local) | — | free | free | $0.000 |
| Judge | `llama-3.1-8b-instant` | Groq | $0.05/1M | $0.08/1M | ~$0.001 |

### Recommended setup

```
# .env
LLM_MODEL=llama-3.1-8b-instant       # Groq — fast, cheap, sufficient for structured generation
LLM_JUDGE_MODEL=qwen2.5:3b           # Ollama local — free, strong instruction following for 0/1 output
LLM_JUDGE_BASE_URL=http://localhost:11434/v1
LLM_JUDGE_API_KEY=ollama
LLM_JUDGE_RATE_LIMIT_DELAY=0.0
```

**Estimated total cost per full 7-phase run: ~$0.003** (generation only; judging is free locally).

### When to upgrade the generation model

Switch to `llama-3.3-70b-versatile` or `gpt-4o-mini` if Phase 2 structural validation
drops below 90% or Phase 5 quality pass rates are consistently low — those are signals
that the 8B model is struggling with schema compliance or content depth, and the ~10×
cost increase is justified.

---

## Quality Targets

| Metric | Target |
|---|---|
| Minimum dataset size per run | ≥ 50 Q&A pairs |
| Benchmark calibration pass rate (Phase 3) | ≥ 95% on benchmark items |
| Baseline failure rate (Phase 4) | ≥ 15% (establishes a measurable problem to correct) |
| Overall quality pass rate — all 8 dimensions (Phase 5) | ≥ 80% |
| Post-correction failure rate (Phase 7) | ≤ 15% |
| Post-correction quality pass rate (Phase 7) | ≥ 80% |
| Post-correction failure rate reduction (Phase 7) | ≥ 80% vs baseline |

---

## Output Files

Each run writes to `output/<batch-label>/`. Phase 7 writes its corrected-run output to a `corrected/` subdirectory.

| File | Written by | Contents |
|---|---|---|
| `generation_results.json` | Phase 1 | All generated items including failures |
| `structurally_valid_qa_pairs.json` | Phase 2 | Pydantic-valid items only |
| `validation_summary.json` | Phase 2 | Total/valid/invalid counts and common errors |
| `benchmark_eval.csv` | Phase 3 | Per-item quality scores on benchmark set |
| `benchmark_report.json` | Phase 3 | Calibration pass/fail and per-dimension rates |
| `gate_report.json` | Phase 2 | Gate pass/fail counts, category distribution, dedup count |
| `failure_labeled_data.{csv,json}` | Phase 4 | 6 binary failure flags per item |
| `quality_eval_data.{csv,json}` | Phase 5 | 9 quality dimension scores per item |
| `human_labels.json` | `human_labeler.py` / mock | 6 human-rated dimensions per item (used by `agreement` subcommand) |
| `analysis_report.json` | Phase 6 | Aggregated rates, thresholds met, benchmark gap, problematic trace_ids |
| `corrected/before_after_comparison.json` | Phase 7 | Improvement %, iterations run, diversity score, per-mode deltas |

---

## Failure Modes Reference (Phase 4)

Six binary flags assigned per sample by the LLM judge:

| Mode | Description |
|---|---|
| `incomplete_answer` | Answer is missing key repair steps |
| `safety_violations` | Guidance omits or contradicts safety requirements |
| `unrealistic_tools` | Recommends tools a homeowner is unlikely to have |
| `overcomplicated_solution` | Solution is unnecessarily complex for the problem |
| `missing_context` | Answer lacks situational context needed to perform the repair |
| `poor_quality_tips` | Tips are vague, obvious, or not specific to the task |

Criteria are defined in `failure_modes/<name>.yaml`. Add a new file to introduce a new mode — no code changes needed.

---

## Quality Dimensions Reference (Phase 5)

Nine pass/fail scores assigned per sample by the LLM judge:

| Dimension | Threshold | Description |
|---|---|---|
| `answer_coherence` | 90% | Narrative flow, logical progression from problem to resolution |
| `answer_completeness` | 85% | Answer covers all necessary steps without leaving critical gaps |
| `step_actionability` | 85% | Every step has a specific action verb and enough detail to execute |
| `tool_realism` | 95% | All tools cost < $50 and are available at hardware stores |
| `safety_specificity` | 90% | Names a specific hazard and gives a specific protective action |
| `tip_usefulness` | 85% | Tips are non-obvious, task-specific, and provide concrete value |
| `problem_answer_alignment` | 95% | Answer directly addresses the stated problem without drift |
| `appropriate_scope` | 95% | Complexity matches what an average homeowner can handle |
| `category_accuracy` | 98% | Q&A pair clearly belongs to the stated repair category |

Criteria and thresholds are defined in `quality_dimensions/<name>.yaml`. Edit a file to adjust a threshold or criteria without touching Python code.

---

## Project Structure

```
src/
├── main.py                   # CLI orchestrator — all 7 phases + compare/agreement/mock subcommands
├── config.py                 # Settings (env vars / .env)
├── schema.py                 # Pydantic schemas shared across all phases
├── llm_client.py             # Cached OpenAI-compatible client with backoff
├── prompts.py                # Prompt template loader (reads prompts/<strategy>/)
│
├── phase1_generation.py      # LLM generation via Instructor
├── phase2_validation.py      # Structural validation
├── phase3_benchmark.py       # Judge calibration against real-world benchmark
├── phase4_failure_labeling.py # LLM-as-Judge: 6 failure modes
├── phase5_quality_eval.py    # LLM-as-Judge: 9 quality dimensions
├── phase6_analysis.py        # Analysis, visualizations, reports + benchmark gap
├── phase7_correction.py      # Data-driven prompt correction with iterative loop
├── mock_seeder.py            # Mock pipeline seeder — phases 1–6 with no API calls
├── agreement.py              # Phase A: human/LLM agreement analysis
├── human_labeler.py          # Interactive CLI for collecting human labels
│
├── prompts/                  # Prompt templates organised by strategy
│   ├── zero_shot/            # 5 category YAMLs
│   ├── few_shot/
│   ├── chain_of_thought/
│   └── human_feedback/       # Corrected prompts targeting observed failures
│
├── failure_modes/            # One YAML per failure mode (Phase 4)
│   ├── incomplete_answer.yaml
│   ├── safety_violations.yaml
│   └── ...
│
├── quality_dimensions/       # One YAML per quality dimension (Phase 5)
│   ├── answer_coherence.yaml
│   ├── step_actionability.yaml
│   └── ...
│
└── output/                   # One subdirectory per run (gitignored)
    └── <batch-label>/
```
