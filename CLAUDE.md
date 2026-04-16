# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

```
projects/
├── mini-project-1/
│   ├── mini-project-1.md   # Full project specification (read this first)
│   └── solution/           # Complete 7-phase pipeline implementation
└── debug_project_01.ipynb  # Scratch notebook
```

## mini-project-1/solution — setup & commands

All commands run from `mini-project-1/solution/`.

```bash
# First-time setup
cp .env.example .env        # fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
pip install -r requirements3.txt

# Run the full 7-phase pipeline (50 samples)
python3 main.py

# Run a specific phase range (useful for iteration)
python3 main.py --phase 1-5          # stop after analysis
python3 main.py --phase 6            # prompt correction only (needs phases 1-5 output)
python3 main.py --phase 7            # benchmark only (needs phase 4 output)

# Common options
python3 main.py --samples 10 --phase 1-2   # quick smoke test
python3 main.py --corrected                # use corrected prompts in phase 1
python3 main.py stats                      # print summaries from existing output files
```

## LLM provider configuration

The client (`llm_client.py`) uses the OpenAI Python SDK against any OpenAI-compatible endpoint. Provider is set via `.env`:

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | `LLM_MODEL` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `sk-...` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | `ollama` | `llama3.2` |

## Architecture

Data flows linearly through 7 phases. Each phase reads its input from `output/` (JSON/CSV) and writes its output there, so phases can be re-run independently.

```
prompts.py (BASELINE / CORRECTED templates)
    │
    ▼
phase1_generation.py   → output/generation_results.json
    │
    ▼
phase2_validation.py   → output/structurally_valid_qa_pairs.json
    │
    ▼
phase3_failure_labeling.py  → output/failure_labeled_data.{csv,json}
    │                           6 binary failure modes per item
    ▼
phase4_quality_eval.py      → output/quality_eval_data.{csv,json}
    │                           8 quality dimensions per item (pass/fail)
    ▼
phase5_analysis.py     → output/*.png (6 charts) + output/analysis_report.json
    │
    ├── phase6_correction.py  → re-runs phases 1-4 with CORRECTED_TEMPLATES
    │                           output/corrected/ + before_after_comparison.json
    │
    └── phase7_benchmark.py   → loads dipenbhuva/home-diy-repair-qa (HuggingFace)
                                evaluates 50 benchmark items, outputs benchmark_report.json
```

**Key modules:**

- `models.py` — all Pydantic v2 schemas. Module-level constants `FAILURE_MODE_FIELDS` (list of 6 mode names) and `QUALITY_DIMENSION_FIELDS` / `QUALITY_THRESHOLDS` are used across phases.
- `prompts.py` — two template sets: `BASELINE_TEMPLATES` and `CORRECTED_TEMPLATES` (5 repair categories each). Edit `CORRECTED_TEMPLATES` when iterating on prompt quality.
- `config.py` — single `get_settings()` call returns a `Settings` dataclass from env. `llm_client.py` caches one `OpenAI` client globally.
- `main.py` — thin orchestrator; each phase is imported and called in sequence. Phase skipping is handled by `--phase start-end` argument.

**Corrected-run output goes to `output/corrected/`** — a subdirectory mirroring the same file structure as the baseline `output/`. Phase 5 visualizations are updated in-place in `output/` to include both baseline and corrected data.

## Quality targets (from spec)

- Baseline failure rate must be ≥ 15% (establishes a measurable problem)
- Post-correction failure rate must drop by > 80% vs baseline
- Overall quality pass rate (all 8 dimensions): ≥ 80%
- Benchmark judge calibration: ≥ 95% pass rate on benchmark items (Phase 7)
- Minimum dataset: ≥ 50 Q&A pairs per run

## Output files reference

| File | Written by | Contents |
|---|---|---|
| `generation_results.json` | Phase 1 | All generated items incl. failures |
| `structurally_valid_qa_pairs.json` | Phase 2 | Pydantic-valid items only |
| `failure_labeled_data.{csv,json}` | Phase 3 | 6 binary failure flags per item |
| `quality_eval_data.{csv,json}` | Phase 4 | 8 quality dimension scores per item |
| `analysis_report.json` | Phase 5 | Aggregated rates, thresholds met, problematic trace_ids |
| `corrected/before_after_comparison.json` | Phase 6 | Improvement % and per-mode deltas |
| `benchmark_report.json` | Phase 7 | Calibration pass/fail, generated-vs-benchmark gaps |
