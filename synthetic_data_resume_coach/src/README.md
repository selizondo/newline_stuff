# Resume Coach — Synthetic Data Pipeline (P2)

4-phase pipeline that generates, validates, labels, and corrects synthetic resume–job-description
pairs for training and evaluating resume coaching models.
See [synthetic_data_resume_coach.md](../synthetic_data_resume_coach.md) for the full project spec.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # set LLM_API_KEY, LLM_MODEL, LLM_JUDGE_MODEL
```

## Run

```bash
# Full 4-phase pipeline
python main.py --batch-label my-run

# Individual phases
python main.py --phase 1 --batch-label my-run
python main.py --phase 2 --batch-label my-run
python main.py --phase 3 --batch-label my-run
python main.py --phase 4 --batch-label my-run

# Status across all runs
python main.py stats

# Cross-run comparison charts
python main.py compare
```

## Pipeline phases

| Phase | Module | What it does |
|---|---|---|
| 1 | `phase1_generation.py` | LLM generates resume–job pairs via Instructor |
| 2 | `phase2_validation.py` | Structural validation + heuristic gates |
| 3 | `phase3_labeling.py` | LLM-as-Judge: failure modes + quality dimensions |
| 4 | `phase4_correction.py` | Data-driven prompt correction with iterative loop |

## Module layout

```
src/
├── main.py                # CLI orchestrator — all phases + subcommands
├── config.py              # Settings (env vars / .env)
├── schema.py              # Pydantic schemas
├── generators.py          # Generation helpers
├── pipeline.py            # Phase orchestration logic
├── prompts.py             # YAML prompt loader
├── phase1_generation.py
├── phase2_validation.py
├── phase3_labeling.py
├── phase4_correction.py
├── analysis/              # failure_labeler, failure_modes, heatmap, llm_judge
├── api/                   # FastAPI routes (main.py, routes.py)
├── correction/            # llm_correction.py
├── failure_modes/         # YAML failure mode definitions
├── prompts/               # YAML prompt templates per strategy
├── schemas/
├── utils/
├── validators/
└── tests/
```

## Output

Each run writes to `data/output/<batch-label>/`. Phases are independently re-runnable
with the same `--batch-label`.
