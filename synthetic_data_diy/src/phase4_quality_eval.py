"""
Phase 4: Quality Evaluation (LLM-as-Judge)
Scores each Q&A pair across 8 quality dimensions defined in YAML config files.

Quality dimension configs live in quality_dimensions/<name>.yaml (one file per dimension).
Add a new file to introduce a new dimension — no Python changes needed.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from config import get_settings
from llm_client import chat_complete
from schema import QAPair, QualityEvalResult, ValidatedResult

# Default config directory relative to this file
DEFAULT_QUALITY_DIMS_DIR = Path(__file__).parent / "quality_dimensions"


@dataclass
class QualityDimension:
    name: str
    label: str        # human-readable label from spec
    threshold: float  # required pass rate
    prompt_template: str


def load_quality_dimensions(config_dir: Path = DEFAULT_QUALITY_DIMS_DIR) -> list[QualityDimension]:
    """Load all quality dimension definitions from YAML files in config_dir.

    Each file must have: name, label, threshold, prompt_template.
    Files are loaded in alphabetical order for determinism; the order only
    affects iteration (evaluation order per item), not the final scores.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"Quality dimensions directory not found: {config_dir}")

    dims: list[QualityDimension] = []
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {config_dir}")

    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)
        missing = [k for k in ("name", "label", "threshold", "prompt_template") if k not in data]
        if missing:
            raise ValueError(f"{path.name} is missing required keys: {missing}")
        dims.append(QualityDimension(
            name=data["name"],
            label=data["label"],
            threshold=float(data["threshold"]),
            prompt_template=data["prompt_template"],
        ))

    return dims


# Module-level constant — loaded once; mirrors the pattern in phase3_failure_labeling.py
QUALITY_DIMENSIONS: list[QualityDimension] = load_quality_dimensions()


class QualityEvaluator:
    def __init__(self, judge_model: str):
        self.model = judge_model

    def _judge_dimension(self, dim: QualityDimension, qa: QAPair, category: str) -> int:
        prompt = dim.prompt_template.format(
            question=qa.question,
            answer=qa.answer,
            equipment_problem=qa.equipment_problem,
            tools=", ".join(qa.tools_required),
            steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(qa.steps)),
            safety_info=qa.safety_info,
            tips="\n".join(f"- {t}" for t in qa.tips),
            category=category,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a quality evaluator for DIY repair content. Respond with exactly one digit: 0 or 1.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            raw = chat_complete(messages, model=self.model, temperature=0.1, max_tokens=10)
            digit = raw.strip()[0]
            return int(digit) if digit in ("0", "1") else 0
        except Exception:
            return 0  # default to fail on error

    def evaluate(self, result: ValidatedResult) -> QualityEvalResult:
        qa = result.qa_pair
        scores: dict[str, int] = {}
        for dim in QUALITY_DIMENSIONS:
            scores[dim.name] = self._judge_dimension(dim, qa, result.category)
            time.sleep(0.15)

        overall_pass = 1 if all(v == 1 for v in scores.values()) else 0
        return QualityEvalResult(
            trace_id=result.trace_id,
            category=result.category,
            overall_quality_pass=overall_pass,
            **scores,
        )


def run_quality_eval_phase(
    valid_results: list[ValidatedResult],
    judge_model: str,
    output_dir: Path,
) -> pd.DataFrame:
    evaluator = QualityEvaluator(judge_model=judge_model)
    eval_results: list[QualityEvalResult] = []

    for i, result in enumerate(valid_results):
        print(f"  [{i+1}/{len(valid_results)}] Evaluating {result.trace_id[:8]}... ", end="", flush=True)
        eval_result = evaluator.evaluate(result)
        eval_results.append(eval_result)
        dims_failed = [d.name for d in QUALITY_DIMENSIONS if getattr(eval_result, d.name) == 0]
        print("FAIL: " + ", ".join(dims_failed) if dims_failed else "PASS (all 8)")
        time.sleep(get_settings().rate_limit_delay)

    rows = [r.model_dump() for r in eval_results]
    df = pd.DataFrame(rows)

    pass_rate = df["overall_quality_pass"].mean()
    print(f"\nQuality evaluation complete: {pass_rate*100:.1f}% overall quality pass rate")

    print("\nPer-dimension pass rates:")
    for dim in QUALITY_DIMENSIONS:
        rate = df[dim.name].mean()
        met = "✓" if rate >= dim.threshold else "✗"
        print(f"  {met} {dim.label}: {rate*100:.1f}% (threshold: {dim.threshold*100:.0f}%)")

    df.to_csv(output_dir / "quality_eval_data.csv", index=False)
    df.to_json(output_dir / "quality_eval_data.json", orient="records", indent=2)
    print(f"\nSaved → {output_dir / 'quality_eval_data.csv'}")
    return df
