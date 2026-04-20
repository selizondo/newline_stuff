"""
Phase 3: Failure Labeling (LLM-as-Judge)
Evaluates each Q&A pair against binary failure modes loaded from YAML config files.

Failure mode configs live in failure_modes/<name>.yaml (one file per mode).
Add a new file to introduce a new failure mode — no Python changes needed.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from config import get_settings
from llm_client import chat_complete
from schema import FAILURE_MODE_FIELDS, FailureLabelResult, ValidatedResult

# Default config directory relative to this file
DEFAULT_FAILURE_MODES_DIR = Path(__file__).parent / "failure_modes"


@dataclass
class FailureMode:
    name: str
    description: str
    prompt_template: str  # placeholders: {question}, {answer}, {steps}, {safety_info}, {tips}, {tools}, {equipment_problem}


def load_failure_modes(config_dir: Path = DEFAULT_FAILURE_MODES_DIR) -> list[FailureMode]:
    """Load all failure mode definitions from YAML files in config_dir.

    Each file must have: name, description, prompt_template.
    Files are loaded in alphabetical order for determinism; the order only
    affects iteration (evaluation order per item), not the final scores.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"Failure modes directory not found: {config_dir}")

    modes: list[FailureMode] = []
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        raise FileNotFoundError(f"No .yaml files found in {config_dir}")

    for path in yaml_files:
        with path.open() as f:
            data = yaml.safe_load(f)
        missing = [k for k in ("name", "description", "prompt_template") if k not in data]
        if missing:
            raise ValueError(f"{path.name} is missing required keys: {missing}")
        modes.append(FailureMode(
            name=data["name"],
            description=data["description"],
            prompt_template=data["prompt_template"],
        ))

    return modes


class FailureLabeler:
    def __init__(self, judge_model: str, failure_modes: list[FailureMode]):
        self.model = judge_model
        self.failure_modes = failure_modes
        self.settings = get_settings()

    def _format_prompt(self, mode: FailureMode, qa) -> str:
        return mode.prompt_template.format(
            question=qa.question,
            answer=qa.answer,
            equipment_problem=qa.equipment_problem,
            tools=", ".join(qa.tools_required),
            steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(qa.steps)),
            safety_info=qa.safety_info,
            tips="\n".join(f"- {t}" for t in qa.tips),
        )

    def _judge_one_mode(self, mode: FailureMode, qa) -> int:
        prompt = self._format_prompt(mode, qa)
        messages = [
            {
                "role": "system",
                "content": "You are a quality evaluator for DIY repair content. Respond with exactly one digit: 0 or 1.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            raw = chat_complete(messages, model=self.model, temperature=0.1, max_tokens=10, use_judge_client=True)
            digit = raw.strip()[0]
            return int(digit) if digit in ("0", "1") else 1
        except Exception:
            return 1  # default to failure on error

    def evaluate(self, result: ValidatedResult) -> FailureLabelResult:
        qa = result.qa_pair
        scores: dict[str, int] = {}
        for mode in self.failure_modes:
            scores[mode.name] = self._judge_one_mode(mode, qa)
            time.sleep(0.2)

        failure_count = sum(scores.values())
        return FailureLabelResult(
            trace_id=result.trace_id,
            category=result.category,
            overall_failure=1 if failure_count > 0 else 0,
            failure_count=failure_count,
            **scores,
        )


def run_failure_labeling_phase(
    valid_results: list[ValidatedResult],
    judge_model: str,
    output_dir: Path,
    config_dir: Path = DEFAULT_FAILURE_MODES_DIR,
) -> pd.DataFrame:
    failure_modes = load_failure_modes(config_dir)
    print(f"Loaded {len(failure_modes)} failure modes from {config_dir}")

    labeler = FailureLabeler(judge_model=judge_model, failure_modes=failure_modes)
    label_results: list[FailureLabelResult] = []

    for i, result in enumerate(valid_results):
        print(f"  [{i+1}/{len(valid_results)}] Labeling {result.trace_id[:8]}... ", end="", flush=True)
        label = labeler.evaluate(result)
        label_results.append(label)
        fail_names = [m for m in FAILURE_MODE_FIELDS if getattr(label, m) == 1]
        print("FAIL: " + ", ".join(fail_names) if fail_names else "PASS")
        time.sleep(get_settings().rate_limit_delay)

    rows = [r.model_dump() for r in label_results]
    df = pd.DataFrame(rows)

    overall_rate = df["overall_failure"].mean()
    print(f"\nFailure labeling complete: {overall_rate*100:.1f}% overall failure rate")

    df.to_csv(output_dir / "failure_labeled_data.csv", index=False)
    df.to_json(output_dir / "failure_labeled_data.json", orient="records", indent=2)
    print(f"Saved → {output_dir / 'failure_labeled_data.csv'}")
    return df
