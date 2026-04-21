"""
Phase 4: Failure Labeling (LLM-as-Judge)
Evaluates each Q&A pair against binary failure modes loaded from YAML config files.

Runs after Phase 3 benchmark calibration confirms the judge is trustworthy.
Failure mode configs live in failure_modes/<name>.yaml (one file per mode).
Add a new file to introduce a new failure mode — no Python changes needed.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from llm_client import judge_binary
from schema import FAILURE_MODE_FIELDS, FailureLabelResult, ValidatedResult, qa_format_kwargs

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
    def __init__(self, judge_model: str, failure_modes: list[FailureMode], additional_context: str = ""):
        self.model = judge_model
        self.failure_modes = failure_modes
        self.additional_context = additional_context

    def _build_prompt(self, mode: FailureMode, qa) -> str:
        prompt = mode.prompt_template.format(**qa_format_kwargs(qa))
        if self.additional_context:
            prompt = f"{self.additional_context}\n\n{prompt}"
        return prompt

    def evaluate(self, result: ValidatedResult) -> FailureLabelResult:
        qa = result.qa_pair
        scores: dict[str, int] = {
            mode.name: judge_binary(self._build_prompt(mode, qa), self.model, default_on_error=1)
            for mode in self.failure_modes
        }
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
    additional_context: str = "",
) -> pd.DataFrame:
    failure_modes = load_failure_modes(config_dir)
    print(f"Loaded {len(failure_modes)} failure modes from {config_dir}")

    labeler = FailureLabeler(judge_model=judge_model, failure_modes=failure_modes, additional_context=additional_context)
    label_results: list[FailureLabelResult] = []

    for i, result in enumerate(valid_results):
        print(f"  [{i+1}/{len(valid_results)}] Labeling {result.trace_id[:8]}... ", end="", flush=True)
        label = labeler.evaluate(result)
        label_results.append(label)
        fail_names = [m for m in FAILURE_MODE_FIELDS if getattr(label, m) == 1]
        print("FAIL: " + ", ".join(fail_names) if fail_names else "PASS")

    rows = [r.model_dump() for r in label_results]
    df = pd.DataFrame(rows)

    overall_rate = df["overall_failure"].mean()
    print(f"\nFailure labeling complete: {overall_rate*100:.1f}% overall failure rate")

    df.to_csv(output_dir / "failure_labeled_data.csv", index=False)
    df.to_json(output_dir / "failure_labeled_data.json", orient="records", indent=2)
    print(f"Saved → {output_dir / 'failure_labeled_data.csv'}")
    return df
