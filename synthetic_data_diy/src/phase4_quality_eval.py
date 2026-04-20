"""
Phase 4: Quality Evaluation (LLM-as-Judge)
Scores each Q&A pair across 8 quality dimensions defined in the project spec.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import get_settings
from llm_client import chat_complete
from schema import QAPair, QualityEvalResult, ValidatedResult


@dataclass
class QualityDimension:
    name: str
    label: str       # human-readable label from spec
    threshold: float  # required pass rate
    prompt_template: str


QUALITY_DIMENSIONS: list[QualityDimension] = [
    QualityDimension(
        name="answer_coherence",
        label="Q1: Answer Coherence",
        threshold=0.90,
        prompt_template="""Evaluate Q1 — Answer Coherence.

Question: {question}
Answer: {answer}

The answer PASSES if it:
- Reads as a coherent narrative, not a disjointed list of disconnected facts
- Has logical flow from problem identification through resolution
- Uses transitional language that connects ideas

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="step_actionability",
        label="Q2: Step Actionability",
        threshold=0.85,
        prompt_template="""Evaluate Q2 — Step Actionability.

Question: {question}
Steps: {steps}

Steps PASS if every step:
- Contains a specific action verb (tighten, remove, apply, measure, etc.)
- Avoids vague language ("check", "look at", "make sure it works")
- Gives enough detail to actually perform the action

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="tool_realism",
        label="Q3: Tool Realism",
        threshold=0.95,
        prompt_template="""Evaluate Q3 — Tool Realism.

Tools Required: {tools}

Tools PASS if ALL of them:
- Cost less than $50 each
- Are available at hardware stores (Home Depot, Lowe's, Ace Hardware)
- Do not require professional licensing to purchase or use

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="safety_specificity",
        label="Q4: Safety Specificity",
        threshold=0.90,
        prompt_template="""Evaluate Q4 — Safety Specificity.

Question: {question}
Safety Info: {safety_info}

Safety info PASSES if it:
- Names a specific hazard (e.g., "120V live circuit", "pressurized steam", "sharp metal edges")
- Gives a specific protective action (e.g., "turn off circuit breaker and verify with non-contact tester")
- Is at least 80 characters long
- Is relevant to the specific repair task

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="tip_usefulness",
        label="Q5: Tip Usefulness",
        threshold=0.85,
        prompt_template="""Evaluate Q5 — Tip Usefulness.

Question: {question}
Tips: {tips}

The tip PASSES if it:
- Is non-obvious (a beginner would not think of it)
- Is specific to this exact repair task
- Provides concrete, actionable value (saves time, prevents damage, or avoids a common mistake)
- Is NOT generic advice like "wear gloves" or "be careful" or "read the manual"

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="problem_answer_alignment",
        label="Q6: Problem-Answer Alignment",
        threshold=0.95,
        prompt_template="""Evaluate Q6 — Problem-Answer Alignment.

Question: {question}
Equipment Problem: {equipment_problem}
Answer: {answer}

The answer PASSES if it:
- Directly addresses the specific problem stated in the question
- Does not answer a different (related but distinct) question
- Does not omit the core repair described in the question

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="appropriate_scope",
        label="Q7: Appropriate Scope",
        threshold=0.95,
        prompt_template="""Evaluate Q7 — Appropriate Scope.

Question: {question}
Answer: {answer}
Steps: {steps}

The scope is APPROPRIATE if:
- The repair complexity matches what an average homeowner with basic tools can handle
- It does not recommend tearing apart an entire system when a targeted fix would work
- It does not recommend replacing the whole appliance/system for a repairable problem
- The number of steps is proportional to the complexity of the task

Respond with exactly one digit: 1 if PASS, 0 if FAIL.""",
    ),
    QualityDimension(
        name="category_accuracy",
        label="Q8: Category Accuracy",
        threshold=0.98,
        prompt_template="""Evaluate Q8 — Category Accuracy.

Question: {question}
Category: {category}

Valid categories: appliance_repair, plumbing_repair, electrical_repair, hvac_maintenance, general_home_repair

The category PASSES if the question and answer are clearly about the stated repair domain.

Respond with exactly one digit: 1 if PASS (correct category), 0 if FAIL (wrong category).""",
    ),
]


class QualityEvaluator:
    def __init__(self, model: str):
        self.model = model

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
    model: str,
    output_dir: Path,
) -> pd.DataFrame:
    evaluator = QualityEvaluator(model=model)
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

    # Per-dimension pass rates vs thresholds
    dim_map = {d.name: d for d in QUALITY_DIMENSIONS}
    print("\nPer-dimension pass rates:")
    for dim in QUALITY_DIMENSIONS:
        rate = df[dim.name].mean()
        met = "✓" if rate >= dim.threshold else "✗"
        print(f"  {met} {dim.label}: {rate*100:.1f}% (threshold: {dim.threshold*100:.0f}%)")

    df.to_csv(output_dir / "quality_eval_data.csv", index=False)
    df.to_json(output_dir / "quality_eval_data.json", orient="records", indent=2)
    print(f"\nSaved → {output_dir / 'quality_eval_data.csv'}")
    return df
