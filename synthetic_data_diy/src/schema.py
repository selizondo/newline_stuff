"""
Pydantic v2 data models for the DIY Repair Q&A pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core Q&A schema
# ---------------------------------------------------------------------------

def strip_respond_line(text: str) -> str:
    """Remove trailing 'Respond with exactly one digit...' instruction lines from a prompt."""
    lines = text.rstrip("\n").splitlines()
    while lines and lines[-1].strip().startswith("Respond with"):
        lines.pop()
    return "\n".join(lines).rstrip()


def qa_format_kwargs(qa: "QAPair", category: str = "") -> dict:
    """Return formatted QAPair fields for use in prompt .format() calls.

    Both failure-labeling and quality-eval prompts share the same 7-field
    interpolation; quality-eval additionally passes {category}.
    """
    kwargs = dict(
        question=qa.question,
        answer=qa.answer,
        equipment_problem=qa.equipment_problem,
        tools=", ".join(qa.tools_required),
        steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(qa.steps)),
        safety_info=qa.safety_info,
        tips="\n".join(f"- {t}" for t in qa.tips),
    )
    if category:
        kwargs["category"] = category
    return kwargs


class QAPair(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    answer: str = Field(..., min_length=20, max_length=2000)
    equipment_problem: str = Field(..., min_length=5, max_length=200)
    tools_required: list[str] = Field(..., min_length=1)
    steps: list[str] = Field(..., min_length=3)
    safety_info: str = Field(..., min_length=10, max_length=500)
    tips: list[str] = Field(..., min_length=1)

    @field_validator("question", "answer", "equipment_problem", "safety_info", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("tools_required", "steps", "tips", mode="before")
    @classmethod
    def strip_list_items(cls, v: list) -> list:
        return [item.strip() for item in v if isinstance(item, str) and item.strip()]

    @model_validator(mode="after")
    def check_non_empty(self) -> QAPair:
        for field in ("question", "answer", "equipment_problem", "safety_info"):
            if not getattr(self, field):
                raise ValueError(f"{field} must not be blank after stripping")
        if not self.tips:
            raise ValueError("tips must not be empty")
        return self


# ---------------------------------------------------------------------------
# Generation result (Phase 1 output — JSON parsing only, no schema validation)
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    trace_id: str                          # per-sample ID; used to join this record across phases (Phases 3, 4, 5)
    category: str
    batch_id: str = ""                     # per-run ID; groups all records from the same pipeline run — batch_id is shared, trace_id is unique
    batch_label: str = ""                  # human-readable run label (e.g. "zero-shot-run1")
    prompt_strategy: str = ""              # zero_shot | few_shot | chain_of_thought
    raw_response: str = ""
    raw_dict: Optional[dict] = None        # parsed JSON from LLM; None if parsing failed
    parse_error: Optional[str] = None      # set when the LLM response could not be parsed as JSON
    validation_errors: Optional[list[dict]] = None   # Pydantic field errors from InstructorRetryException
    validation_attempts: Optional[int] = None        # how many instructor retries were burned
    generation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Validated result (Phase 2 output — Pydantic schema passed)
# ---------------------------------------------------------------------------

class ValidatedResult(BaseModel):
    trace_id: str
    category: str
    qa_pair: QAPair


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

class ValidationSummary(BaseModel):
    total_generated: int
    total_valid: int
    total_invalid: int
    validation_rate: float
    common_errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3: Failure label result (6 binary modes)
# ---------------------------------------------------------------------------

FAILURE_MODE_FIELDS: list[str] = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]


class FailureLabelResult(BaseModel):
    trace_id: str
    category: str
    incomplete_answer: int = Field(..., ge=0, le=1)
    safety_violations: int = Field(..., ge=0, le=1)
    unrealistic_tools: int = Field(..., ge=0, le=1)
    overcomplicated_solution: int = Field(..., ge=0, le=1)
    missing_context: int = Field(..., ge=0, le=1)
    poor_quality_tips: int = Field(..., ge=0, le=1)
    overall_failure: int  # 1 if ANY mode fails
    failure_count: int


# ---------------------------------------------------------------------------
# Phase 5: Quality evaluation result (6 dimensions — D1–D6 per spec)
# ---------------------------------------------------------------------------

# Human labeler field name → LLM judge field name.
# Single source of truth for agreement.py and mock_seeder.py.
# D4 has different names: human uses scope_appropriateness, LLM uses appropriate_scope.
HUMAN_TO_LLM: dict[str, str] = {
    "answer_completeness": "answer_completeness",    # D1
    "safety_specificity": "safety_specificity",      # D2
    "tool_realism": "tool_realism",                  # D3
    "scope_appropriateness": "appropriate_scope",    # D4
    "context_clarity": "context_clarity",            # D5
    "tip_usefulness": "tip_usefulness",              # D6
}

QUALITY_DIMENSION_FIELDS: list[str] = [
    "answer_completeness",    # D1
    "safety_specificity",     # D2
    "tool_realism",           # D3
    "appropriate_scope",      # D4
    "context_clarity",        # D5
    "tip_usefulness",         # D6
]


class QualityEvalResult(BaseModel):
    trace_id: str
    category: str
    answer_completeness: int = Field(..., ge=0, le=1)   # D1
    safety_specificity: int = Field(..., ge=0, le=1)    # D2
    tool_realism: int = Field(..., ge=0, le=1)          # D3
    appropriate_scope: int = Field(..., ge=0, le=1)     # D4
    context_clarity: int = Field(..., ge=0, le=1)       # D5
    tip_usefulness: int = Field(..., ge=0, le=1)        # D6
    overall_quality_pass: int  # 1 if ALL 6 dimensions pass


# ---------------------------------------------------------------------------
# Phase 3: Benchmark calibration report
# ---------------------------------------------------------------------------

class BenchmarkReport(BaseModel):
    benchmark_samples_evaluated: int
    benchmark_quality_pass_rate: float
    calibration_passed: bool              # True if pass rate >= 95% — judge is trustworthy
    benchmark_dimension_rates: dict[str, float]  # per-dimension pass rates; used by Phase 6 for gap analysis


# ---------------------------------------------------------------------------
# Phase 1a: Shared question set (fixed inputs for controlled baseline comparison)
# ---------------------------------------------------------------------------

class SharedQuestion(BaseModel):
    trace_id: str
    category: str
    question: str = Field(..., min_length=10, max_length=500)
    equipment_problem: str = Field(..., min_length=5, max_length=200)


# ---------------------------------------------------------------------------
# Phase 6: Analysis summary
# ---------------------------------------------------------------------------

class AnalysisSummary(BaseModel):
    total_samples: int
    overall_failure_rate: float
    failure_rates_by_mode: dict[str, float]
    failure_rates_by_category: dict[str, float]
    quality_pass_rates_by_dimension: dict[str, float]
    overall_quality_pass_rate: float
    thresholds_met: dict[str, bool]
    most_problematic_items: list[str]       # trace_ids with 3+ failures
    # Benchmark gap — populated when Phase 3 benchmark_eval.csv is present
    overall_benchmark_gap: Optional[float] = None   # benchmark_pass_rate − generated_pass_rate (apples-to-apples)
    benchmark_dimension_gaps: Optional[dict[str, float]] = None  # per-dimension gaps


# ---------------------------------------------------------------------------
# Phase 7: Before/after correction comparison
# ---------------------------------------------------------------------------

# Absolute quality targets (derived from project spec)
CORRECTION_TARGET_FAILURE_RATE: float = 0.15   # corrected failure rate must be ≤ 15%
CORRECTION_TARGET_QUALITY_PASS: float = 0.80   # corrected quality pass rate must be ≥ 80%
CORRECTION_TARGET_IMPROVEMENT: float = 80.0    # relative failure reduction must be ≥ 80%

class ComparisonReport(BaseModel):
    baseline_failure_rate: float
    corrected_failure_rate: float
    improvement_pct: float          # (baseline − corrected) / baseline * 100
    target_met: bool                # corrected_failure_rate ≤ 15% AND quality_pass ≥ 80% AND improvement ≥ 80%
    per_mode_delta: dict[str, float]
    baseline_quality_pass_rate: float
    corrected_quality_pass_rate: float
    per_dim_quality_delta: dict[str, float] = {}  # per-dim pass rate delta (baseline − corrected; positive = worse)
    iterations_run: int = 1         # how many correction iterations were needed
    diversity_score: float = 1.0    # fraction of answer pairs with Jaccard similarity ≤ 0.8 (1.0 = fully diverse)
