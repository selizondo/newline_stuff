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
# Phase 4: Quality evaluation result (8 dimensions)
# ---------------------------------------------------------------------------

QUALITY_DIMENSION_FIELDS: list[str] = [
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
]

QUALITY_THRESHOLDS: dict[str, float] = {
    "answer_coherence": 0.90,
    "step_actionability": 0.85,
    "tool_realism": 0.95,
    "safety_specificity": 0.90,
    "tip_usefulness": 0.85,
    "problem_answer_alignment": 0.95,
    "appropriate_scope": 0.95,
    "category_accuracy": 0.98,
}


class QualityEvalResult(BaseModel):
    trace_id: str
    category: str
    answer_coherence: int = Field(..., ge=0, le=1)       # Q1
    step_actionability: int = Field(..., ge=0, le=1)     # Q2
    tool_realism: int = Field(..., ge=0, le=1)           # Q3
    safety_specificity: int = Field(..., ge=0, le=1)     # Q4
    tip_usefulness: int = Field(..., ge=0, le=1)         # Q5
    problem_answer_alignment: int = Field(..., ge=0, le=1)  # Q6
    appropriate_scope: int = Field(..., ge=0, le=1)      # Q7
    category_accuracy: int = Field(..., ge=0, le=1)      # Q8
    overall_quality_pass: int  # 1 if ALL dimensions pass


# ---------------------------------------------------------------------------
# Phase 5: Analysis summary
# ---------------------------------------------------------------------------

class AnalysisSummary(BaseModel):
    total_samples: int
    overall_failure_rate: float
    failure_rates_by_mode: dict[str, float]
    failure_rates_by_category: dict[str, float]
    quality_pass_rates_by_dimension: dict[str, float]
    overall_quality_pass_rate: float
    thresholds_met: dict[str, bool]
    most_problematic_items: list[str]  # trace_ids with 3+ failures


# ---------------------------------------------------------------------------
# Phase 6: Before/after comparison
# ---------------------------------------------------------------------------

class ComparisonReport(BaseModel):
    baseline_failure_rate: float
    corrected_failure_rate: float
    improvement_pct: float  # (baseline - corrected) / baseline * 100
    target_met: bool        # improvement_pct >= 80
    per_mode_delta: dict[str, float]
    baseline_quality_pass_rate: float
    corrected_quality_pass_rate: float


# ---------------------------------------------------------------------------
# Phase 7: Benchmark calibration report
# ---------------------------------------------------------------------------

class BenchmarkReport(BaseModel):
    benchmark_samples_evaluated: int
    benchmark_quality_pass_rate: float
    calibration_passed: bool  # >= 95% pass rate
    generated_vs_benchmark: dict[str, float]  # dimension -> gap
    overall_gap: float
