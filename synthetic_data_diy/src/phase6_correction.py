"""
Phase 6: Prompt Correction & Re-evaluation
Re-runs Phases 1-4 with corrected prompt templates, then generates a before/after report.
"""

import json
from pathlib import Path

import pandas as pd

from schema import ComparisonReport
from phase1_generation import run_generation_phase
from phase2_validation import run_validation_phase
from schema import FAILURE_MODE_FIELDS as FAILURE_MODE_NAMES
from phase3_failure_labeling import run_failure_labeling_phase
from schema import QUALITY_DIMENSION_FIELDS as QUALITY_DIM_NAMES
from phase4_quality_eval import run_quality_eval_phase


def run_correction_phase(
    num_samples: int,
    generation_model: str,
    judge_model: str,
    baseline_dir: Path,
    corrected_dir: Path,
) -> ComparisonReport:
    """
    1. Load baseline failure/quality metrics.
    2. Re-run Phases 1-4 using CORRECTED_TEMPLATES.
    3. Compute before/after comparison.
    """
    # Load baseline metrics
    baseline_failure_csv = baseline_dir / "failure_labeled_data.csv"
    baseline_quality_csv = baseline_dir / "quality_eval_data.csv"
    if not baseline_failure_csv.exists():
        raise FileNotFoundError(f"Baseline failure data not found: {baseline_failure_csv}. Run Phases 1-3 first.")

    baseline_fdf = pd.read_csv(baseline_failure_csv)
    baseline_qdf = pd.read_csv(baseline_quality_csv) if baseline_quality_csv.exists() else pd.DataFrame()

    baseline_failure_rate = float(baseline_fdf["overall_failure"].mean())
    baseline_quality_pass = float(baseline_qdf["overall_quality_pass"].mean()) if not baseline_qdf.empty else 0.0

    print(f"Baseline failure rate: {baseline_failure_rate*100:.1f}%")
    print(f"Baseline quality pass: {baseline_quality_pass*100:.1f}%")
    print()

    # Re-run pipeline with corrected prompts
    corrected_dir.mkdir(parents=True, exist_ok=True)

    print("--- Phase 1 (corrected): Generation ---")
    gen_results = run_generation_phase(num_samples, generation_model, corrected_dir, strategy="human_feedback")

    valid_results_corrected, _ = run_validation_phase(gen_results, corrected_dir)
    if not valid_results_corrected:
        raise RuntimeError("No valid Q&A pairs generated in corrected run. Check prompts and LLM output.")

    print("\n--- Phase 3 (corrected): Failure Labeling ---")
    corrected_fdf = run_failure_labeling_phase(valid_results_corrected, judge_model, corrected_dir)

    print("\n--- Phase 4 (corrected): Quality Evaluation ---")
    corrected_qdf = run_quality_eval_phase(valid_results_corrected, judge_model, corrected_dir)

    # Compute comparison
    corrected_failure_rate = float(corrected_fdf["overall_failure"].mean())
    corrected_quality_pass = float(corrected_qdf["overall_quality_pass"].mean())

    improvement_pct = (
        (baseline_failure_rate - corrected_failure_rate) / baseline_failure_rate * 100
        if baseline_failure_rate > 0
        else 0.0
    )

    per_mode_delta = {}
    for mode in FAILURE_MODE_NAMES:
        if mode in baseline_fdf.columns and mode in corrected_fdf.columns:
            base_rate = float(baseline_fdf[mode].mean())
            corr_rate = float(corrected_fdf[mode].mean())
            per_mode_delta[mode] = round(base_rate - corr_rate, 4)

    report = ComparisonReport(
        baseline_failure_rate=baseline_failure_rate,
        corrected_failure_rate=corrected_failure_rate,
        improvement_pct=round(improvement_pct, 2),
        target_met=improvement_pct >= 80.0,
        per_mode_delta=per_mode_delta,
        baseline_quality_pass_rate=baseline_quality_pass,
        corrected_quality_pass_rate=corrected_quality_pass,
    )

    report_path = corrected_dir / "before_after_comparison.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2))

    print("\n" + "=" * 50)
    print("BEFORE / AFTER COMPARISON")
    print("=" * 50)
    print(f"Baseline failure rate  : {baseline_failure_rate*100:.1f}%")
    print(f"Corrected failure rate : {corrected_failure_rate*100:.1f}%")
    print(f"Improvement            : {improvement_pct:.1f}%")
    print(f"Target met (>80%)      : {'YES ✓' if report.target_met else 'NO ✗'}")
    print(f"Baseline quality pass  : {baseline_quality_pass*100:.1f}%")
    print(f"Corrected quality pass : {corrected_quality_pass*100:.1f}%")
    print("\nPer-mode improvement (positive = fewer failures):")
    for mode, delta in per_mode_delta.items():
        arrow = "↓" if delta > 0 else ("→" if delta == 0 else "↑")
        print(f"  {arrow} {mode.replace('_', ' ').title()}: {delta*100:+.1f}pp")
    print(f"\nSaved → {report_path}")
    return report
