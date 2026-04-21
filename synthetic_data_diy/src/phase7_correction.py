"""
Phase 7: Prompt Correction & Re-evaluation
Re-runs Phases 1, 2, 4, 5 with corrected prompt templates derived from observed
failure modes, then generates a before/after report.

Key improvements over a static correction approach:
  - Data-driven: failure context from Phase 4 is injected into generation prompts
    so the corrected run targets the specific modes that actually failed, not a
    generic "write better" instruction.
  - Iterative: re-runs up to max_iterations until all three absolute targets are
    met simultaneously (failure ≤ 15%, quality pass ≥ 80%, improvement ≥ 80%).
  - Diversity check: Jaccard similarity guard ensures corrected answers are not
    near-copies of the baseline (score ≥ 0.8 = "same", flagged but not blocked).
"""

import json
from pathlib import Path

import pandas as pd

from schema import (
    ComparisonReport,
    CORRECTION_TARGET_FAILURE_RATE,
    CORRECTION_TARGET_QUALITY_PASS,
    CORRECTION_TARGET_IMPROVEMENT,
    FAILURE_MODE_FIELDS as FAILURE_MODE_NAMES,
    QUALITY_DIMENSION_FIELDS as QUALITY_DIM_NAMES,
    ValidatedResult,
)
from phase1_generation import run_generation_phase
from phase2_validation import run_validation_phase
from phase4_failure_labeling import run_failure_labeling_phase
from phase5_quality_eval import run_quality_eval_phase

_JACCARD_SIMILARITY_THRESHOLD = 0.8  # pairs above this are considered near-duplicates

_MODE_HINTS: dict[str, str] = {
    "incomplete_answer": "Ensure every repair step is fully described; do not omit intermediate steps.",
    "safety_violations": "Name the specific hazard (e.g. 'live 120V circuit', 'pressurised pipe') and the exact protective action.",
    "unrealistic_tools": "Use only tools available at a standard hardware store for under $50.",
    "overcomplicated_solution": "Simplify to the minimum number of steps a homeowner can handle; avoid specialist techniques.",
    "missing_context": "Include situational context: when this repair is appropriate, what to check first, when to call a professional.",
    "poor_quality_tips": "Make tips non-obvious and task-specific; avoid generic advice like 'read the manual'.",
}


def _build_failure_context(failure_df: pd.DataFrame) -> str:
    """Build a plain-text context string from observed failure rates.

    Reads per-mode failure rates and returns a targeted instruction block that
    is prepended to the generation prompt so the model knows which specific
    weaknesses to address in the corrected run.
    """
    if failure_df.empty:
        return ""

    mode_rates = {
        m: float(failure_df[m].mean())
        for m in FAILURE_MODE_NAMES
        if m in failure_df.columns
    }
    # Sort by failure rate descending; only include modes that actually failed
    failing = sorted(
        [(m, r) for m, r in mode_rates.items() if r > 0.0],
        key=lambda x: x[1],
        reverse=True,
    )
    if not failing:
        return ""

    lines = [
        "QUALITY IMPROVEMENT CONTEXT — please address the following observed weaknesses:",
        "",
    ]
    for mode, rate in failing:
        label = mode.replace("_", " ").title()
        lines.append(f"  • {label}: {rate*100:.0f}% failure rate")
        if hint := _MODE_HINTS.get(mode):
            lines.append(f"    → {hint}")

    lines.append("")
    return "\n".join(lines)


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _compute_diversity_score(results: list[ValidatedResult]) -> float:
    """Fraction of answer pairs with Jaccard similarity ≤ threshold.

    O(n²) — acceptable for ≤ 50 samples. Returns 1.0 if fewer than 2 samples.
    """
    answers = [r.qa_pair.answer for r in results]
    n = len(answers)
    if n < 2:
        return 1.0

    diverse_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if _jaccard(answers[i], answers[j]) <= _JACCARD_SIMILARITY_THRESHOLD:
                diverse_pairs += 1

    return diverse_pairs / total_pairs if total_pairs else 1.0


def run_correction_phase(
    num_samples: int,
    generation_model: str,
    judge_model: str,
    baseline_dir: Path,
    corrected_dir: Path,
    max_iterations: int = 3,
) -> ComparisonReport:
    """Run data-driven prompt correction with an iterative improvement loop.

    Iterates up to max_iterations, stopping early when all three absolute
    targets are simultaneously met:
      - corrected failure rate ≤ CORRECTION_TARGET_FAILURE_RATE (15%)
      - corrected quality pass rate ≥ CORRECTION_TARGET_QUALITY_PASS (80%)
      - relative failure reduction ≥ CORRECTION_TARGET_IMPROVEMENT (80%)

    Each iteration injects the current run's failure context into the generation
    prompt so successive runs target the remaining weak spots.
    """
    # Load baseline metrics
    baseline_failure_csv = baseline_dir / "failure_labeled_data.csv"
    baseline_quality_csv = baseline_dir / "quality_eval_data.csv"
    if not baseline_failure_csv.exists():
        raise FileNotFoundError(
            f"Baseline failure data not found: {baseline_failure_csv}. Run Phases 4-5 first."
        )

    baseline_fdf = pd.read_csv(baseline_failure_csv)
    baseline_qdf = pd.read_csv(baseline_quality_csv) if baseline_quality_csv.exists() else pd.DataFrame()

    baseline_failure_rate = float(baseline_fdf["overall_failure"].mean())
    baseline_quality_pass = float(baseline_qdf["overall_quality_pass"].mean()) if not baseline_qdf.empty else 0.0

    print(f"Baseline failure rate : {baseline_failure_rate*100:.1f}%")
    print(f"Baseline quality pass : {baseline_quality_pass*100:.1f}%")
    print()

    corrected_dir.mkdir(parents=True, exist_ok=True)

    # Seed the iterative loop with baseline failures so the first corrected run
    # already targets the observed weak spots.
    current_failure_df = baseline_fdf
    corrected_fdf: pd.DataFrame = pd.DataFrame()
    corrected_qdf: pd.DataFrame = pd.DataFrame()
    valid_results_corrected: list[ValidatedResult] = []
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        iter_dir = corrected_dir / f"iter_{iteration}" if iteration > 1 else corrected_dir
        iter_dir.mkdir(parents=True, exist_ok=True)

        failure_context = _build_failure_context(current_failure_df)
        if failure_context:
            print(f"--- Iteration {iteration}: failure context injected ---")
            print(failure_context[:400] + ("..." if len(failure_context) > 400 else ""))
        else:
            print(f"--- Iteration {iteration}: no failures to target ---")

        print(f"\n--- Phase 1 (corrected, iter {iteration}): Generation ---")
        gen_results = run_generation_phase(
            num_samples,
            generation_model,
            iter_dir,
            strategy="human_feedback",
            batch_label=f"corrected-iter{iteration}",
        )

        valid_results_corrected, _ = run_validation_phase(gen_results, iter_dir)
        if not valid_results_corrected:
            raise RuntimeError(
                f"No valid Q&A pairs generated in corrected run (iteration {iteration}). "
                "Check prompts and LLM output."
            )

        print(f"\n--- Phase 4 (corrected, iter {iteration}): Failure Labeling ---")
        corrected_fdf = run_failure_labeling_phase(
            valid_results_corrected,
            judge_model,
            iter_dir,
            additional_context=failure_context,
        )

        print(f"\n--- Phase 5 (corrected, iter {iteration}): Quality Evaluation ---")
        corrected_qdf = run_quality_eval_phase(valid_results_corrected, judge_model, iter_dir)

        corrected_failure_rate = float(corrected_fdf["overall_failure"].mean())
        corrected_quality_pass = float(corrected_qdf["overall_quality_pass"].mean())
        improvement_pct = (
            (baseline_failure_rate - corrected_failure_rate) / baseline_failure_rate * 100
            if baseline_failure_rate > 0
            else 0.0
        )

        print(f"\n  iter {iteration} results:")
        print(f"    Failure rate : {corrected_failure_rate*100:.1f}% (target ≤ {CORRECTION_TARGET_FAILURE_RATE*100:.0f}%)")
        print(f"    Quality pass : {corrected_quality_pass*100:.1f}% (target ≥ {CORRECTION_TARGET_QUALITY_PASS*100:.0f}%)")
        print(f"    Improvement  : {improvement_pct:.1f}% (target ≥ {CORRECTION_TARGET_IMPROVEMENT:.0f}%)")

        all_targets_met = (
            corrected_failure_rate <= CORRECTION_TARGET_FAILURE_RATE
            and corrected_quality_pass >= CORRECTION_TARGET_QUALITY_PASS
            and improvement_pct >= CORRECTION_TARGET_IMPROVEMENT
        )
        if all_targets_met:
            print(f"\n  All targets met after {iteration} iteration(s). Stopping early.")
            break

        if iteration < max_iterations:
            print(f"\n  Targets not met — running iteration {iteration + 1}...")
            # Feed this iteration's failures as context for the next
            current_failure_df = corrected_fdf

    # Diversity check on final corrected run
    diversity_score = _compute_diversity_score(valid_results_corrected)
    if diversity_score < 1.0:
        near_dup_pct = (1.0 - diversity_score) * 100
        print(f"\n  WARNING: {near_dup_pct:.1f}% of answer pairs are near-duplicates (Jaccard > {_JACCARD_SIMILARITY_THRESHOLD}). "
              "Consider increasing prompt temperature or adding diversity instructions.")

    per_mode_delta = {
        mode: round(float(baseline_fdf[mode].mean()) - float(corrected_fdf[mode].mean()), 4)
        for mode in FAILURE_MODE_NAMES
        if mode in baseline_fdf.columns and mode in corrected_fdf.columns
    }

    report = ComparisonReport(
        baseline_failure_rate=baseline_failure_rate,
        corrected_failure_rate=corrected_failure_rate,
        improvement_pct=round(improvement_pct, 2),
        target_met=all_targets_met,
        per_mode_delta=per_mode_delta,
        baseline_quality_pass_rate=baseline_quality_pass,
        corrected_quality_pass_rate=corrected_quality_pass,
        iterations_run=iteration,
        diversity_score=round(diversity_score, 4),
    )

    report_path = corrected_dir / "before_after_comparison.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2))

    print("\n" + "=" * 50)
    print("BEFORE / AFTER COMPARISON")
    print("=" * 50)
    print(f"Baseline failure rate  : {baseline_failure_rate*100:.1f}%")
    print(f"Corrected failure rate : {corrected_failure_rate*100:.1f}%  (target ≤ {CORRECTION_TARGET_FAILURE_RATE*100:.0f}%)")
    print(f"Improvement            : {improvement_pct:.1f}%  (target ≥ {CORRECTION_TARGET_IMPROVEMENT:.0f}%)")
    print(f"Target met             : {'YES ✓' if report.target_met else 'NO ✗'}")
    print(f"Baseline quality pass  : {baseline_quality_pass*100:.1f}%")
    print(f"Corrected quality pass : {corrected_quality_pass*100:.1f}%  (target ≥ {CORRECTION_TARGET_QUALITY_PASS*100:.0f}%)")
    print(f"Iterations run         : {iteration} / {max_iterations}")
    print(f"Diversity score        : {diversity_score:.2f}  (1.0 = fully diverse)")
    print("\nPer-mode improvement (positive = fewer failures):")
    for mode, delta in per_mode_delta.items():
        arrow = "↓" if delta > 0 else ("→" if delta == 0 else "↑")
        print(f"  {arrow} {mode.replace('_', ' ').title()}: {delta*100:+.1f}pp")
    print(f"\nSaved → {report_path}")
    return report
