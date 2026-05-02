"""Phase 4 — Correction loop for invalid records."""

from pathlib import Path

from .correction.llm_correction import LLMCorrector
from .validators.schema_validator import ValidationResult


def run_correction_phase(
    invalid_resumes: list[ValidationResult],
    invalid_jobs: list[ValidationResult],
    model: str,
    max_retries: int,
    output_dir: str,
    run_label: str,
) -> dict:
    """Feed Pydantic validation errors back to the LLM and re-validate.

    Attempts up to max_retries corrections per invalid record. Tracks success
    rate per attempt to evaluate correction loop effectiveness (target >50%).

    Args:
        invalid_resumes: ValidationResult objects that failed schema validation.
        invalid_jobs: ValidationResult objects that failed schema validation.
        model: LLM model name.
        max_retries: Maximum correction attempts per record (spec: 3).
        output_dir: Base output directory.
        run_label: Timestamp label used for output filenames.

    Returns:
        Dict with keys: stats, correction_results, files.
    """
    corrector = LLMCorrector(model=model, max_retries=max_retries)
    validated_dir = Path(output_dir) / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)

    total_invalid = len(invalid_resumes) + len(invalid_jobs)
    if total_invalid == 0:
        print("  No invalid records — correction phase skipped")
        return {"stats": {}, "correction_results": [], "files": {}}

    print(f"  {total_invalid} invalid records to correct ({len(invalid_resumes)} resumes, {len(invalid_jobs)} jobs)")

    # ── Step 4.1: Correct invalid resumes ─────────────────────────────────────
    correction_results = []

    if invalid_resumes:
        print(f"  Correcting {len(invalid_resumes)} resumes (max {max_retries} retries each)...")
        resume_corrections = corrector.correct_batch(invalid_resumes, data_type="resume")
        correction_results.extend(resume_corrections)
        corrected = sum(1 for r in resume_corrections if getattr(r, "success", False))
        print(f"  [{corrected}/{len(invalid_resumes)}] resumes corrected")

    # ── Step 4.2: Correct invalid jobs ────────────────────────────────────────
    if invalid_jobs:
        print(f"  Correcting {len(invalid_jobs)} jobs (max {max_retries} retries each)...")
        job_corrections = corrector.correct_batch(invalid_jobs, data_type="job")
        correction_results.extend(job_corrections)
        corrected = sum(1 for r in job_corrections if getattr(r, "success", False))
        print(f"  [{corrected}/{len(invalid_jobs)}] jobs corrected")

    # ── Step 4.3: Save and report ─────────────────────────────────────────────
    files: dict[str, str] = {}
    stats = corrector.get_stats()

    if correction_results:
        correction_file = corrector.save_results(
            correction_results,
            output_dir=str(validated_dir),
            filename=f"corrections_{run_label}.jsonl",
        )
        files["corrections"] = str(correction_file)
        print(f"  Saved → {correction_file}")

    overall_rate = stats.get("success_rate", 0)
    status = "✓" if overall_rate >= 0.5 else "✗"
    print(f"\n  Phase 4 complete: {status} {overall_rate*100:.1f}% correction rate — target >50%")

    return {
        "stats": stats,
        "correction_results": correction_results,
        "files": files,
    }
