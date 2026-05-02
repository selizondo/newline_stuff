"""Phase 2 — Schema validation and error categorization."""

from pathlib import Path
from typing import Optional

from .schema import JobDescription, Resume, ResumeJobPair
from .utils.storage import save_jsonl
from .validators.schema_validator import SchemaValidator


def run_validation_phase(
    jobs: list[JobDescription],
    resumes: list[Resume],
    output_dir: str,
    run_label: str,
    generate_heatmaps: bool = True,
) -> dict:
    """Run Pydantic schema validation on generated jobs and resumes.

    Args:
        jobs: Generated job descriptions.
        resumes: Generated resumes.
        output_dir: Base output directory.
        run_label: Timestamp label used for output filenames.

    Returns:
        Dict with keys: resume_results, job_results, resume_summary,
        job_summary, invalid_resumes, invalid_jobs, files.
    """
    validator = SchemaValidator()
    validated_dir = Path(output_dir) / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 2.1: Validate resumes ────────────────────────────────────────────
    print(f"  Validating {len(resumes)} resumes...")
    resume_data = [r.model_dump(mode="json") for r in resumes]
    resume_results, resume_summary = validator.validate_batch(resume_data, data_type="resume")

    valid_count = resume_summary.get("valid", 0)
    total_count = resume_summary.get("total", len(resumes))
    rate = valid_count / total_count if total_count else 0
    status = "✓" if rate >= 0.9 else "✗"
    print(f"  {status} Resumes: {valid_count}/{total_count} valid ({rate*100:.1f}%) — target >90%")

    # ── Step 2.2: Validate jobs ────────────────────────────────────────────────
    print(f"  Validating {len(jobs)} job descriptions...")
    job_data = [j.model_dump(mode="json") for j in jobs]
    job_results, job_summary = validator.validate_batch(job_data, data_type="job")

    valid_jobs = job_summary.get("valid", 0)
    total_jobs = job_summary.get("total", len(jobs))
    job_rate = valid_jobs / total_jobs if total_jobs else 0
    status = "✓" if job_rate >= 0.9 else "✗"
    print(f"  {status} Jobs: {valid_jobs}/{total_jobs} valid ({job_rate*100:.1f}%) — target >90%")

    # ── Step 2.3: Save invalid records ────────────────────────────────────────
    files: dict[str, str] = {}
    labeled_dir = Path(output_dir) / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    invalid_resumes = [r for r in resume_results if not r.is_valid]
    invalid_jobs = [r for r in job_results if not r.is_valid]

    if invalid_resumes or invalid_jobs:
        invalid_data = [
            {"type": "resume", **r.to_dict()} for r in invalid_resumes
        ] + [
            {"type": "job", **r.to_dict()} for r in invalid_jobs
        ]
        invalid_file = labeled_dir / f"invalid_{run_label}.jsonl"
        save_jsonl(invalid_data, invalid_file)
        files["invalid_records"] = str(invalid_file)
        print(f"  Saved → {invalid_file}")

    print(f"\n  Phase 2 complete: {len(invalid_resumes)} invalid resumes, {len(invalid_jobs)} invalid jobs")

    # ── Step 2.4: Generate field-level validation heatmaps ───────────────────
    if generate_heatmaps:
        from .analysis.heatmap import HeatmapGenerator
        viz_dir = validated_dir / "visualizations"
        hm = HeatmapGenerator(output_dir=str(viz_dir))
        heatmap_tasks = [
            ("resume_validation_heatmap", lambda: hm.create_field_validation_heatmap(
                resume_results, data_type="resume",
                filename=f"resume_field_validation_{run_label}.png")),
            ("job_validation_heatmap", lambda: hm.create_field_validation_heatmap(
                job_results, data_type="job",
                filename=f"job_field_validation_{run_label}.png")),
        ]
        for key, fn in heatmap_tasks:
            try:
                files[key] = str(fn())
            except Exception as exc:
                print(f"  Warning: heatmap '{key}' failed: {exc}")
        heatmap_count = sum(1 for k in ("resume_validation_heatmap", "job_validation_heatmap") if k in files)
        if heatmap_count:
            print(f"  Generated {heatmap_count} heatmaps → {viz_dir}")

    return {
        "resume_results": resume_results,
        "job_results": job_results,
        "resume_summary": resume_summary,
        "job_summary": job_summary,
        "invalid_resumes": invalid_resumes,
        "invalid_jobs": invalid_jobs,
        "files": files,
    }
