"""Phase 3 — Failure mode labeling for resume-job pairs."""

from pathlib import Path

from .analysis.failure_labeler import FailureLabeler
from .schema import ResumeJobPair
from .utils.storage import save_jsonl


def run_labeling_phase(
    pairs: list[ResumeJobPair],
    output_dir: str,
    run_label: str,
    generate_heatmaps: bool = True,
) -> dict:
    """Calculate all 6 failure metrics for every resume-job pair.

    Metrics computed per pair:
      - Skills overlap (Jaccard similarity, threshold 0.5)
      - Experience mismatch (years gap < 50% of required)
      - Seniority mismatch (|resume_level - job_level| > 1)
      - Missing core skill (any of top-3 required skills absent)
      - Hallucinated skill (excessive Expert claims for experience level)
      - Awkward language (buzzword density or repetition patterns)

    Args:
        pairs: Resume-job pairs to label.
        output_dir: Base output directory.
        run_label: Timestamp label used for output filenames.

    Returns:
        Dict with keys: labels, statistics, files.
    """
    labeler = FailureLabeler()
    labeled_dir = Path(output_dir) / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3.1: Label all pairs ─────────────────────────────────────────────
    print(f"  Labeling {len(pairs)} resume-job pairs...")
    labels = labeler.label_pairs(pairs)

    for i, label in enumerate(labels):
        fail_flags = []
        if label.skills_overlap_ratio < 0.5:
            fail_flags.append(f"overlap={label.skills_overlap_ratio:.2f}")
        if label.experience_mismatch:
            fail_flags.append("exp_mismatch")
        if label.seniority_mismatch:
            fail_flags.append("seniority_mismatch")
        if label.missing_core_skill:
            fail_flags.append("missing_core")
        if label.hallucinated_skill:
            fail_flags.append("hallucination")
        if label.awkward_language_flag:
            fail_flags.append("awkward_lang")

        status = f"PASS" if label.overall_pass else f"FAIL [{', '.join(fail_flags)}]"
        print(f"  [{i+1}/{len(labels)}] {label.trace_id[:12]}... {status}")

    # ── Step 3.2: Save labels ──────────────────────────────────────────────────
    labels_file = labeler.save_labels(
        output_dir=str(labeled_dir),
        filename=f"failure_labels_{run_label}.jsonl",
    )
    print(f"  Saved → {labels_file}")

    # ── Step 3.3: Save failed pairs separately ────────────────────────────────
    files: dict[str, str] = {"failure_labels": str(labels_file)}
    invalid_labels = [lb for lb in labels if not lb.overall_pass]
    if invalid_labels:
        invalid_data = [lb.to_dict() for lb in invalid_labels]
        invalid_file = labeled_dir / f"failed_pairs_{run_label}.jsonl"
        save_jsonl(invalid_data, invalid_file)
        files["failed_pairs"] = str(invalid_file)
        print(f"  Saved → {invalid_file}")

    # ── Step 3.4: Print summary ────────────────────────────────────────────────
    stats = labeler.get_statistics()
    pass_rate = stats.get("overall_pass_rate", 0)
    avg_overlap = stats.get("average_skills_overlap", 0)
    failure_rates = stats.get("failure_rates", {})

    print(f"\n  Phase 3 complete: {pass_rate*100:.1f}% pass rate, avg skills overlap {avg_overlap:.2f}")
    print("  Failure rates by mode:")
    mode_labels = {
        "experience_mismatch": "Experience Mismatch",
        "seniority_mismatch": "Seniority Mismatch",
        "missing_core_skill": "Missing Core Skill",
        "hallucinated_skill": "Hallucination",
        "awkward_language_flag": "Awkward Language",
    }
    for field, label_text in mode_labels.items():
        rate = failure_rates.get(field, 0)
        print(f"    {label_text}: {rate*100:.1f}%")

    # ── Step 3.5: Generate heatmaps ──────────────────────────────────────────
    if generate_heatmaps:
        from .analysis.heatmap import HeatmapGenerator
        viz_dir = labeled_dir / "visualizations"
        hm = HeatmapGenerator(output_dir=str(viz_dir))
        heatmap_files: dict[str, str] = {}
        heatmap_tasks = [
            ("failure_mode_correlation", lambda: hm.create_failure_mode_correlation_matrix(
                labeler, f"failure_mode_correlation_{run_label}.png")),
            ("failure_by_template", lambda: hm.create_failure_rates_by_template_heatmap(
                labeler, f"failure_by_template_{run_label}.png")),
            ("failure_by_fit_level", lambda: hm.create_failure_rates_by_fit_level_heatmap(
                labeler, f"failure_by_fit_level_{run_label}.png")),
            ("niche_vs_standard", lambda: hm.create_niche_vs_standard_comparison(
                labeler, f"niche_vs_standard_{run_label}.png")),
            ("hallucination_by_seniority", lambda: hm.create_hallucination_by_seniority_chart(
                labeler, f"hallucination_by_seniority_{run_label}.png")),
        ]
        for key, fn in heatmap_tasks:
            try:
                heatmap_files[key] = str(fn())
            except Exception as exc:
                print(f"  Warning: heatmap '{key}' failed: {exc}")
        files.update(heatmap_files)
        if heatmap_files:
            print(f"  Generated {len(heatmap_files)} heatmaps → {viz_dir}")

    return {
        "labels": labels,
        "statistics": stats,
        "files": files,
    }
