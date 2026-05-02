"""CLI entry point for the Synthetic Data Resume Coach pipeline."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import PipelineConfig
from llm_utils.config import get_settings
from .phase1_generation import run_generation_phase
from .phase2_validation import run_validation_phase
from .phase3_labeling import run_labeling_phase
from .phase4_correction import run_correction_phase


# ── Console helpers ────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{text}\n{line}")


def _section(text: str) -> None:
    print(f"\n{'─' * 50}\n{text}\n{'─' * 50}")


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Synthetic Data Resume Coach — Data Generation Pipeline"
    )
    parser.add_argument("--num-jobs", "-n", type=int, default=10,
                        help="Number of job descriptions to generate (default: 10)")
    parser.add_argument("--resumes-per-job", type=int, default=5,
                        help="Resumes per job, one per fit level (default: 5)")
    parser.add_argument("--model", "-m", type=str, default="",
                        help="LLM model for generation (default: LLM_MODEL env var)")
    parser.add_argument("--judge-model", type=str, default="",
                        help="LLM model for judging (defaults to --model)")
    parser.add_argument("--no-correction", action="store_true",
                        help="Disable correction loop (Phase 4)")
    parser.add_argument("--no-heatmaps", action="store_true",
                        help="Disable heatmap generation")
    parser.add_argument("--enable-llm-judge", action="store_true",
                        help="Enable LLM judge quality assessment (slower)")
    parser.add_argument("--enable-braintrust", action="store_true",
                        help="Enable Braintrust logging (requires BRAINTRUST_API_KEY)")
    parser.add_argument("--output-dir", "-o", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--phase", type=str, default="1-4",
                        help="Phase range to run, e.g. '1-4', '1', '3-4' (default: 1-4)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume a prior run by its run_label (e.g. 20260501_002340). "
                             "Reuses existing checkpoint files and skips already-generated items.")

    args = parser.parse_args()

    # Parse phase range
    if "-" in args.phase:
        phase_start, phase_end = (int(x) for x in args.phase.split("-", 1))
    else:
        phase_start = phase_end = int(args.phase)

    settings = get_settings()  # validates LLM_API_KEY at startup

    model = args.model or settings.generation_model
    config = PipelineConfig(
        num_jobs=args.num_jobs,
        resumes_per_job=args.resumes_per_job,
        model=model,
        output_dir=args.output_dir,
        generate_heatmaps=not args.no_heatmaps,
        enable_correction=not args.no_correction,
        enable_llm_judge=args.enable_llm_judge,
        enable_braintrust=args.enable_braintrust,
    )

    if args.resume:
        run_label = args.resume
        print(f"\n  Resuming run: {run_label}")
    else:
        run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    pipeline_start = time.perf_counter()

    _banner("RESUME COACH — SYNTHETIC DATA PIPELINE")
    print(f"  run_label : {run_label}")
    print(f"  num_jobs  : {config.num_jobs}")
    print(f"  resumes   : {config.resumes_per_job} per job  ({config.num_jobs * config.resumes_per_job} total pairs)")
    print(f"  model     : {config.model}")
    print(f"  output    : {output_dir}")
    print(f"  phases    : {phase_start}–{phase_end}")

    results: dict = {
        "run_label": run_label,
        "config": {
            "num_jobs": config.num_jobs,
            "resumes_per_job": config.resumes_per_job,
            "model": config.model,
        },
        "phases": {},
        "stage_times_seconds": {},
        "files": {},
    }

    # ── Phase 1: Generation ───────────────────────────────────────────────────
    if phase_start <= 1 <= phase_end:
        _section("PHASE 1 — Generation")
        t0 = time.perf_counter()

        gen = run_generation_phase(
            num_jobs=config.num_jobs,
            resumes_per_job=config.resumes_per_job,
            model=config.model,
            output_dir=str(output_dir),
            run_label=run_label,
        )

        results["stage_times_seconds"]["phase1_generation"] = round(time.perf_counter() - t0, 2)
        results["phases"]["generation"] = {
            "jobs_generated": len(gen["jobs"]),
            "pairs_generated": len(gen["pairs"]),
        }
        results["files"].update(gen["files"])

    # ── Phase 2: Validation ───────────────────────────────────────────────────
    if phase_start <= 2 <= phase_end:
        _section("PHASE 2 — Schema Validation")
        t0 = time.perf_counter()

        val = run_validation_phase(
            jobs=gen["jobs"],
            resumes=gen["resumes"],
            output_dir=str(output_dir),
            run_label=run_label,
            generate_heatmaps=config.generate_heatmaps,
        )

        results["stage_times_seconds"]["phase2_validation"] = round(time.perf_counter() - t0, 2)
        results["phases"]["validation"] = {
            "resume_summary": val["resume_summary"],
            "job_summary": val["job_summary"],
        }
        results["files"].update(val["files"])

    # ── Phase 3: Failure Labeling ─────────────────────────────────────────────
    if phase_start <= 3 <= phase_end:
        _section("PHASE 3 — Failure Mode Labeling")
        t0 = time.perf_counter()

        lab = run_labeling_phase(
            pairs=gen["pairs"],
            output_dir=str(output_dir),
            run_label=run_label,
            generate_heatmaps=config.generate_heatmaps,
        )

        results["stage_times_seconds"]["phase3_labeling"] = round(time.perf_counter() - t0, 2)
        results["phases"]["labeling"] = lab["statistics"]
        results["files"].update(lab["files"])

    # ── Phase 4: Correction Loop ──────────────────────────────────────────────
    if phase_start <= 4 <= phase_end and config.enable_correction:
        _section("PHASE 4 — Correction Loop")
        t0 = time.perf_counter()

        corr = run_correction_phase(
            invalid_resumes=val["invalid_resumes"],
            invalid_jobs=val["invalid_jobs"],
            model=config.model,
            max_retries=config.max_correction_retries,
            output_dir=str(output_dir),
            run_label=run_label,
        )

        results["stage_times_seconds"]["phase4_correction"] = round(time.perf_counter() - t0, 2)
        results["phases"]["correction"] = corr["stats"]
        results["files"].update(corr["files"])
    elif phase_start <= 4 <= phase_end:
        _section("PHASE 4 — Correction Loop (skipped — disabled)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = round(time.perf_counter() - pipeline_start, 2)
    results["total_time_seconds"] = total_time

    _section("PIPELINE COMPLETE")
    print(f"  Total time: {total_time}s")
    print("\n  Stage timings:")
    for stage, secs in results["stage_times_seconds"].items():
        print(f"    {stage}: {secs}s")

    print("\n  Output files:")
    for key, path in results["files"].items():
        print(f"    {key}: {path}")

    # ── Fix 3: summary at output root ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"pipeline_summary_{run_label}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Summary → {summary_file}")

    # ── Fix 2: structured iteration log ──────────────────────────────────────
    log_file = output_dir / "iteration_log.jsonl"
    before_stats: dict = {}
    if log_file.exists():
        lines = log_file.read_text().strip().splitlines()
        if lines:
            try:
                before_stats = json.loads(lines[-1]).get("after", {})
            except (json.JSONDecodeError, KeyError):
                pass

    phases = results.get("phases", {})
    after_stats = {
        "jobs_generated": phases.get("generation", {}).get("jobs_generated"),
        "pairs_generated": phases.get("generation", {}).get("pairs_generated"),
        "resume_valid_rate": phases.get("validation", {}).get("resume_summary", {}).get("valid_rate"),
        "labeling_pass_rate": phases.get("labeling", {}).get("overall_pass_rate"),
        "correction_success_rate": phases.get("correction", {}).get("success_rate"),
        "total_time_seconds": results.get("total_time_seconds"),
    }
    delta = {
        k: round(after_stats[k] - before_stats[k], 4)
        for k in after_stats
        if k in before_stats
        and after_stats[k] is not None
        and isinstance(after_stats[k], (int, float))
    }
    log_entry = {
        "date": run_label,
        "component": "pipeline",
        "change": "",
        "reason": "",
        "config": results.get("config", {}),
        "before": before_stats,
        "after": after_stats,
        "delta": delta,
        "keep": True,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")
    print(f"  Log     → {log_file}")


if __name__ == "__main__":
    main()
