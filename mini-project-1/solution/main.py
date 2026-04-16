"""
Main CLI orchestrator for the Home DIY Repair Q&A Synthetic Data Pipeline.

Usage:
  python main.py                        # Full pipeline (all 7 phases), 50 samples
  python main.py --samples 100          # Custom sample count
  python main.py --phase 1-5           # Stop after analysis
  python main.py --phase 6             # Correction phase only
  python main.py --phase 7             # Benchmark phase only
  python main.py --corrected           # Use corrected prompts in Phase 1
  python main.py stats                 # Print summary from existing output files
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_phase_range(phase_str: str) -> tuple[int, int]:
    """Parse '1-5', '6', '3-7' into (start, end) inclusive."""
    if "-" in phase_str:
        parts = phase_str.split("-", 1)
        return int(parts[0]), int(parts[1])
    n = int(phase_str)
    return n, n


def _banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{text}\n{line}")


def _section(text: str) -> None:
    print(f"\n{'─'*50}\n{text}\n{'─'*50}")


# ---------------------------------------------------------------------------
# stats subcommand
# ---------------------------------------------------------------------------

def quick_stats(output_dir: Path) -> None:
    for name, fname in [
        ("Validation summary", "validation_summary.json"),
        ("Analysis report", "analysis_report.json"),
        ("Before/after comparison", "corrected/before_after_comparison.json"),
        ("Benchmark report", "benchmark_report.json"),
    ]:
        path = output_dir / fname
        if path.exists():
            data = json.loads(path.read_text())
            print(f"\n── {name} ({path}) ──")
            print(json.dumps(data, indent=2))
        else:
            print(f"\n── {name}: not found ({path}) ──")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Home DIY Repair Q&A Synthetic Data Pipeline — all 7 phases"
    )
    parser.add_argument("command", nargs="?", default=None, help="Optional subcommand: 'stats'")
    parser.add_argument("--samples", type=int, default=50, help="Q&A pairs to generate (default: 50)")
    parser.add_argument("--model", type=str, default=None, help="LLM model override (default: from .env)")
    parser.add_argument("--phase", type=str, default="1-7", help="Phase range to run, e.g. '1-5', '6', '7' (default: 1-7)")
    parser.add_argument("--prompts-dir", type=str, default=None,
                        help="Path to prompt templates directory (default: prompts/baseline)")
    parser.add_argument("--corrected", action="store_true",
                        help="Shorthand for --prompts-dir prompts/corrected")
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory (default: output)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = output_dir / "corrected"

    if args.command == "stats":
        quick_stats(output_dir)
        return

    # Resolve model from CLI or config
    from config import get_settings
    from prompts import BASELINE_DIR, CORRECTED_DIR
    settings = get_settings()
    model = args.model or settings.model

    # Resolve prompts directory: explicit > --corrected shorthand > default baseline
    if args.prompts_dir:
        prompts_dir = Path(args.prompts_dir)
    elif args.corrected:
        prompts_dir = CORRECTED_DIR
    else:
        prompts_dir = BASELINE_DIR

    phase_start, phase_end = _parse_phase_range(args.phase)

    _banner("HOME DIY REPAIR Q&A SYNTHETIC DATA PIPELINE")
    print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model      : {model}")
    print(f"LLM base   : {settings.base_url}")
    print(f"Samples    : {args.samples}")
    print(f"Phases     : {phase_start}–{phase_end}")
    print(f"Prompts    : {prompts_dir}")
    print(f"Output dir : {output_dir.absolute()}")

    generation_results = None
    valid_results = None

    # ── Phase 1: Generation ───────────────────────────────────────────────
    if phase_start <= 1 <= phase_end:
        _section("PHASE 1 — Generation")
        from phase1_generation import run_generation_phase
        generation_results = run_generation_phase(
            num_samples=args.samples,
            model=model,
            output_dir=output_dir,
            prompts_dir=prompts_dir,
        )

    # ── Phase 2: Structural Validation ───────────────────────────────────
    if phase_start <= 2 <= phase_end:
        _section("PHASE 2 — Structural Validation")
        from phase1_generation import load_generation_results
        from phase2_validation import run_validation_phase
        if generation_results is None:
            generation_results = load_generation_results(output_dir)
            print(f"Loaded {len(generation_results)} generation results from disk.")
        valid_results, _ = run_validation_phase(generation_results, output_dir)

    # ── Phase 3: Failure Labeling ─────────────────────────────────────────
    if phase_start <= 3 <= phase_end:
        _section("PHASE 3 — Failure Labeling (LLM-as-Judge, 6 modes)")
        from phase2_validation import load_valid_data
        from phase3_failure_labeling import run_failure_labeling_phase
        if valid_results is None:
            valid_results = load_valid_data(output_dir)
        run_failure_labeling_phase(valid_results, model, output_dir)

    # ── Phase 4: Quality Evaluation ───────────────────────────────────────
    if phase_start <= 4 <= phase_end:
        _section("PHASE 4 — Quality Evaluation (LLM-as-Judge, 8 dimensions)")
        from phase2_validation import load_valid_data
        from phase4_quality_eval import run_quality_eval_phase
        if valid_results is None:
            valid_results = load_valid_data(output_dir)
        run_quality_eval_phase(valid_results, model, output_dir)

    # ── Phase 5: Analysis & Visualizations ───────────────────────────────
    if phase_start <= 5 <= phase_end:
        _section("PHASE 5 — Failure & Quality Analysis")
        from phase5_analysis import run_analysis_phase
        run_analysis_phase(output_dir=output_dir)

    # ── Phase 6: Prompt Correction ────────────────────────────────────────
    if phase_start <= 6 <= phase_end:
        _section("PHASE 6 — Prompt Correction & Re-evaluation")
        from phase6_correction import run_correction_phase
        report = run_correction_phase(
            num_samples=args.samples,
            model=model,
            baseline_dir=output_dir,
            corrected_dir=corrected_dir,
        )
        # Re-run Phase 5 to include corrected data in visualizations
        from phase5_analysis import run_analysis_phase
        print("\nUpdating visualizations with corrected data...")
        run_analysis_phase(output_dir=output_dir, corrected_dir=corrected_dir)

    # ── Phase 7: Benchmark Comparison ────────────────────────────────────
    if phase_start <= 7 <= phase_end:
        _section("PHASE 7 — Benchmark Comparison")
        from phase7_benchmark import run_benchmark_phase
        bench_report = run_benchmark_phase(
            model=model,
            output_dir=output_dir,
            num_samples=50,
        )
        # Update benchmark comparison visualization
        from phase5_analysis import run_analysis_phase
        import pandas as pd
        from models import QUALITY_DIMENSION_FIELDS as QUALITY_DIM_NAMES
        bench_csv = output_dir / "benchmark_eval.csv"
        bench_rates = None
        if bench_csv.exists():
            bdf = pd.read_csv(bench_csv)
            bench_rates = {d: float(bdf[d].mean()) for d in QUALITY_DIM_NAMES if d in bdf.columns}
        print("\nUpdating benchmark comparison chart...")
        run_analysis_phase(
            output_dir=output_dir,
            corrected_dir=corrected_dir if corrected_dir.exists() else None,
            benchmark_rates=bench_rates,
        )

    # ── Final summary ─────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"Output files in: {output_dir.absolute()}")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        from pathlib import Path as _Path
        quick_stats(_Path("output"))
    else:
        main()
