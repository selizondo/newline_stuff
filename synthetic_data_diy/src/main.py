"""
Main CLI orchestrator for the Home DIY Repair Q&A Synthetic Data Pipeline.

Usage:
  python main.py                                       # Full pipeline, zero_shot, 50 samples
  python main.py --samples 20 --prompt-strategy few_shot --batch-label few-shot-run1
  python main.py --prompt-strategy chain_of_thought --batch-label cot-run1
  python main.py --phase 1 --samples 10              # Phase 1 only
  python main.py --phase 1-5                          # Stop after analysis
  python main.py --debug                              # Write JSON instead of JSONL
  python main.py stats                                # Print summary from existing output files

Prompt strategies:
  zero_shot        Minimal instructions, no examples (default)
  few_shot         Detailed instructions + one worked example per category
  chain_of_thought Explicit reasoning steps before generating output

Output is written to output/<batch-label>/ so every run is isolated.
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

def quick_stats(output_dir: Path, batch_label: str | None = None) -> None:
    """Print summaries from an existing run.

    If batch_label is given, reads from output_dir/batch_label/.
    Otherwise lists all available batch directories under output_dir.
    """
    if batch_label:
        run_dir = output_dir / batch_label
        if not run_dir.exists():
            print(f"Batch directory not found: {run_dir}")
            available = [d.name for d in sorted(output_dir.iterdir()) if d.is_dir()]
            if available:
                print(f"Available batches: {', '.join(available)}")
            return
        for name, fname in [
            ("Validation summary", "validation_summary.json"),
            ("Analysis report", "analysis_report.json"),
            ("Before/after comparison", "corrected/before_after_comparison.json"),
            ("Benchmark report", "benchmark_report.json"),
        ]:
            path = run_dir / fname
            if path.exists():
                data = json.loads(path.read_text())
                print(f"\n── {name} ({path}) ──")
                print(json.dumps(data, indent=2))
            else:
                print(f"\n── {name}: not found ({path}) ──")
    else:
        batches = sorted([d for d in output_dir.iterdir() if d.is_dir()]) if output_dir.exists() else []
        if not batches:
            print(f"No batch directories found under {output_dir}.")
            print("Run the pipeline first, or use --batch-label <name> to target a specific run.")
            return
        print(f"Available batches in {output_dir}:")
        for d in batches:
            files = [f.name for f in sorted(d.iterdir()) if f.is_file()]
            print(f"  {d.name}  ({len(files)} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Home DIY Repair Q&A Synthetic Data Pipeline — all 7 phases"
    )
    parser.add_argument("command", nargs="?", default=None, help="Optional subcommand: 'stats'")
    parser.add_argument("--samples", type=int, default=50, help="Q&A pairs to generate (default: 50)")
    parser.add_argument("--generation-model", type=str, default=None, dest="generation_model", help="Generation model override (default: LLM_MODEL from .env)")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM-as-Judge model override for Phases 3, 4, 7 (default: LLM_JUDGE_MODEL from .env, falls back to --model)")
    parser.add_argument("--phase", type=str, default="1-7", help="Phase range to run, e.g. '1-5', '6', '7' (default: 1-7)")
    parser.add_argument("--prompt-strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "chain_of_thought"],
                        help="Prompt strategy for Phase 1 (default: zero_shot)")
    parser.add_argument("--batch-label", type=str, default=None,
                        help="Human-readable label for this run, used as output subdirectory name. "
                             "Defaults to '<strategy>-<timestamp>'.")
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory (default: output)")
    parser.add_argument("--debug", action="store_true", help="Write output as pretty-printed JSON instead of JSONL")

    args = parser.parse_args()

    base_output = Path(args.output_dir)

    if args.command == "stats":
        quick_stats(base_output, batch_label=args.batch_label)
        return

    # Resolve models from CLI or config
    from config import get_settings
    settings = get_settings()
    generation_model = args.generation_model or settings.generation_model
    judge_model = args.judge_model or settings.judge_model

    strategy = args.prompt_strategy
    batch_label = args.batch_label or f"{strategy}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Each run gets its own subdirectory so runs never overwrite each other
    output_dir = base_output / batch_label
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = output_dir / "corrected"

    phase_start, phase_end = _parse_phase_range(args.phase)

    _banner("HOME DIY REPAIR Q&A SYNTHETIC DATA PIPELINE")
    print(f"Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Generation model: {generation_model}")
    print(f"Judge model     : {judge_model}")
    print(f"LLM base        : {settings.base_url}")
    print(f"Samples         : {args.samples}")
    print(f"Phases          : {phase_start}–{phase_end}")
    print(f"Prompt strategy : {strategy}")
    print(f"Batch label     : {batch_label}")
    print(f"Output dir      : {output_dir.absolute()}")

    generation_results = None
    valid_results = None

    # ── Phase 1: Generation ───────────────────────────────────────────────
    if phase_start <= 1 <= phase_end:
        _section("PHASE 1 — Generation")
        from phase1_generation import run_generation_phase
        generation_results = run_generation_phase(
            num_samples=args.samples,
            generation_model=generation_model,
            output_dir=output_dir,
            strategy=strategy,
            batch_label=batch_label,
            debug=args.debug,
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
        run_failure_labeling_phase(valid_results, judge_model, output_dir)

    # ── Phase 4: Quality Evaluation ───────────────────────────────────────
    if phase_start <= 4 <= phase_end:
        _section("PHASE 4 — Quality Evaluation (LLM-as-Judge, 8 dimensions)")
        from phase2_validation import load_valid_data
        from phase4_quality_eval import run_quality_eval_phase
        if valid_results is None:
            valid_results = load_valid_data(output_dir)
        run_quality_eval_phase(valid_results, judge_model, output_dir)

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
            generation_model=generation_model,
            judge_model=judge_model,
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
            judge_model=judge_model,
            output_dir=output_dir,
            num_samples=50,
        )
        # Update benchmark comparison visualization
        from phase5_analysis import run_analysis_phase
        import pandas as pd
        from schema import QUALITY_DIMENSION_FIELDS as QUALITY_DIM_NAMES
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
    main()
