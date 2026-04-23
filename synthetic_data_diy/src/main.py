"""
Main CLI orchestrator for the Home DIY Repair Q&A Synthetic Data Pipeline.

Usage:
  python main.py                                       # Full pipeline, zero_shot, 50 samples
  python main.py --samples 20 --prompt-strategy few_shot --batch-label few-shot-run1
  python main.py --prompt-strategy chain_of_thought --batch-label cot-run1
  python main.py --phase 1 --samples 10              # Phase 1 only
  python main.py --phase 1-6                          # Stop after analysis
  python main.py --phase 7 --batch-label my-run-1    # Correction only (requires phases 4-5 output)
  python main.py stats                                # Print summary from existing output files
  python main.py compare                              # Cross-strategy comparison charts (output/_comparison/)
  python main.py agreement --batch-label my-run       # Phase A: human/LLM agreement on 6 quality dims
  python main.py mock                                 # Mock pipeline seeded from HF benchmark (no API calls)
  python main.py mock --batch-label baseline-mock --num-samples 50 --seed 42

Pipeline phases (in order):
  1  Generation             — LLM generates Q&A pairs
  2  Structural Validation  — Pydantic schema gate
  3  Benchmark Calibration  — judge verified against real-world dataset BEFORE use
  4  Failure Labeling       — LLM-as-Judge: 6 binary failure modes
  5  Quality Evaluation     — LLM-as-Judge: 9 quality dimensions
  6  Analysis               — visualizations + apples-to-apples benchmark gap
  7  Prompt Correction      — data-driven re-run with iterative improvement loop

Prompt strategies:
  zero_shot        Minimal instructions, no examples (default)
  few_shot         Detailed instructions + one worked example per category
  chain_of_thought Explicit reasoning steps before generating output
  human_feedback   Corrected prompts targeting observed failure modes (RLHF baseline)

Output is written to output/<batch-label>/ so every run is isolated.
"""

import argparse
import json
import sys
import time
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


def _section(text: str) -> float:
    """Print a phase section header with timestamp. Returns start time (monotonic)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─'*50}\n{text}  [{ts}]\n{'─'*50}")
    return time.monotonic()


def _phase_done(t0: float, summary: str = "") -> None:
    elapsed = time.monotonic() - t0
    mins, secs = divmod(int(elapsed), 60)
    duration = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    suffix = f"  →  {summary}" if summary else ""
    print(f"  ✓ done in {duration}{suffix}")


# ---------------------------------------------------------------------------
# stats subcommand
# ---------------------------------------------------------------------------

def _phase_status(run_dir: Path) -> dict:
    """Derive per-phase completion status and key metrics from a batch directory."""
    import pandas as pd

    files = {f.name for f in run_dir.rglob("*") if f.is_file()}

    ph1 = "generation_results.json" in files
    ph2 = "structurally_valid_qa_pairs.json" in files
    ph3 = "benchmark_report.json" in files

    failure_rate: float | None = None
    quality_rate: float | None = None
    ph5_corrupted = False

    ph4 = "failure_labeled_data.csv" in files
    if ph4:
        try:
            fdf = pd.read_csv(run_dir / "failure_labeled_data.csv")
            failure_rate = fdf["overall_failure"].mean()
        except Exception:
            pass

    ph5 = "quality_eval_data.csv" in files
    if ph5:
        try:
            qdf = pd.read_csv(run_dir / "quality_eval_data.csv")
            dim_cols = [c for c in qdf.columns if c not in ("trace_id", "category", "overall_quality_pass")]
            all_zero_frac = (qdf[dim_cols] == 0).all(axis=1).mean()
            if all_zero_frac >= 0.4:
                ph5_corrupted = True
                ph5 = False  # treat as incomplete
            else:
                quality_rate = qdf["overall_quality_pass"].mean()
        except Exception:
            pass

    ph6 = "analysis_report.json" in files
    ph7 = (run_dir / "corrected" / "before_after_comparison.json").exists()

    return {
        "ph1": ph1, "ph2": ph2, "ph3": ph3,
        "ph4": ph4, "ph5": ph5, "ph5_corrupted": ph5_corrupted,
        "ph6": ph6, "ph7": ph7,
        "failure_rate": failure_rate,
        "quality_rate": quality_rate,
    }


def _fmt_check(ok: bool | None, corrupted: bool = False) -> str:
    if corrupted:
        return " ✗!"
    if ok is None:
        return "  — "
    return "  ✓ " if ok else "  — "


def quick_stats(output_dir: Path, batch_label: str | None = None) -> None:
    """Print run status.

    With --batch-label: dumps raw JSON reports for that run.
    Without: prints a phase-completion table across all runs.
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
            ("Benchmark report", "benchmark_report.json"),
            ("Analysis report", "analysis_report.json"),
            ("Before/after comparison", "corrected/before_after_comparison.json"),
        ]:
            path = run_dir / fname
            if path.exists():
                data = json.loads(path.read_text())
                print(f"\n── {name} ({path}) ──")
                print(json.dumps(data, indent=2))
            else:
                print(f"\n── {name}: not found ({path}) ──")
        return

    batches = (
        sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])
        if output_dir.exists() else []
    )
    if not batches:
        print(f"No batch directories found under {output_dir}.")
        print("Run the pipeline first, or use --batch-label <name> to inspect a specific run.")
        return

    from baselines import active_labels as _active_labels
    try:
        _active = _active_labels()
    except Exception:
        _active = set()

    col_label = max(len(d.name) for d in batches) + 2  # +2 for ★ marker
    col_label = max(col_label, 22)

    header = f"{'Batch':<{col_label}}  Ph1  Ph2  Ph3  Ph4  Ph5  Ph6  Ph7   Failure   Quality"
    print(f"\n{header}")
    print("─" * len(header))

    for d in batches:
        s = _phase_status(d)
        fail_str = f"{s['failure_rate']*100:>5.1f}%" if s["failure_rate"] is not None else "    —  "
        qual_str = f"{s['quality_rate']*100:>5.1f}%" if s["quality_rate"] is not None else "    —  "
        ph5_mark = " ✗!" if s["ph5_corrupted"] else ("  ✓ " if s["ph5"] else "  — ")
        marker = "★ " if d.name in _active else "  "
        label = f"{marker}{d.name}"
        row = (
            f"{label:<{col_label}}"
            f"{_fmt_check(s['ph1'])}"
            f"{_fmt_check(s['ph2'])}"
            f"{_fmt_check(s['ph3'])}"
            f"{_fmt_check(s['ph4'])}"
            f"{ph5_mark}"
            f"{_fmt_check(s['ph6'])}"
            f"{_fmt_check(s['ph7'])}"
            f"   {fail_str}   {qual_str}"
        )
        print(row)

    print(f"\n  ★  = active baseline (baselines.yaml)")
    print(f"  ✓  = complete    —  = not run    ✗! = corrupted (>40% all-zero rows)")
    print(f"  Use 'python main.py stats --batch-label <name>' for full JSON reports.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase execution helper
# ---------------------------------------------------------------------------

def _run_phases(
    phase_start: int,
    phase_end: int,
    batch_label: str,
    strategy: str,
    generation_model: str,
    judge_model: str,
    num_samples: int,
    base_output: Path,
    max_iterations: int,
) -> dict[str, float]:
    """Run the requested phase range for a single batch. Returns phase timings."""
    output_dir = base_output / batch_label
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = output_dir / "corrected"

    # mock strategy always uses the benchmark generator regardless of generation_model
    _gen_model = "mock" if strategy == "mock" else generation_model

    generation_results = None
    valid_results = None
    phase_timings: dict[str, float] = {}

    # ── Phase 1: Generation ───────────────────────────────────────────────
    if phase_start <= 1 <= phase_end:
        t0 = _section("PHASE 1 — Generation")
        from phase1_generation import run_generation_phase
        generation_results = run_generation_phase(
            num_samples=num_samples,
            generation_model=_gen_model,
            output_dir=output_dir,
            strategy=strategy,
            batch_label=batch_label,
            output_base=base_output,
        )
        parsed = sum(1 for r in generation_results if r.parse_error is None)
        phase_timings["1 Generation"] = time.monotonic() - t0
        _phase_done(t0, f"{parsed}/{len(generation_results)} parsed ({parsed/len(generation_results)*100:.0f}%)")

    # ── Phase 2: Structural Validation ───────────────────────────────────
    if phase_start <= 2 <= phase_end:
        t0 = _section("PHASE 2 — Structural Validation")
        from phase1_generation import load_generation_results
        from phase2_validation import run_validation_phase
        if generation_results is None:
            generation_results = load_generation_results(output_dir)
            print(f"Loaded {len(generation_results)} generation results from disk.")
        valid_results, summary = run_validation_phase(generation_results, output_dir)
        phase_timings["2 Validation"] = time.monotonic() - t0
        _phase_done(t0, f"{summary.total_valid}/{summary.total_generated} valid ({summary.validation_rate*100:.0f}%)")

    # ── Phase 3: Benchmark Calibration ───────────────────────────────────
    if phase_start <= 3 <= phase_end:
        t0 = _section("PHASE 3 — Benchmark Calibration (judge verification)")
        from phase3_benchmark import run_benchmark_phase
        bench = run_benchmark_phase(
            judge_model=judge_model,
            output_dir=output_dir,
            num_samples=50,
            base_output=base_output,
        )
        phase_timings["3 Benchmark"] = time.monotonic() - t0
        _phase_done(t0, f"pass rate {bench.benchmark_quality_pass_rate*100:.1f}% ({'cached' if time.monotonic()-t0 < 5 else 'calibration passed' if bench.calibration_passed else 'WARNING: calibration failed'})")

    # ── Phase 4: Failure Labeling ─────────────────────────────────────────
    if phase_start <= 4 <= phase_end:
        t0 = _section("PHASE 4 — Failure Labeling (LLM-as-Judge, 6 modes)")
        from phase2_validation import load_valid_data
        from phase4_failure_labeling import run_failure_labeling_phase
        if valid_results is None:
            valid_results = load_valid_data(output_dir)
        df4 = run_failure_labeling_phase(valid_results, judge_model, output_dir)
        phase_timings["4 Failure Label"] = time.monotonic() - t0
        _phase_done(t0, f"overall failure rate {df4['overall_failure'].mean()*100:.1f}%")

    # ── Phase 5: Quality Evaluation ───────────────────────────────────────
    if phase_start <= 5 <= phase_end:
        t0 = _section("PHASE 5 — Quality Evaluation (LLM-as-Judge, 9 dimensions)")
        from phase2_validation import load_valid_data
        from phase5_quality_eval import run_quality_eval_phase
        if valid_results is None:
            valid_results = load_valid_data(output_dir)
        df5 = run_quality_eval_phase(valid_results, judge_model, output_dir)
        phase_timings["5 Quality Eval"] = time.monotonic() - t0
        _phase_done(t0, f"overall quality pass rate {df5['overall_quality_pass'].mean()*100:.1f}%")

    # ── Phase 6: Analysis & Visualizations ───────────────────────────────
    if phase_start <= 6 <= phase_end:
        t0 = _section("PHASE 6 — Failure & Quality Analysis")
        from phase6_analysis import run_analysis_phase
        run_analysis_phase(output_dir=output_dir)
        phase_timings["6 Analysis"] = time.monotonic() - t0
        _phase_done(t0)

    # ── Phase 7: Prompt Correction ────────────────────────────────────────
    if phase_start <= 7 <= phase_end:
        t0 = _section("PHASE 7 — Prompt Correction & Re-evaluation")
        from phase7_correction import run_correction_phase
        run_correction_phase(
            num_samples=num_samples,
            generation_model=_gen_model,
            judge_model=judge_model,
            baseline_dir=output_dir,
            corrected_dir=corrected_dir,
            max_iterations=max_iterations,
        )
        from phase6_analysis import run_analysis_phase
        print("\nUpdating visualizations with corrected data...")
        run_analysis_phase(output_dir=output_dir, corrected_dir=corrected_dir)
        phase_timings["7 Correction"] = time.monotonic() - t0
        _phase_done(t0)

    return phase_timings


def _print_phase_timings(phase_timings: dict[str, float], output_dir: Path) -> None:
    _banner("PIPELINE COMPLETE")
    print(f"Output dir: {output_dir.absolute()}")
    if phase_timings:
        print("\nPhase timings:")
        total = sum(phase_timings.values())
        for name, elapsed in phase_timings.items():
            mins, secs = divmod(int(elapsed), 60)
            dur = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            bar = "█" * int(elapsed / total * 20)
            print(f"  {name:<18} {dur:>6}  {bar}")
        total_mins, total_secs = divmod(int(total), 60)
        print(f"  {'Total':<18} {total_mins}m {total_secs:02d}s")
    print("\nOutput files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Home DIY Repair Q&A Synthetic Data Pipeline — all 7 phases"
    )
    parser.add_argument("command", nargs="?", default=None, help="Optional subcommand: 'stats', 'compare', 'agreement', 'mock'")
    parser.add_argument("--samples", type=int, default=50, help="Q&A pairs to generate (default: 50)")
    parser.add_argument("--generation-model", type=str, default=None, dest="generation_model",
                        help="Generation model override (default: LLM_MODEL from .env)")
    parser.add_argument("--judge-model", type=str, default=None, dest="judge_model",
                        help="LLM-as-Judge model override for Phases 3, 4, 5 (default: LLM_JUDGE_MODEL from .env)")
    parser.add_argument("--phase", type=str, default="1-7",
                        help="Phase range to run, e.g. '1-6', '3', '7' (default: 1-7)")
    parser.add_argument("--prompt-strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "chain_of_thought", "human_feedback", "mock"],
                        help="Prompt strategy for Phase 1 (default: zero_shot)")
    parser.add_argument("--batch-label", type=str, default=None,
                        help="Human-readable label for this run, used as output subdirectory name. "
                             "Defaults to '<strategy>-<timestamp>'. Ignored when --all-active is set.")
    parser.add_argument("--all-active", action="store_true", dest="all_active",
                        help="Run the requested phases for every active baseline in baselines.yaml sequentially.")
    parser.add_argument("--max-iterations", type=int, default=3, dest="max_iterations",
                        help="Maximum correction iterations in Phase 7 (default: 3)")
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory (default: output)")
    parser.add_argument("--num-samples", type=int, default=50, dest="num_samples",
                        help="Benchmark rows to seed (mock subcommand, default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Bernoulli draws (mock subcommand, default: 42)")
    parser.add_argument("--skip-human-labels", action="store_true", dest="skip_human_labels",
                        help="Skip generating mock human_labels.json (mock subcommand)")

    args = parser.parse_args()

    base_output = Path(args.output_dir)

    if args.command == "stats":
        quick_stats(base_output, batch_label=args.batch_label)
        return

    if args.command == "compare":
        from baselines import active_labels
        from phase6_analysis import run_multi_batch_comparison
        run_multi_batch_comparison(base_output, labels=active_labels())
        return

    if args.command == "agreement":
        if not args.batch_label:
            print("Error: --batch-label is required for the 'agreement' subcommand.")
            print("  python main.py agreement --batch-label baseline-run")
            return
        from agreement import run_agreement
        run_agreement(batch_label=args.batch_label, output_dir=base_output)
        return

    if args.command == "mock":
        from mock_seeder import run_mock_pipeline
        mock_label = args.batch_label or "baseline-mock"
        _banner("HOME DIY REPAIR Q&A — MOCK PIPELINE (no API calls)")
        print(f"Batch label  : {mock_label}")
        print(f"Num samples  : {args.num_samples}")
        print(f"Seed         : {args.seed}")
        print(f"Human labels : {'skipped' if args.skip_human_labels else 'generated'}")
        t0 = time.monotonic()
        out_dir = run_mock_pipeline(
            batch_label=mock_label,
            num_samples=args.num_samples,
            seed=args.seed,
            skip_human_labels=args.skip_human_labels,
            output_dir_base=base_output,
        )
        elapsed = time.monotonic() - t0
        _banner("MOCK PIPELINE COMPLETE")
        print(f"Output dir: {out_dir.absolute()}")
        print(f"Total time: {int(elapsed)}s")
        print("\nOutput files:")
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out_dir)}")
        return

    # Resolve models from CLI or config
    from config import get_settings
    settings = get_settings()
    generation_model = args.generation_model or settings.generation_model
    judge_model = args.judge_model or settings.judge_model

    phase_start, phase_end = _parse_phase_range(args.phase)

    # ── --all-active: loop through every active baseline sequentially ─────
    if args.all_active:
        from baselines import active_baselines
        baselines = active_baselines()
        if not baselines:
            print("No active baselines found in baselines.yaml.")
            return
        labels = ", ".join(b.label for b in baselines)
        _banner(f"HOME DIY REPAIR Q&A — ALL ACTIVE BASELINES ({len(baselines)})")
        print(f"Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Generation model: {generation_model} ({settings.base_url})")
        print(f"Judge model     : {judge_model} ({settings.judge_base_url})")
        print(f"Samples         : {args.samples}")
        print(f"Phases          : {phase_start}–{phase_end}")
        print(f"Baselines       : {labels}")

        all_timings: dict[str, float] = {}
        for i, baseline in enumerate(baselines):
            print(f"\n{'━'*60}")
            print(f"  Baseline {i+1}/{len(baselines)}: {baseline.label}  (strategy: {baseline.strategy})")
            print(f"{'━'*60}")
            timings = _run_phases(
                phase_start=phase_start,
                phase_end=phase_end,
                batch_label=baseline.label,
                strategy=baseline.strategy,
                generation_model=generation_model,
                judge_model=judge_model,
                num_samples=args.samples,
                base_output=base_output,
                max_iterations=args.max_iterations,
            )
            for k, v in timings.items():
                all_timings[f"{baseline.label}/{k}"] = v

        _banner(f"ALL ACTIVE BASELINES COMPLETE ({len(baselines)} runs)")
        total = sum(all_timings.values())
        total_mins, total_secs = divmod(int(total), 60)
        print(f"Total wall time: {total_mins}m {total_secs:02d}s across {len(baselines)} baselines")
        return

    # ── Single baseline run ───────────────────────────────────────────────
    strategy = args.prompt_strategy
    batch_label = args.batch_label or f"{strategy}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _banner("HOME DIY REPAIR Q&A SYNTHETIC DATA PIPELINE")
    print(f"Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Generation model: {generation_model} ({settings.base_url})")
    print(f"Judge model     : {judge_model} ({settings.judge_base_url})")
    print(f"Samples         : {args.samples}")
    print(f"Phases          : {phase_start}–{phase_end}")
    print(f"Prompt strategy : {strategy}")
    print(f"Batch label     : {batch_label}")
    print(f"Max iterations  : {args.max_iterations} (Phase 7)")
    print(f"Output dir      : {(base_output / batch_label).absolute()}")

    phase_timings = _run_phases(
        phase_start=phase_start,
        phase_end=phase_end,
        batch_label=batch_label,
        strategy=strategy,
        generation_model=generation_model,
        judge_model=judge_model,
        num_samples=args.samples,
        base_output=base_output,
        max_iterations=args.max_iterations,
    )
    _print_phase_timings(phase_timings, base_output / batch_label)


if __name__ == "__main__":
    main()
