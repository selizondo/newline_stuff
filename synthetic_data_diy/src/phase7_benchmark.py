"""
Phase 7: Benchmark Comparison
Loads the dipenbhuva/home-diy-repair-qa HuggingFace dataset, evaluates 50+ items
using the Phase 4 quality judge, and produces a calibration + gap report.
"""

import json
import random
import time
from pathlib import Path

import pandas as pd

from config import get_settings
from models import BenchmarkReport, QAPair, QualityEvalResult, ValidatedResult
from phase4_quality_eval import QUALITY_DIM_NAMES, QUALITY_DIMENSIONS, QualityEvaluator

BENCHMARK_DATASET = "dipenbhuva/home-diy-repair-qa"
MIN_BENCHMARK_SAMPLES = 50


def _map_benchmark_row(row: dict) -> QAPair | None:
    """
    Map a HuggingFace benchmark row to QAPair.
    The dataset columns may vary; we attempt common column names and fall back gracefully.
    """
    try:
        question = row.get("question") or row.get("Question") or ""
        answer = row.get("answer") or row.get("Answer") or ""
        equipment_problem = (
            row.get("equipment_problem")
            or row.get("problem")
            or row.get("Problem")
            or question[:100]
        )
        tools = row.get("tools_required") or row.get("tools") or row.get("Tools") or []
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",") if t.strip()]
        if not tools:
            tools = ["standard household tools"]

        steps = row.get("steps") or row.get("Steps") or []
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split("\n") if s.strip()]
        if not steps:
            # Split answer into sentences as fallback steps
            import re
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            steps = sentences[:5] or ["Inspect the issue", "Apply the fix", "Test the repair"]

        safety_info = row.get("safety_info") or row.get("safety") or row.get("Safety") or "Follow standard safety precautions."
        tips = row.get("tips") or row.get("Tips") or "Consult a professional if unsure."

        return QAPair(
            question=question[:500] if question else "No question provided",
            answer=answer[:2000] if answer else "No answer provided",
            equipment_problem=str(equipment_problem)[:200],
            tools_required=tools[:10],
            steps=steps[:15],
            safety_info=str(safety_info)[:500],
            tips=[str(tips)[:500]] if not isinstance(tips, list) else [str(t)[:500] for t in tips],
        )
    except Exception:
        return None


def run_benchmark_phase(
    model: str,
    output_dir: Path,
    num_samples: int = MIN_BENCHMARK_SAMPLES,
) -> BenchmarkReport:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print(f"Loading benchmark dataset: {BENCHMARK_DATASET} ...")
    try:
        dataset = load_dataset(BENCHMARK_DATASET, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark dataset: {e}")

    total_rows = len(dataset)
    print(f"Dataset loaded: {total_rows} rows. Sampling {num_samples} items...")

    indices = random.sample(range(total_rows), min(num_samples, total_rows))
    sampled_rows = [dataset[i] for i in indices]

    valid_results: list[ValidatedResult] = []
    skipped = 0
    for i, row in enumerate(sampled_rows):
        qa = _map_benchmark_row(row)
        if qa is None:
            skipped += 1
            continue
        import uuid
        valid_results.append(
            ValidatedResult(
                trace_id=f"benchmark-{uuid.uuid4()}",
                category=row.get("category", "general_home_repair"),
                qa_pair=qa,
            )
        )

    if skipped:
        print(f"  Skipped {skipped} rows that could not be mapped to QAPair schema.")

    print(f"Evaluating {len(valid_results)} benchmark items on 8 quality dimensions...")
    evaluator = QualityEvaluator(model=model)
    eval_results: list[QualityEvalResult] = []

    for i, result in enumerate(valid_results):
        print(f"  [{i+1}/{len(valid_results)}] Evaluating benchmark item... ", end="", flush=True)
        eval_result = evaluator.evaluate(result)
        eval_results.append(eval_result)
        status = "PASS" if eval_result.overall_quality_pass else "FAIL"
        print(status)
        time.sleep(get_settings().rate_limit_delay)

    bench_df = pd.DataFrame([r.model_dump() for r in eval_results])
    bench_df.to_csv(output_dir / "benchmark_eval.csv", index=False)
    bench_df.to_json(output_dir / "benchmark_eval.json", orient="records", indent=2)

    benchmark_pass_rate = float(bench_df["overall_quality_pass"].mean())
    calibration_passed = benchmark_pass_rate >= 0.95

    benchmark_dim_rates = {d: float(bench_df[d].mean()) for d in QUALITY_DIM_NAMES if d in bench_df.columns}

    # Load generated quality data for gap analysis
    generated_csv = output_dir / "quality_eval_data.csv"
    generated_dim_rates: dict[str, float] = {}
    if generated_csv.exists():
        gen_df = pd.read_csv(generated_csv)
        generated_dim_rates = {d: float(gen_df[d].mean()) for d in QUALITY_DIM_NAMES if d in gen_df.columns}

    gap = {
        d: round(benchmark_dim_rates.get(d, 0) - generated_dim_rates.get(d, 0), 4)
        for d in QUALITY_DIM_NAMES
    }
    overall_gap = round(
        benchmark_pass_rate - (
            sum(generated_dim_rates.values()) / len(generated_dim_rates)
            if generated_dim_rates else 0
        ),
        4,
    )

    report = BenchmarkReport(
        benchmark_samples_evaluated=len(eval_results),
        benchmark_quality_pass_rate=benchmark_pass_rate,
        calibration_passed=calibration_passed,
        generated_vs_benchmark=gap,
        overall_gap=overall_gap,
    )

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2))

    print("\n" + "=" * 50)
    print("BENCHMARK CALIBRATION REPORT")
    print("=" * 50)
    print(f"Benchmark items evaluated : {report.benchmark_samples_evaluated}")
    print(f"Benchmark pass rate       : {benchmark_pass_rate*100:.1f}%")
    print(f"Calibration passed (≥95%) : {'YES ✓' if calibration_passed else 'NO ✗'}")
    print(f"Overall quality gap       : {overall_gap*100:+.1f}pp (benchmark − generated)")
    print("\nPer-dimension gap (benchmark − generated):")
    for dim, g in gap.items():
        arrow = "↑" if g > 0 else ("→" if g == 0 else "↓")
        print(f"  {arrow} {dim.replace('_', ' ').title()}: {g*100:+.1f}pp")
    print(f"\nSaved → {report_path}")
    return report
