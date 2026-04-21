"""
Phase 3: Judge Calibration via Benchmark
Loads the dipenbhuva/home-diy-repair-qa HuggingFace dataset, evaluates 50+ items
using the quality judge, and verifies the judge is trustworthy BEFORE it is used
to score generated data in Phases 4 and 5.

Running calibration early means any systematic judge miscalibration is caught
before it silently distorts all downstream quality metrics and correction targets.
"""

import json
import time
import uuid
from pathlib import Path

import pandas as pd

from config import get_settings
from schema import BenchmarkReport, QAPair, QualityEvalResult, ValidatedResult
from schema import QUALITY_DIMENSION_FIELDS as QUALITY_DIM_NAMES

BENCHMARK_DATASET = "dipenbhuva/home-diy-repair-qa"
MIN_BENCHMARK_SAMPLES = 50
CALIBRATION_PASS_THRESHOLD = 0.80  # judge must score ≥ 80% of benchmark items as pass

_QA_FIELDS = frozenset(QAPair.model_fields)


def _map_benchmark_row(row: dict) -> QAPair | None:
    """Map a HuggingFace benchmark row to QAPair. Dataset field names match QAPair directly."""
    try:
        return QAPair.model_validate({k: row[k] for k in _QA_FIELDS if k in row})
    except Exception:
        return None


def _check_judge_connectivity(judge_model: str) -> None:
    """Send a minimal test prompt to the judge endpoint and raise clearly if unreachable.

    Prevents silently scoring every benchmark item as 0 when the judge endpoint
    (e.g. local Ollama) is down — a failure mode that would invalidate calibration.
    """
    from llm_client import chat_complete
    test_messages = [
        {"role": "system", "content": "Respond with exactly one digit: 1"},
        {"role": "user", "content": "Ping. Reply with 1."},
    ]
    try:
        response = chat_complete(test_messages, model=judge_model, temperature=0.0, max_tokens=5, use_judge_client=True)
        if not response.strip():
            raise RuntimeError("Judge returned an empty response to connectivity check.")
        print(f"  Judge connectivity OK (model: {judge_model}, response: '{response.strip()[:20]}')")
    except Exception as e:
        raise RuntimeError(
            f"Judge endpoint unreachable for model '{judge_model}'. "
            f"If using Ollama, ensure it is running ('ollama serve') and the model is pulled ('ollama pull {judge_model}'). "
            f"Error: {e}"
        ) from e


def run_benchmark_phase(
    judge_model: str,
    output_dir: Path,
    num_samples: int = MIN_BENCHMARK_SAMPLES,
) -> BenchmarkReport:
    """Evaluate benchmark items and verify judge calibration.

    Saves benchmark_eval.csv and benchmark_report.json.
    The benchmark_dimension_rates in the report are consumed by Phase 6 (analysis)
    for an apples-to-apples gap comparison against generated data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    # Import here to avoid circular dependency at module level
    from phase5_quality_eval import QualityEvaluator

    print(f"Loading benchmark dataset: {BENCHMARK_DATASET} ...")
    try:
        dataset = load_dataset(BENCHMARK_DATASET, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark dataset: {e}")

    total_rows = len(dataset)
    print(f"Dataset loaded: {total_rows} rows. Sampling {num_samples} items (stratified by category)...")

    encoded = dataset.class_encode_column("category")
    label_names = encoded.features["category"].names
    split = encoded.train_test_split(test_size=min(num_samples, total_rows), stratify_by_column="category")
    sampled_rows = [{**row, "category": label_names[row["category"]]} for row in split["test"]]

    from collections import Counter
    cat_counts = Counter(row.get("category", "unknown") for row in sampled_rows)
    print(f"  Sampled {len(sampled_rows)} items: " + ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items())))

    valid_results: list[ValidatedResult] = []
    skipped = 0
    for row in sampled_rows:
        qa = _map_benchmark_row(row)
        if qa is None:
            skipped += 1
            continue
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

    # Verify judge endpoint is reachable before processing all items.
    # If Ollama or the remote judge is down, every call silently defaults to 0
    # and the calibration report becomes meaningless.
    _check_judge_connectivity(judge_model)

    evaluator = QualityEvaluator(judge_model=judge_model)
    eval_results: list[QualityEvalResult] = []

    for i, result in enumerate(valid_results):
        print(f"  [{i+1}/{len(valid_results)}] Evaluating benchmark item... ", end="", flush=True)
        eval_result = evaluator.evaluate(result)
        eval_results.append(eval_result)
        print("PASS" if eval_result.overall_quality_pass else "FAIL")
        time.sleep(get_settings().judge_rate_limit_delay)

    bench_df = pd.DataFrame([r.model_dump() for r in eval_results])
    bench_df.to_csv(output_dir / "benchmark_eval.csv", index=False)
    bench_df.to_json(output_dir / "benchmark_eval.json", orient="records", indent=2)

    benchmark_pass_rate = float(bench_df["overall_quality_pass"].mean())
    calibration_passed = benchmark_pass_rate >= CALIBRATION_PASS_THRESHOLD
    benchmark_dimension_rates = {
        d: float(bench_df[d].mean()) for d in QUALITY_DIM_NAMES if d in bench_df.columns
    }

    report = BenchmarkReport(
        benchmark_samples_evaluated=len(eval_results),
        benchmark_quality_pass_rate=benchmark_pass_rate,
        calibration_passed=calibration_passed,
        benchmark_dimension_rates=benchmark_dimension_rates,
    )

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2))

    print("\n" + "=" * 50)
    print("BENCHMARK CALIBRATION REPORT")
    print("=" * 50)
    print(f"Benchmark items evaluated : {report.benchmark_samples_evaluated}")
    print(f"Benchmark pass rate       : {benchmark_pass_rate*100:.1f}%")
    print(f"Calibration passed (≥{CALIBRATION_PASS_THRESHOLD*100:.0f}%) : {'YES ✓' if calibration_passed else 'NO ✗'}")
    if not calibration_passed:
        print("  WARNING: Judge calibration failed. Quality scores in Phases 4-5 may be unreliable.")
        print("  Consider reviewing judge prompts in quality_dimensions/ before proceeding.")
    print("\nPer-dimension pass rates:")
    for dim, rate in benchmark_dimension_rates.items():
        print(f"  {dim.replace('_', ' ').title()}: {rate*100:.1f}%")
    print(f"\nSaved → {report_path}")
    return report
