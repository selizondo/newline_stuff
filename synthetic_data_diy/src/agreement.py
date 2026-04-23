"""
Phase A: Judge–Human Agreement
Computes per-dimension agreement between human labels and LLM quality eval scores.
Flags dimensions below 80% agreement so prompt/criteria can be adjusted before
trusting the judge on the full dataset.

Usage:
  python agreement.py --batch-label baseline-run
  python agreement.py --batch-label baseline-run --threshold 0.75
"""

import argparse
import json
from pathlib import Path

from schema import HUMAN_TO_LLM

AGREEMENT_THRESHOLD = 0.80  # flag any dimension below this

HUMAN_LABELS: list[str] = [
    "D1 — Answer Completeness",
    "D2 — Safety Specificity",
    "D3 — Tool Realism",
    "D4 — Scope Appropriateness",
    "D5 — Context Clarity",
    "D6 — Tip Usefulness",
]

_HUMAN_KEY_TO_LABEL: dict[str, str] = {
    "answer_completeness": "D1 — Answer Completeness",
    "safety_specificity": "D2 — Safety Specificity",
    "tool_realism": "D3 — Tool Realism",
    "scope_appropriateness": "D4 — Scope Appropriateness",
    "context_clarity": "D5 — Context Clarity",
    "tip_usefulness": "D6 — Tip Usefulness",
}


def run_agreement(batch_label: str, output_dir: Path, threshold: float = AGREEMENT_THRESHOLD) -> dict:
    run_dir = output_dir / batch_label
    human_path = run_dir / "human_labels.json"
    llm_path = run_dir / "quality_eval_data.json"

    if not human_path.exists():
        raise FileNotFoundError(
            f"Human labels not found: {human_path}\n"
            "Run python human_labeler.py --batch-label {batch_label} first."
        )
    if not llm_path.exists():
        raise FileNotFoundError(
            f"LLM quality eval not found: {llm_path}\n"
            "Run Phase 5 first."
        )

    human_records: list[dict] = json.loads(human_path.read_text())
    llm_records: list[dict] = json.loads(llm_path.read_text())

    # Index by trace_id
    llm_by_id = {r["trace_id"]: r for r in llm_records}

    # Join — only items labeled by both
    joined: list[tuple[dict, dict]] = []
    for h in human_records:
        tid = h["trace_id"]
        if tid in llm_by_id:
            joined.append((h, llm_by_id[tid]))

    if not joined:
        raise ValueError(
            f"No trace_ids overlap between human labels ({len(human_records)} items) "
            f"and LLM quality eval ({len(llm_records)} items). "
            "Ensure both were generated from the same batch."
        )

    print(f"\nBatch            : {batch_label}")
    print(f"Human labels     : {len(human_records)} items")
    print(f"LLM quality eval : {len(llm_records)} items")
    print(f"Matched (joined) : {len(joined)} items")
    print(f"Agreement threshold: {threshold*100:.0f}%\n")

    # Compute per-dimension agreement
    dim_results: dict[str, dict] = {}
    for human_key, llm_key in HUMAN_TO_LLM.items():
        agreements = []
        for h, l in joined:
            h_val = h.get(human_key)
            l_val = l.get(llm_key)
            if h_val is None or l_val is None:
                continue
            agreements.append(int(h_val) == int(l_val))

        if not agreements:
            continue

        rate = sum(agreements) / len(agreements)
        n_agree = sum(agreements)
        n_total = len(agreements)

        # True positive rate (both say 1), true negative rate (both say 0)
        tp = sum(1 for h, l in joined if h.get(human_key) == 1 and l.get(llm_key) == 1)
        tn = sum(1 for h, l in joined if h.get(human_key) == 0 and l.get(llm_key) == 0)
        fp = sum(1 for h, l in joined if h.get(human_key) == 0 and l.get(llm_key) == 1)
        fn = sum(1 for h, l in joined if h.get(human_key) == 1 and l.get(llm_key) == 0)

        dim_results[human_key] = {
            "human_key": human_key,
            "llm_key": llm_key,
            "label": _HUMAN_KEY_TO_LABEL[human_key],
            "agreement_rate": round(rate, 4),
            "n_agreed": n_agree,
            "n_total": n_total,
            "meets_threshold": rate >= threshold,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,   # LLM says pass, human says fail
            "false_negative": fn,   # LLM says fail, human says pass
        }

    # Print table
    print(f"{'Dimension':<35} {'Agreement':>9}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}  Status")
    print("─" * 75)
    all_met = True
    for res in dim_results.values():
        status = "✓" if res["meets_threshold"] else "✗ BELOW THRESHOLD"
        if not res["meets_threshold"]:
            all_met = False
        print(
            f"  {res['label']:<33} {res['agreement_rate']*100:>8.1f}%"
            f"  {res['true_positive']:>4} {res['true_negative']:>4}"
            f"  {res['false_positive']:>4} {res['false_negative']:>4}  {status}"
        )

    overall_agreement = sum(r["agreement_rate"] for r in dim_results.values()) / len(dim_results)
    print("─" * 75)
    print(f"  {'Mean agreement':<33} {overall_agreement*100:>8.1f}%")

    if all_met:
        print(f"\n✓ All dimensions meet the {threshold*100:.0f}% agreement threshold — judge is trustworthy.")
    else:
        failing = [r["label"] for r in dim_results.values() if not r["meets_threshold"]]
        print(f"\n✗ {len(failing)} dimension(s) below threshold: {', '.join(failing)}")
        print("  → Review the corresponding quality_dimensions/*.yaml and adjust criteria,")
        print("    then re-run Phase 5 on the same batch and re-check agreement.")

    report = {
        "batch_label": batch_label,
        "n_human_labels": len(human_records),
        "n_llm_labels": len(llm_records),
        "n_matched": len(joined),
        "threshold": threshold,
        "all_dimensions_meet_threshold": all_met,
        "overall_mean_agreement": round(overall_agreement, 4),
        "dimensions": dim_results,
    }

    report_path = run_dir / "agreement_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved → {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A — compute human/LLM agreement on 6 quality dimensions"
    )
    parser.add_argument("--batch-label", required=True, dest="batch_label",
                        help="Batch label to evaluate (must have human_labels.json and quality_eval_data.json)")
    parser.add_argument("--threshold", type=float, default=AGREEMENT_THRESHOLD,
                        help=f"Agreement threshold to flag dimensions (default: {AGREEMENT_THRESHOLD})")
    parser.add_argument("--output-dir", type=str, default="output", dest="output_dir",
                        help="Base output directory (default: output)")
    args = parser.parse_args()

    run_agreement(
        batch_label=args.batch_label,
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
