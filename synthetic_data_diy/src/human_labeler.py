"""
Human Labeling CLI — Step 3

Walk a reviewer through each validated Q&A item and collect binary pass/fail
labels on the 6 spec quality dimensions. Saves incrementally so progress is
never lost on interrupt. Supports resuming a previous session.

Usage:
  python human_labeler.py --batch-label baseline-v2
  python human_labeler.py --batch-label baseline-v2 --items 25
  python human_labeler.py --batch-label baseline-v2 --items 5   # label 5 more

Output:
  output/<batch-label>/human_labels.json
  output/<batch-label>/human_labels.csv
"""

import argparse
import csv
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

DIMENSIONS = [
    {
        "key": "answer_completeness",
        "label": "D1 — Answer Completeness",
        "question": "Does the answer contain enough detail to complete the repair end to end?",
        "pass_hint": "Covers all key stages; homeowner can follow without looking up additional guidance",
        "fail_hint": "Stops short, skips major stages, or is too brief to act on",
    },
    {
        "key": "safety_specificity",
        "label": "D2 — Safety Specificity",
        "question": "Does safety_info name the SPECIFIC hazard AND the SPECIFIC precaution?",
        "pass_hint": "Names exact hazard (e.g. '120V live circuit') + exact action (e.g. 'flip breaker, verify with tester')",
        "fail_hint": "Generic phrases like 'be careful', 'use caution', 'stay safe'",
    },
    {
        "key": "tool_realism",
        "label": "D3 — Tool Realism",
        "question": "Are all tools items a homeowner would own or buy at a hardware store for under $50?",
        "pass_hint": "Standard homeowner tools: screwdrivers, pliers, voltage tester, wrench, etc.",
        "fail_hint": "Professional/trade-only tools, specialty equipment, or items costing >$50",
    },
    {
        "key": "scope_appropriateness",
        "label": "D4 — Scope Appropriateness",
        "question": "Is the repair within realistic DIY capability for an average homeowner?",
        "pass_hint": "A typical homeowner can do this safely; if pro help is needed the answer says so",
        "fail_hint": "Requires professional skills/tools but gives DIY instructions without warning",
    },
    {
        "key": "context_clarity",
        "label": "D5 — Context Clarity",
        "question": "Do question and answer have enough context, and does the answer address the equipment_problem directly?",
        "pass_hint": "Problem is clearly described; answer directly addresses it without drifting to a different repair",
        "fail_hint": "Vague question, or answer addresses a related but distinct problem",
    },
    {
        "key": "tip_usefulness",
        "label": "D6 — Tip Usefulness",
        "question": "Do the tips provide non-obvious, task-specific advice beyond what the steps already cover?",
        "pass_hint": "Tips add concrete value a beginner wouldn't know; include specific measurements, part names, or techniques",
        "fail_hint": "Tips restate a step, give generic encouragement, or are obvious ('unplug first', 'be careful')",
    },
]

_DIM_KEYS = [d["key"] for d in DIMENSIONS]
_CSV_FIELDS = ["trace_id", "category", "labeler", "timestamp"] + _DIM_KEYS + ["overall_pass"]


def _wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent, subsequent_indent=indent)


def _ask_binary(label: str) -> int:
    while True:
        raw = input(f"    Pass? [y/n]: ").strip().lower()
        if raw in ("y", "yes", "1"):
            return 1
        if raw in ("n", "no", "0"):
            return 0
        print("    → Enter y or n.")


def _display_item(item: dict, index: int, total: int) -> None:
    qa = item["qa_pair"]
    print("\n" + "═" * 72)
    print(f"  Item {index}/{total}  |  category: {item['category']}  |  id: {item['trace_id'][:8]}")
    print("═" * 72)
    print(f"\n  QUESTION:\n{_wrap(qa['question'])}")
    print(f"\n  ANSWER:\n{_wrap(qa['answer'])}")
    print(f"\n  EQUIPMENT PROBLEM:\n{_wrap(qa['equipment_problem'])}")
    print(f"\n  TOOLS REQUIRED:\n{_wrap(', '.join(qa['tools_required']))}")
    print(f"\n  STEPS:")
    for i, step in enumerate(qa["steps"], 1):
        print(_wrap(f"{i}. {step}"))
    print(f"\n  SAFETY INFO:\n{_wrap(qa['safety_info'])}")
    print(f"\n  TIPS:")
    for tip in qa["tips"]:
        print(_wrap(f"• {tip}"))


def label_item(item: dict, index: int, total: int) -> dict:
    _display_item(item, index, total)

    print("\n" + "─" * 72)
    print("  LABELS  (y = pass, n = fail, Ctrl-C to quit and save)")
    print("─" * 72)

    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        print(f"\n  {dim['label']}")
        print(f"    {dim['question']}")
        print(f"    ✓ {dim['pass_hint']}")
        print(f"    ✗ {dim['fail_hint']}")
        scores[dim["key"]] = _ask_binary(dim["label"])

    overall_pass = 1 if all(v == 1 for v in scores.values()) else 0
    return {
        "trace_id": item["trace_id"],
        "category": item["category"],
        "labeler": "human",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **scores,
        "overall_pass": overall_pass,
    }


def _append_csv(path: Path, record: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def run_human_labeling(batch_label: str, num_items: int, output_dir: Path) -> None:
    run_dir = output_dir / batch_label
    valid_file = run_dir / "structurally_valid_qa_pairs.json"
    labels_json = run_dir / "human_labels.json"
    labels_csv = run_dir / "human_labels.csv"

    if not valid_file.exists():
        raise FileNotFoundError(f"Not found: {valid_file}. Run Phases 1-2 first.")

    all_items: list[dict] = json.loads(valid_file.read_text())

    existing: list[dict] = json.loads(labels_json.read_text()) if labels_json.exists() else []
    labeled_ids = {r["trace_id"] for r in existing}
    unlabeled = [it for it in all_items if it["trace_id"] not in labeled_ids]
    to_label = unlabeled[:num_items]

    if not to_label:
        print(f"Nothing to label — {len(existing)} items already labeled, none remain.")
        print(f"Run with a larger --items value or after generating more data.")
        return

    print(f"\nBatch        : {batch_label}")
    print(f"Total items  : {len(all_items)}")
    print(f"Already done : {len(existing)}")
    print(f"This session : {len(to_label)}")
    print("\nPress Ctrl-C at any time to quit and save progress.")

    results = list(existing)
    session_records: list[dict] = []

    for i, item in enumerate(to_label, 1):
        try:
            record = label_item(item, len(existing) + i, len(existing) + len(to_label))
        except (KeyboardInterrupt, EOFError):
            print("\n\nInterrupted — saving progress.")
            break

        results.append(record)
        session_records.append(record)

        labels_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        _append_csv(labels_csv, record)

    if not session_records:
        print("No items labeled this session.")
        return

    print(f"\n{'─'*72}")
    print(f"Session complete  |  labeled this session: {len(session_records)}  |  total: {len(results)}")
    print("\nPer-dimension pass rates (this session):")
    for dim in DIMENSIONS:
        rate = sum(r[dim["key"]] for r in session_records) / len(session_records)
        print(f"  {dim['label']}: {rate*100:.0f}%")
    overall = sum(r["overall_pass"] for r in session_records) / len(session_records)
    print(f"  Overall pass (all 6): {overall*100:.0f}%")
    print(f"\nSaved → {labels_json}")
    print(f"Saved → {labels_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human CLI labeler — collect 6-dimension pass/fail labels on generated Q&A items"
    )
    parser.add_argument("--batch-label", required=True, dest="batch_label",
                        help="Batch label of the run to label (e.g. baseline-v2)")
    parser.add_argument("--items", type=int, default=20,
                        help="Number of items to label in this session (default: 20)")
    parser.add_argument("--output-dir", type=str, default="output", dest="output_dir",
                        help="Base output directory (default: output)")
    args = parser.parse_args()

    run_human_labeling(
        batch_label=args.batch_label,
        num_items=args.items,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
