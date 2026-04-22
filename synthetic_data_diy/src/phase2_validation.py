"""
Phase 2: Structural Validation
Owns all Pydantic schema validation. Consumes Phase 1 GenerationResult objects
(which carry raw parsed dicts) and produces ValidatedResult objects.

After Pydantic validation, per-item heuristic gates drop items that will fail
LLM-as-Judge scoring anyway, saving judge budget in Phases 4–5.
Batch-level checks (dedup, category distribution) flag issues without dropping items.
"""

import json
import re
from collections import Counter
from pathlib import Path

from schema import QAPair, GenerationResult, ValidatedResult, ValidationSummary

# Per-item gate constants
_MIN_SAFETY_INFO_LEN = 80
_MIN_TIP_LEN = 30
_SAFETY_GENERIC_PHRASES = frozenset(["be careful", "use caution", "stay safe", "good luck"])
_TOOL_BLOCKLIST = frozenset(["professional-grade", "trade-only", "specialty"])

# Batch-level threshold: each category must be ≥ 18% of the valid set
_MIN_CATEGORY_FRACTION = 0.18


# ---------------------------------------------------------------------------
# Per-item heuristic gates
# ---------------------------------------------------------------------------

def _apply_heuristic_gates(qa: QAPair) -> list[str]:
    """Return gate failure reasons; empty list means all gates passed."""
    failures: list[str] = []

    # D2: safety_info minimum length
    safety_len = len(qa.safety_info.strip())
    if safety_len < _MIN_SAFETY_INFO_LEN:
        failures.append(f"safety_info too short ({safety_len} < {_MIN_SAFETY_INFO_LEN} chars)")

    # D2/D6: generic phrase blocklist in safety_info or tips
    combined_lower = (qa.safety_info + " " + " ".join(qa.tips)).lower()
    for phrase in _SAFETY_GENERIC_PHRASES:
        if phrase in combined_lower:
            failures.append(f"generic phrase '{phrase}' in safety_info/tips")
            break

    # D3: trade/professional tool blocklist
    tools_lower = " ".join(qa.tools_required).lower()
    for term in _TOOL_BLOCKLIST:
        if term in tools_lower:
            failures.append(f"blocked tool term '{term}' in tools_required")
            break

    # D6: tip minimum length
    short_tips = [t for t in qa.tips if len(t.strip()) < _MIN_TIP_LEN]
    if short_tips:
        failures.append(f"{len(short_tips)} tip(s) under {_MIN_TIP_LEN} chars")

    return failures


# ---------------------------------------------------------------------------
# Batch-level checks
# ---------------------------------------------------------------------------

def _normalize_question(q: str) -> str:
    return re.sub(r"[^\w\s]", "", q.lower()).strip()


def _run_dedup(items: list[ValidatedResult]) -> tuple[list[ValidatedResult], int]:
    """Drop items whose normalized question was already seen. Returns (deduped, n_removed)."""
    seen: set[str] = set()
    deduped: list[ValidatedResult] = []
    for item in items:
        key = _normalize_question(item.qa_pair.question)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped, len(items) - len(deduped)


def _check_category_distribution(
    items: list[ValidatedResult],
) -> tuple[dict[str, float], bool]:
    """Return per-category fractions and whether all meet _MIN_CATEGORY_FRACTION."""
    if not items:
        return {}, False
    counts = Counter(r.category for r in items)
    fractions = {cat: counts[cat] / len(items) for cat in sorted(counts)}
    ok = all(f >= _MIN_CATEGORY_FRACTION for f in fractions.values())
    return fractions, ok


# ---------------------------------------------------------------------------
# Validator class (Pydantic gate)
# ---------------------------------------------------------------------------

class QAPairValidator:
    def _validate_one(
        self, result: GenerationResult
    ) -> tuple[bool, list[str], QAPair | None]:
        if result.parse_error is not None:
            return False, [result.parse_error], None
        if result.raw_dict is None:
            return False, ["No JSON data returned by LLM"], None

        try:
            qa = QAPair(**result.raw_dict)
        except Exception as e:
            return False, [f"Schema error: {e}"], None

        errors: list[str] = []
        if len(qa.steps) < 3:
            errors.append(f"Too few steps: {len(qa.steps)} (need ≥3)")
        if not qa.tools_required:
            errors.append("tools_required is empty")
        if len(qa.safety_info.strip()) < 10:
            errors.append("safety_info too short")

        return len(errors) == 0, errors, qa if not errors else None

    def validate_batch(
        self, results: list[GenerationResult]
    ) -> tuple[list[ValidatedResult], ValidationSummary]:
        valid: list[ValidatedResult] = []
        all_errors: list[str] = []

        for result in results:
            ok, errors, qa = self._validate_one(result)
            if ok and qa is not None:
                valid.append(ValidatedResult(
                    trace_id=result.trace_id,
                    category=result.category,
                    qa_pair=qa,
                ))
            else:
                all_errors.extend(errors)

        common = [err for err, _ in Counter(all_errors).most_common(5)]
        summary = ValidationSummary(
            total_generated=len(results),
            total_valid=len(valid),
            total_invalid=len(results) - len(valid),
            validation_rate=len(valid) / len(results) if results else 0.0,
            common_errors=common,
        )
        return valid, summary


# ---------------------------------------------------------------------------
# Phase entry point
# ---------------------------------------------------------------------------

def run_validation_phase(
    results: list[GenerationResult],
    output_dir: Path,
) -> tuple[list[ValidatedResult], ValidationSummary]:
    validator = QAPairValidator()
    schema_valid, summary = validator.validate_batch(results)

    print(f"Structural validation: {summary.total_valid}/{summary.total_generated} passed ({summary.validation_rate*100:.1f}%)")
    if summary.common_errors:
        print("  Common errors:")
        for err in summary.common_errors:
            print(f"    • {err}")

    # Per-item heuristic gates
    gate_passed: list[ValidatedResult] = []
    gate_failures: list[dict] = []
    for item in schema_valid:
        failures = _apply_heuristic_gates(item.qa_pair)
        if failures:
            gate_failures.append({"trace_id": item.trace_id, "category": item.category, "failures": failures})
        else:
            gate_passed.append(item)
    print(f"Heuristic gates:      {len(gate_passed)}/{len(schema_valid)} passed ({len(gate_failures)} dropped)")
    for detail in gate_failures:
        print(f"  • {detail['trace_id'][:8]} [{detail['category']}]: {'; '.join(detail['failures'])}")

    # Deduplication
    final_results, n_dupes = _run_dedup(gate_passed)
    if n_dupes:
        print(f"Deduplication:        removed {n_dupes} duplicate question(s)")

    # Category distribution
    cat_fractions, dist_ok = _check_category_distribution(final_results)
    if not dist_ok:
        low = [f"{c} ({f*100:.0f}%)" for c, f in cat_fractions.items() if f < _MIN_CATEGORY_FRACTION]
        print(f"  WARNING: category distribution unbalanced — below {_MIN_CATEGORY_FRACTION*100:.0f}%: {', '.join(low)}")
    else:
        print("Category distribution OK: " + ", ".join(f"{c}={f*100:.0f}%" for c, f in cat_fractions.items()))

    # Write outputs
    valid_file = output_dir / "structurally_valid_qa_pairs.json"
    valid_file.write_text(json.dumps(
        [{"trace_id": r.trace_id, "category": r.category, "qa_pair": r.qa_pair.model_dump()} for r in final_results],
        indent=2, ensure_ascii=False,
    ))

    (output_dir / "validation_summary.json").write_text(json.dumps(summary.model_dump(), indent=2))

    gate_report = {
        "schema_passed": len(schema_valid),
        "gate_passed": len(gate_passed),
        "gate_failed": len(gate_failures),
        "gate_failure_details": gate_failures,
        "dedup_removed": n_dupes,
        "final_valid": len(final_results),
        "category_distribution": cat_fractions,
        "category_distribution_ok": dist_ok,
    }
    (output_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2))

    print(f"Saved → {valid_file}")
    return final_results, summary


def load_valid_data(output_dir: Path) -> list[ValidatedResult]:
    """Load Phase 2 output from disk for downstream phases."""
    valid_file = output_dir / "structurally_valid_qa_pairs.json"
    if not valid_file.exists():
        raise FileNotFoundError(f"Not found: {valid_file}. Run Phase 2 first.")

    data = json.loads(valid_file.read_text())
    return [
        ValidatedResult(
            trace_id=item["trace_id"],
            category=item["category"],
            qa_pair=QAPair(**item["qa_pair"]),
        )
        for item in data
    ]
