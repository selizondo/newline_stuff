"""
Phase 2: Structural Validation
Owns all Pydantic schema validation. Consumes Phase 1 GenerationResult objects
(which carry raw parsed dicts) and produces ValidatedResult objects.
"""

import json
from collections import Counter
from pathlib import Path

from schema import QAPair, GenerationResult, ValidatedResult, ValidationSummary


class QAPairValidator:
    def _validate_one(
        self, result: GenerationResult
    ) -> tuple[bool, list[str], QAPair | None]:
        # If Phase 1 could not parse JSON, fail immediately
        if result.parse_error is not None:
            return False, [result.parse_error], None
        if result.raw_dict is None:
            return False, ["No JSON data returned by LLM"], None

        errors: list[str] = []
        qa: QAPair | None = None

        try:
            qa = QAPair(**result.raw_dict)
        except Exception as e:
            errors.append(f"Schema error: {e}")
            return False, errors, None

        # Extra checks beyond Pydantic field constraints
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


def run_validation_phase(
    results: list[GenerationResult],
    output_dir: Path,
) -> tuple[list[ValidatedResult], ValidationSummary]:
    validator = QAPairValidator()
    valid_results, summary = validator.validate_batch(results)

    print(f"Structural validation: {summary.total_valid}/{summary.total_generated} passed ({summary.validation_rate*100:.1f}%)")
    if summary.common_errors:
        print("  Common errors:")
        for err in summary.common_errors:
            print(f"    • {err}")

    valid_file = output_dir / "structurally_valid_qa_pairs.json"
    valid_data = [
        {"trace_id": r.trace_id, "category": r.category, "qa_pair": r.qa_pair.model_dump()}
        for r in valid_results
    ]
    valid_file.write_text(json.dumps(valid_data, indent=2, ensure_ascii=False))

    summary_file = output_dir / "validation_summary.json"
    summary_file.write_text(json.dumps(summary.model_dump(), indent=2))

    print(f"Saved → {valid_file}")
    return valid_results, summary


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
