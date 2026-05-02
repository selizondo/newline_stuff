"""Failure mode labeling and analysis for validation errors."""

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import logfire
import pandas as pd

from ..validators.schema_validator import ValidationResult, ValidationError_


class FailureCategory(str, Enum):
    """Categories of validation failures."""

    MISSING_REQUIRED = "missing_required_field"
    INVALID_TYPE = "invalid_type"
    INVALID_FORMAT = "invalid_format"
    DATE_ERROR = "date_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNKNOWN = "unknown"


@dataclass
class FailureMode:
    """Detailed failure mode information."""

    category: FailureCategory
    field: str
    error_type: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
            "count": self.count,
            "examples": self.examples[:5],  # Limit examples
        }


class FailureModeAnalyzer:
    """Analyze and categorize validation failures."""

    # Mapping of Pydantic error types to failure categories
    ERROR_TYPE_MAPPING = {
        "missing": FailureCategory.MISSING_REQUIRED,
        "value_error.missing": FailureCategory.MISSING_REQUIRED,
        "type_error": FailureCategory.INVALID_TYPE,
        "type_error.integer": FailureCategory.INVALID_TYPE,
        "type_error.string": FailureCategory.INVALID_TYPE,
        "type_error.float": FailureCategory.INVALID_TYPE,
        "type_error.list": FailureCategory.INVALID_TYPE,
        "type_error.dict": FailureCategory.INVALID_TYPE,
        "value_error.email": FailureCategory.INVALID_FORMAT,
        "value_error.url": FailureCategory.INVALID_FORMAT,
        "value_error.date": FailureCategory.DATE_ERROR,
        "value_error.datetime": FailureCategory.DATE_ERROR,
        "value_error": FailureCategory.CONSTRAINT_VIOLATION,
        "assertion_error": FailureCategory.LOGICAL_INCONSISTENCY,
    }

    def __init__(self):
        """Initialize the failure mode analyzer."""
        logfire.configure()
        self.failure_modes: dict[str, FailureMode] = {}
        self.raw_errors: list[ValidationError_] = []

    def categorize_error(self, error: ValidationError_) -> FailureCategory:
        """Categorize a validation error into a failure category.

        Args:
            error: ValidationError_ to categorize.

        Returns:
            FailureCategory enum value.
        """
        error_type = error.error_type.lower()

        # Check for date-related errors
        if "date" in error.field.lower() or "date" in error.message.lower():
            if "before" in error.message.lower() or "after" in error.message.lower():
                return FailureCategory.LOGICAL_INCONSISTENCY
            return FailureCategory.DATE_ERROR

        # Check for missing field errors
        if "missing" in error_type or "required" in error.message.lower():
            return FailureCategory.MISSING_REQUIRED

        # Check for type errors
        if "type" in error_type:
            return FailureCategory.INVALID_TYPE

        # Check for format errors
        if any(fmt in error_type for fmt in ["email", "url", "phone", "format"]):
            return FailureCategory.INVALID_FORMAT

        # Check for constraint violations
        if any(
            constraint in error.message.lower()
            for constraint in ["greater than", "less than", "minimum", "maximum", "length"]
        ):
            return FailureCategory.CONSTRAINT_VIOLATION

        # Check for logical inconsistencies
        if "must be" in error.message.lower() or "invalid" in error.message.lower():
            return FailureCategory.LOGICAL_INCONSISTENCY

        # Use mapping if available
        for pattern, category in self.ERROR_TYPE_MAPPING.items():
            if pattern in error_type:
                return category

        return FailureCategory.UNKNOWN

    def analyze_results(
        self,
        results: list[ValidationResult],
    ) -> dict[str, FailureMode]:
        """Analyze validation results and identify failure modes.

        Args:
            results: List of ValidationResult objects to analyze.

        Returns:
            Dictionary mapping failure mode keys to FailureMode objects.
        """
        with logfire.span("analyze_failure_modes", result_count=len(results)):
            for result in results:
                if not result.is_valid:
                    for error in result.errors:
                        self.raw_errors.append(error)
                        self._process_error(error)

            logfire.info(
                "Failure mode analysis complete",
                total_errors=len(self.raw_errors),
                unique_modes=len(self.failure_modes),
            )

        return self.failure_modes

    def _process_error(self, error: ValidationError_) -> None:
        """Process a single error and update failure modes.

        Args:
            error: ValidationError_ to process.
        """
        category = self.categorize_error(error)
        key = f"{category.value}:{error.field}:{error.error_type}"

        if key in self.failure_modes:
            self.failure_modes[key].count += 1
            if len(self.failure_modes[key].examples) < 5:
                example = str(error.input_value)[:100] if error.input_value else "N/A"
                if example not in self.failure_modes[key].examples:
                    self.failure_modes[key].examples.append(example)
        else:
            self.failure_modes[key] = FailureMode(
                category=category,
                field=error.field,
                error_type=error.error_type,
                message=error.message,
                count=1,
                examples=[str(error.input_value)[:100] if error.input_value else "N/A"],
            )

    def get_statistics(self) -> dict:
        """Get failure mode statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self.failure_modes:
            return {
                "total_errors": 0,
                "unique_modes": 0,
                "by_category": {},
                "by_field": {},
                "top_failures": [],
            }

        # Count by category
        category_counts = Counter()
        field_counts = Counter()

        for mode in self.failure_modes.values():
            category_counts[mode.category.value] += mode.count
            field_counts[mode.field] += mode.count

        # Get top failures
        sorted_modes = sorted(
            self.failure_modes.values(),
            key=lambda m: m.count,
            reverse=True,
        )

        return {
            "total_errors": len(self.raw_errors),
            "unique_modes": len(self.failure_modes),
            "by_category": dict(category_counts),
            "by_field": dict(field_counts.most_common(10)),
            "top_failures": [m.to_dict() for m in sorted_modes[:10]],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert failure modes to a pandas DataFrame.

        Returns:
            DataFrame with failure mode data.
        """
        data = [mode.to_dict() for mode in self.failure_modes.values()]
        return pd.DataFrame(data)

    def save_analysis(
        self,
        output_dir: str = "data/validated",
        filename: str = "failure_modes.json",
    ) -> Path:
        """Save failure mode analysis to a JSON file.

        Args:
            output_dir: Output directory path.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        data = {
            "statistics": self.get_statistics(),
            "failure_modes": [m.to_dict() for m in self.failure_modes.values()],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logfire.info(f"Saved failure mode analysis to {file_path}")
        return file_path

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.failure_modes = {}
        self.raw_errors = []
