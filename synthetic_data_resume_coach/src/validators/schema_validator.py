"""Schema validator for resume and job description data."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError
import logfire

from ..schema import JobDescription, Resume, ResumeJobPair


@dataclass
class ValidationError_:
    """Detailed validation error information."""

    field: str
    error_type: str
    message: str
    input_value: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
            "input_value": str(self.input_value)[:100] if self.input_value else None,
        }


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    data: Optional[BaseModel] = None
    raw_data: Optional[dict] = None
    errors: list[ValidationError_] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
        }


class SchemaValidator:
    """Validate data against Pydantic schemas."""

    def __init__(self):
        """Initialize the schema validator."""
        logfire.configure()
        self.validation_stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors_by_type": {},
            "errors_by_field": {},
        }

    def validate_resume(self, data: dict) -> ValidationResult:
        """Validate resume data against the Resume schema.

        Args:
            data: Raw resume data dictionary.

        Returns:
            ValidationResult with validation status and errors.
        """
        return self._validate(data, Resume)

    def validate_job(self, data: dict) -> ValidationResult:
        """Validate job description data against the JobDescription schema.

        Args:
            data: Raw job description data dictionary.

        Returns:
            ValidationResult with validation status and errors.
        """
        return self._validate(data, JobDescription)

    def validate_pair(self, data: dict) -> ValidationResult:
        """Validate resume-job pair data.

        Args:
            data: Raw pair data dictionary.

        Returns:
            ValidationResult with validation status and errors.
        """
        return self._validate(data, ResumeJobPair)

    def _validate(
        self,
        data: dict,
        schema: Type[BaseModel],
    ) -> ValidationResult:
        """Validate data against a Pydantic schema.

        Args:
            data: Raw data dictionary.
            schema: Pydantic model class to validate against.

        Returns:
            ValidationResult with validation status and errors.
        """
        self.validation_stats["total"] += 1

        with logfire.span("validate_data", schema=schema.__name__):
            try:
                validated = schema.model_validate(data)
                self.validation_stats["valid"] += 1

                logfire.info(
                    "Validation successful",
                    schema=schema.__name__,
                )

                return ValidationResult(
                    is_valid=True,
                    data=validated,
                    raw_data=data,
                    errors=[],
                )

            except ValidationError as e:
                self.validation_stats["invalid"] += 1
                errors = self._parse_validation_errors(e)

                # Update error statistics
                for error in errors:
                    self.validation_stats["errors_by_type"][error.error_type] = (
                        self.validation_stats["errors_by_type"].get(error.error_type, 0) + 1
                    )
                    self.validation_stats["errors_by_field"][error.field] = (
                        self.validation_stats["errors_by_field"].get(error.field, 0) + 1
                    )

                logfire.warning(
                    "Validation failed",
                    schema=schema.__name__,
                    error_count=len(errors),
                )

                return ValidationResult(
                    is_valid=False,
                    data=None,
                    raw_data=data,
                    errors=errors,
                )

    def _parse_validation_errors(self, exc: ValidationError) -> list[ValidationError_]:
        """Parse Pydantic ValidationError into detailed error objects.

        Args:
            exc: Pydantic ValidationError exception.

        Returns:
            List of ValidationError_ objects.
        """
        errors = []

        for error in exc.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            errors.append(
                ValidationError_(
                    field=field_path,
                    error_type=error["type"],
                    message=error["msg"],
                    input_value=error.get("input"),
                )
            )

        return errors

    def validate_batch(
        self,
        data_list: list[dict],
        data_type: str = "resume",
    ) -> tuple[list[ValidationResult], dict]:
        """Validate a batch of data items.

        Args:
            data_list: List of data dictionaries to validate.
            data_type: Type of data ("resume", "job", or "pair").

        Returns:
            Tuple of (list of ValidationResults, summary statistics).
        """
        results = []

        validator_map = {
            "resume": self.validate_resume,
            "job": self.validate_job,
            "pair": self.validate_pair,
        }

        validate_func = validator_map.get(data_type)
        if not validate_func:
            raise ValueError(f"Invalid data_type: {data_type}")

        with logfire.span("validate_batch", count=len(data_list), data_type=data_type):
            for i, data in enumerate(data_list):
                result = validate_func(data)
                results.append(result)

                if not result.is_valid:
                    logfire.debug(
                        f"Item {i + 1} validation failed",
                        errors=[e.to_dict() for e in result.errors],
                    )

        valid_count = sum(1 for r in results if r.is_valid)
        summary = {
            "total": len(results),
            "valid": valid_count,
            "invalid": len(results) - valid_count,
            "success_rate": valid_count / len(results) if results else 0,
        }

        logfire.info(
            "Batch validation complete",
            **summary,
        )

        return results, summary

    def get_stats(self) -> dict:
        """Get cumulative validation statistics.

        Returns:
            Dictionary of validation statistics.
        """
        stats = self.validation_stats.copy()
        stats["success_rate"] = (
            stats["valid"] / stats["total"] if stats["total"] > 0 else 0
        )
        return stats

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors_by_type": {},
            "errors_by_field": {},
        }

    def save_results(
        self,
        results: list[ValidationResult],
        output_dir: str = "data/validated",
        filename: str = "validation_results.json",
    ) -> Path:
        """Save validation results to a JSON file.

        Args:
            results: List of ValidationResult objects.
            output_dir: Output directory path.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        data = [r.to_dict() for r in results]

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logfire.info(f"Saved {len(results)} validation results to {file_path}")
        return file_path
