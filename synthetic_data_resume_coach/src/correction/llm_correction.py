"""Iterative LLM correction loop for invalid data."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel
import logfire
from llm_utils import instructor_complete

from ..schema import JobDescription, Resume
from ..validators.schema_validator import SchemaValidator, ValidationResult


@dataclass
class CorrectionResult:
    """Result of a correction attempt."""

    original_data: dict
    corrected_data: Optional[dict] = None
    is_corrected: bool = False
    attempts: int = 0
    validation_result: Optional[ValidationResult] = None
    correction_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_corrected": self.is_corrected,
            "attempts": self.attempts,
            "original_data": self.original_data,
            "corrected_data": self.corrected_data,
            "history": self.correction_history,
        }


class LLMCorrector:
    """Iterative LLM-based correction for invalid data."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
    ):
        """Initialize the LLM corrector.

        Args:
            api_key: Groq API key. If not provided, reads from GROQ_API_KEY env var.
            model: Model to use for corrections.
            max_retries: Maximum number of correction attempts.
        """
        self.model = model
        self.max_retries = max_retries
        self.validator = SchemaValidator()

        # Statistics
        self.stats = {
            "total_corrections": 0,
            "successful_corrections": 0,
            "failed_corrections": 0,
            "total_attempts": 0,
        }

        logfire.configure()
        logfire.info(
            "LLMCorrector initialized",
            model=model,
            max_retries=max_retries,
        )

    def correct_resume(self, invalid_result: ValidationResult) -> CorrectionResult:
        """Correct an invalid resume using LLM.

        Args:
            invalid_result: ValidationResult containing the invalid resume data.

        Returns:
            CorrectionResult with correction status and data.
        """
        return self._correct(invalid_result, Resume, "resume")

    def correct_job(self, invalid_result: ValidationResult) -> CorrectionResult:
        """Correct an invalid job description using LLM.

        Args:
            invalid_result: ValidationResult containing the invalid job data.

        Returns:
            CorrectionResult with correction status and data.
        """
        return self._correct(invalid_result, JobDescription, "job")

    def _correct(
        self,
        invalid_result: ValidationResult,
        schema: Type[BaseModel],
        data_type: str,
    ) -> CorrectionResult:
        """Perform iterative LLM correction.

        Args:
            invalid_result: ValidationResult with invalid data.
            schema: Pydantic model class for validation.
            data_type: Type of data ("resume" or "job").

        Returns:
            CorrectionResult with correction status.
        """
        self.stats["total_corrections"] += 1

        result = CorrectionResult(
            original_data=invalid_result.raw_data,
            attempts=0,
        )

        current_data = invalid_result.raw_data.copy()
        current_errors = invalid_result.errors

        with logfire.span(
            "correction_loop",
            data_type=data_type,
            max_retries=self.max_retries,
        ):
            while result.attempts < self.max_retries:
                result.attempts += 1
                self.stats["total_attempts"] += 1

                logfire.info(
                    f"Correction attempt {result.attempts}/{self.max_retries}",
                    error_count=len(current_errors),
                )

                # Format error messages for the LLM
                error_messages = "\n".join(
                    f"- Field '{e.field}': {e.message} (type: {e.error_type})"
                    for e in current_errors
                )

                # Create correction prompt
                prompt = f"""The following {data_type} data has validation errors. Please fix the errors and return valid data.

Current data (JSON):
{json.dumps(current_data, indent=2, default=str)}

Validation errors:
{error_messages}

Please correct the data to fix these errors:
1. Ensure all required fields are present
2. Fix any type mismatches (e.g., strings instead of numbers)
3. Fix date formats to ISO format (YYYY-MM-DD)
4. Ensure dates are logical (end dates after start dates)
5. Fix any constraint violations (min/max values, lengths, etc.)

Return the corrected data as a valid {schema.__name__} object.
"""

                try:
                    corrected = instructor_complete(
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a data correction assistant. Fix the {data_type} data to match the required schema. Use ISO date format (YYYY-MM-DD).",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        response_model=schema,
                        model=self.model,
                    )

                    # Convert to dict for validation
                    corrected_data = corrected.model_dump(mode="json")

                    # Validate the corrected data
                    if data_type == "resume":
                        validation = self.validator.validate_resume(corrected_data)
                    else:
                        validation = self.validator.validate_job(corrected_data)

                    # Record this attempt
                    result.correction_history.append({
                        "attempt": result.attempts,
                        "is_valid": validation.is_valid,
                        "error_count": len(validation.errors) if not validation.is_valid else 0,
                    })

                    if validation.is_valid:
                        result.is_corrected = True
                        result.corrected_data = corrected_data
                        result.validation_result = validation
                        self.stats["successful_corrections"] += 1

                        logfire.info(
                            "Correction successful",
                            attempts=result.attempts,
                        )
                        return result

                    # Update for next iteration
                    current_data = corrected_data
                    current_errors = validation.errors

                except Exception as e:
                    logfire.error(
                        f"Correction attempt {result.attempts} failed",
                        error=str(e),
                    )
                    result.correction_history.append({
                        "attempt": result.attempts,
                        "error": str(e),
                    })

            # Max retries reached without success
            self.stats["failed_corrections"] += 1
            logfire.warning(
                "Correction failed after max retries",
                attempts=result.attempts,
            )

        return result

    def correct_batch(
        self,
        invalid_results: list[ValidationResult],
        data_type: str = "resume",
    ) -> list[CorrectionResult]:
        """Correct a batch of invalid data items.

        Args:
            invalid_results: List of invalid ValidationResult objects.
            data_type: Type of data ("resume" or "job").

        Returns:
            List of CorrectionResult objects.
        """
        results = []

        with logfire.span("correct_batch", count=len(invalid_results), data_type=data_type):
            for i, invalid_result in enumerate(invalid_results):
                logfire.info(f"Correcting item {i + 1}/{len(invalid_results)}")

                if data_type == "resume":
                    correction = self.correct_resume(invalid_result)
                else:
                    correction = self.correct_job(invalid_result)

                results.append(correction)

        return results

    def get_stats(self) -> dict:
        """Get correction statistics.

        Returns:
            Dictionary of correction statistics.
        """
        stats = self.stats.copy()
        if stats["total_corrections"] > 0:
            stats["success_rate"] = (
                stats["successful_corrections"] / stats["total_corrections"]
            )
            stats["avg_attempts"] = stats["total_attempts"] / stats["total_corrections"]
        else:
            stats["success_rate"] = 0
            stats["avg_attempts"] = 0
        return stats

    def reset_stats(self) -> None:
        """Reset correction statistics."""
        self.stats = {
            "total_corrections": 0,
            "successful_corrections": 0,
            "failed_corrections": 0,
            "total_attempts": 0,
        }

    def save_results(
        self,
        results: list[CorrectionResult],
        output_dir: str = "data/validated",
        filename: str = "correction_results.json",
    ) -> Path:
        """Save correction results to a JSON file.

        Args:
            results: List of CorrectionResult objects.
            output_dir: Output directory path.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        data = {
            "statistics": self.get_stats(),
            "results": [r.to_dict() for r in results],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logfire.info(f"Saved correction results to {file_path}")
        return file_path
