"""
Phase 1: Q&A Generation
Generates DIY Repair Q&A pairs using LLM with diverse prompt templates.
"""

import json
import random
import time
import uuid
from pathlib import Path

from instructor.exceptions import InstructorRetryException

from config import get_settings
from llm_client import instructor_complete
from models import QAPair, GenerationResult
from prompts import BASELINE_DIR, load_prompt_templates


class DIYDatasetGenerator:
    def __init__(self, model: str, prompts_dir: Path):
        self.model = model
        self.templates = load_prompt_templates(prompts_dir)
        self.settings = get_settings()

    def generate_single(self, template: dict) -> GenerationResult:
        trace_id = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"]},
        ]
        try:
            qa: QAPair = instructor_complete(
                messages,
                response_model=QAPair,
                model=self.model,
                temperature=0.7,
                max_tokens=1500,
            )
            return GenerationResult(
                trace_id=trace_id,
                category=template["category"],
                raw_response=qa.model_dump_json(),
                raw_dict=qa.model_dump(),
            )
        except InstructorRetryException as e:
            return GenerationResult(
                trace_id=trace_id,
                category=template["category"],
                raw_response=str(e.last_completion),
                parse_error=str(e),
                validation_errors=e.validation_errors,
                validation_attempts=e.validation_attempts,
            )
        except Exception as e:
            return GenerationResult(
                trace_id=trace_id,
                category=template["category"],
                raw_response="",
                parse_error=str(e),
            )

    def generate_batch(self, num_samples: int) -> list[GenerationResult]:
        # Stratified sampling: cycle through all categories in shuffled order to
        # guarantee every category appears even at small sample sizes.
        indices = list(range(len(self.templates)))
        schedule: list[int] = []
        while len(schedule) < num_samples:
            random.shuffle(indices)
            schedule.extend(indices)
        schedule = schedule[:num_samples]

        results: list[GenerationResult] = []
        for i, idx in enumerate(schedule):
            template = self.templates[idx]
            print(f"  [{i+1}/{num_samples}] Generating {template['category']}...", end=" ", flush=True)
            result = self.generate_single(template)
            status = "OK" if result.parse_error is None else f"FAIL ({result.parse_error[:60]})"
            print(status)
            results.append(result)
            if i < num_samples - 1:
                time.sleep(self.settings.rate_limit_delay)
        return results


def load_generation_results(output_dir: Path) -> list[GenerationResult]:
    """Load Phase 1 output from disk for use by downstream phases."""
    results_file = output_dir / "generation_results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"Not found: {results_file}. Run Phase 1 first.")

    return [
        GenerationResult(**json.loads(line))
        for line in results_file.read_text().splitlines()
        if line.strip()
    ]


def run_generation_phase(
    num_samples: int,
    model: str,
    output_dir: Path,
    prompts_dir: Path = BASELINE_DIR,
) -> list[GenerationResult]:
    generator = DIYDatasetGenerator(model=model, prompts_dir=prompts_dir)
    results = generator.generate_batch(num_samples)

    parsed = sum(1 for r in results if r.parse_error is None)
    print(f"\nGeneration complete: {parsed}/{len(results)} parsed ({parsed/len(results)*100:.1f}%)")

    out_file = output_dir / "generation_results.jsonl"
    out_file.write_text(
        "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in results) + "\n"
    )
    print(f"Saved → {out_file}")
    return results
