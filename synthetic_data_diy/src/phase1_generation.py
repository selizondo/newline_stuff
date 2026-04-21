"""
Phase 1: Q&A Generation
Generates DIY Repair Q&A pairs using LLM with diverse prompt templates.
"""

import json
import random
import uuid
from pathlib import Path

from instructor.exceptions import InstructorRetryException

from llm_client import instructor_complete
from schema import QAPair, GenerationResult
from prompts import load_prompt_templates


class DIYDatasetGenerator:
    def __init__(self, generation_model: str, strategy: str, batch_id: str, batch_label: str, additional_context: str = ""):
        self.model = generation_model
        self.strategy = strategy
        self.batch_id = batch_id
        self.batch_label = batch_label
        self.additional_context = additional_context
        self.templates = load_prompt_templates(strategy)

    def generate_single(self, template: dict) -> GenerationResult:
        trace_id = str(uuid.uuid4())
        user_content = template["user"]
        if self.additional_context:
            user_content = f"{self.additional_context}\n\n{user_content}"
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": user_content},
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
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy=self.strategy,
                raw_response=qa.model_dump_json(),
                raw_dict=qa.model_dump(),
            )
        except InstructorRetryException as e:
            return GenerationResult(
                trace_id=trace_id,
                category=template["category"],
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy=self.strategy,
                raw_response=str(e.last_completion),
                parse_error=str(e),
                validation_errors=e.validation_errors,
                validation_attempts=e.validation_attempts,
            )
        except Exception as e:
            return GenerationResult(
                trace_id=trace_id,
                category=template["category"],
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy=self.strategy,
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
        return results


def load_generation_results(output_dir: Path) -> list[GenerationResult]:
    """Load Phase 1 output from disk for use by downstream phases.

    Accepts both generation_results.jsonl (standard) and generation_results.json (debug).
    """
    jsonl_file = output_dir / "generation_results.jsonl"
    json_file = output_dir / "generation_results.json"

    if jsonl_file.exists():
        return [
            GenerationResult(**json.loads(line))
            for line in jsonl_file.read_text().splitlines()
            if line.strip()
        ]
    elif json_file.exists():
        return [GenerationResult(**r) for r in json.loads(json_file.read_text())]
    else:
        raise FileNotFoundError(
            f"Not found: {jsonl_file} or {json_file}. Run Phase 1 first."
        )


def run_generation_phase(
    num_samples: int,
    generation_model: str,
    output_dir: Path,
    strategy: str = "zero_shot",
    batch_label: str = "",
    debug: bool = False,
    additional_context: str = "",
) -> list[GenerationResult]:
    batch_id = str(uuid.uuid4())

    generator = DIYDatasetGenerator(
        generation_model=generation_model,
        strategy=strategy,
        batch_id=batch_id,
        batch_label=batch_label,
        additional_context=additional_context,
    )
    results = generator.generate_batch(num_samples)

    parsed = sum(1 for r in results if r.parse_error is None)
    print(f"\nGeneration complete: {parsed}/{len(results)} parsed ({parsed/len(results)*100:.1f}%)")

    if debug:
        out_file = output_dir / "generation_results.json"
        out_file.write_text(
            json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)
        )
    else:
        out_file = output_dir / "generation_results.jsonl"
        out_file.write_text(
            "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in results) + "\n"
        )
    print(f"Saved → {out_file}")
    return results
