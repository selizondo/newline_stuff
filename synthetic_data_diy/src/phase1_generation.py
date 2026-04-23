"""
Phase 1: Q&A Generation
Generates DIY Repair Q&A pairs using LLM with diverse prompt templates.

Pass generation_model="mock" to run_generation_phase() to sample from the local
benchmark cache instead of calling an LLM (zero API credentials required).
"""

import json
import random
import uuid
from pathlib import Path

from instructor.exceptions import InstructorRetryException

from llm_client import instructor_complete
from schema import QAPair, GenerationResult
from prompts import load_prompt_templates

_QA_FIELDS = frozenset(QAPair.model_fields)


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


class BenchmarkGenerator:
    """Samples schema-valid benchmark rows instead of calling an LLM.

    Implements the same generate_batch() interface as DIYDatasetGenerator so
    run_generation_phase() can dispatch transparently on model=="mock".
    """

    strategy = "mock"

    def __init__(self, batch_id: str, batch_label: str, seed: int, output_base: Path):
        self.batch_id = batch_id
        self.batch_label = batch_label
        self.seed = seed
        self.output_base = output_base

    def generate_batch(self, num_samples: int) -> list[GenerationResult]:
        from benchmark_cache import sample_validated_rows

        # Oversample 3× so enough rows survive Phase 2 heuristic gates.
        rows = sample_validated_rows(num_samples, self.seed, self.output_base, oversample_factor=3)

        results: list[GenerationResult] = []
        for i, row in enumerate(rows):
            raw_dict = {k: row[k] for k in _QA_FIELDS if k in row}
            results.append(GenerationResult(
                trace_id=str(uuid.uuid4()),
                category=row["category"],
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy="mock",
                raw_response=json.dumps(raw_dict),
                raw_dict=raw_dict,
            ))
        print(f"  Loaded {len(results)} rows from benchmark cache (requesting {num_samples}, oversampled 3×)")
        return results


def load_generation_results(output_dir: Path) -> list[GenerationResult]:
    """Load Phase 1 output from disk for use by downstream phases."""
    json_file = output_dir / "generation_results.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Not found: {json_file}. Run Phase 1 first.")
    return [GenerationResult(**r) for r in json.loads(json_file.read_text())]


def run_generation_phase(
    num_samples: int,
    generation_model: str,
    output_dir: Path,
    strategy: str = "zero_shot",
    batch_label: str = "",
    additional_context: str = "",
    seed: int = 42,
    output_base: Path | None = None,
) -> list[GenerationResult]:
    batch_id = str(uuid.uuid4())

    if generation_model == "mock":
        generator: BenchmarkGenerator = BenchmarkGenerator(
            batch_id=batch_id,
            batch_label=batch_label,
            seed=seed,
            output_base=output_base or output_dir.parent,
        )
    else:
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

    out_file = output_dir / "generation_results.json"
    out_file.write_text(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False))
    print(f"Saved → {out_file}")
    return results
