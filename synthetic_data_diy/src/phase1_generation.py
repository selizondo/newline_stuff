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

from pydantic import BaseModel, Field

from llm_client import instructor_complete
from schema import QAPair, GenerationResult, SharedQuestion
from prompts import load_prompt_templates, load_answer_templates

_QA_FIELDS = frozenset(QAPair.model_fields)
SHARED_QUESTIONS_DIR = "_shared"
SHARED_QUESTIONS_FILE = "questions.json"


class _QuestionStem(BaseModel):
    """Instructor response model for Ph1a question-only generation."""
    question: str = Field(..., min_length=10, max_length=500)
    equipment_problem: str = Field(..., min_length=5, max_length=200)


class DIYDatasetGenerator:
    def __init__(self, generation_model: str, strategy: str, batch_id: str, batch_label: str, additional_context: str = ""):
        self.model = generation_model
        self.strategy = strategy
        self.batch_id = batch_id
        self.batch_label = batch_label
        self.additional_context = additional_context
        self.templates = load_prompt_templates(strategy)
        self._answer_templates: dict[str, dict] | None = None  # lazy-loaded for Ph1b

    def _get_answer_template(self, category: str) -> dict:
        if self._answer_templates is None:
            self._answer_templates = {t["category"]: t for t in load_answer_templates(self.strategy)}
        return self._answer_templates[category]

    def generate_single_from_question(self, shared_q: SharedQuestion) -> GenerationResult:
        """Ph1b: generate answer fields for a pre-existing shared question."""
        template = self._get_answer_template(shared_q.category)
        user_content = template["user"].format(
            question=shared_q.question,
            equipment_problem=shared_q.equipment_problem,
        )
        if self.additional_context:
            user_content = f"{self.additional_context}\n\n{user_content}"
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": user_content},
        ]
        obs_context = {
            "trace_id": shared_q.trace_id,
            "batch_label": self.batch_label,
            "phase": 1,
            "category": shared_q.category,
            "prompt_strategy": self.strategy,
        }
        try:
            qa: QAPair = instructor_complete(
                messages,
                response_model=QAPair,
                model=self.model,
                temperature=0.7,
                max_tokens=1500,
                obs_context=obs_context,
            )
            return GenerationResult(
                trace_id=shared_q.trace_id,
                category=shared_q.category,
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy=self.strategy,
                raw_response=qa.model_dump_json(),
                raw_dict=qa.model_dump(),
            )
        except InstructorRetryException as e:
            return GenerationResult(
                trace_id=shared_q.trace_id,
                category=shared_q.category,
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
                trace_id=shared_q.trace_id,
                category=shared_q.category,
                batch_id=self.batch_id,
                batch_label=self.batch_label,
                prompt_strategy=self.strategy,
                raw_response="",
                parse_error=str(e),
            )

    def generate_single(self, template: dict) -> GenerationResult:
        trace_id = str(uuid.uuid4())
        user_content = template["user"]
        if self.additional_context:
            user_content = f"{self.additional_context}\n\n{user_content}"
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": user_content},
        ]
        obs_context = {
            "trace_id": trace_id,
            "batch_label": self.batch_label,
            "phase": 1,
            "category": template["category"],
            "prompt_strategy": self.strategy,
        }
        try:
            qa: QAPair = instructor_complete(
                messages,
                response_model=QAPair,
                model=self.model,
                temperature=0.7,
                max_tokens=1500,
                obs_context=obs_context,
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

    def generate_batch(
        self,
        num_samples: int,
        samples_per_category: int | None = None,
        remaining_per_category: dict[str, int] | None = None,
    ) -> list[GenerationResult]:
        categories = [t["category"] for t in self.templates]

        if remaining_per_category is not None:
            # Resume mode: generate exactly what's still needed per category.
            schedule: list[int] = []
            for idx, t in enumerate(self.templates):
                schedule.extend([idx] * remaining_per_category.get(t["category"], 0))
            random.shuffle(schedule)
        elif samples_per_category is not None:
            # Per-category mode: N samples per category, round-robin.
            indices = list(range(len(self.templates)))
            schedule = []
            for _ in range(samples_per_category):
                shuffled = indices[:]
                random.shuffle(shuffled)
                schedule.extend(shuffled)
        else:
            # Default: stratified total, cycle through categories.
            indices = list(range(len(self.templates)))
            schedule = []
            while len(schedule) < num_samples:
                random.shuffle(indices)
                schedule.extend(indices)
            schedule = schedule[:num_samples]

        total = len(schedule)
        results: list[GenerationResult] = []
        for i, idx in enumerate(schedule):
            template = self.templates[idx]
            print(f"  [{i+1}/{total}] Generating {template['category']}...", end=" ", flush=True)
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


def load_shared_questions(output_base: Path) -> list[SharedQuestion]:
    """Load the shared question set generated by run_question_generation_phase()."""
    path = output_base / SHARED_QUESTIONS_DIR / SHARED_QUESTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Shared questions not found: {path}\n"
            "Run 'python main.py questions' first."
        )
    return [SharedQuestion(**r) for r in json.loads(path.read_text())]


def run_question_generation_phase(
    num_per_category: int,
    generation_model: str,
    output_base: Path,
    seed: int = 42,
    overwrite: bool = False,
) -> list[SharedQuestion]:
    """Ph1a: generate a shared question set (fixed inputs for all baselines).

    Generates num_per_category questions per category using the question_gen
    prompt strategy. Saves to output_base/_shared/questions.json.
    """
    import random
    random.seed(seed)

    out_dir = output_base / SHARED_QUESTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / SHARED_QUESTIONS_FILE

    if out_file.exists() and not overwrite:
        existing = [SharedQuestion(**r) for r in json.loads(out_file.read_text())]
        print(f"Shared questions already exist ({len(existing)} items). Use --overwrite to regenerate.")
        return existing

    templates = load_prompt_templates("question_gen")
    results: list[SharedQuestion] = []

    total = len(templates) * num_per_category
    i = 0
    for template in templates:
        category = template["category"]
        for _ in range(num_per_category):
            i += 1
            trace_id = str(uuid.uuid4())
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": template["user"]},
            ]
            obs_context = {
                "trace_id": trace_id,
                "batch_label": "_shared",
                "phase": "1a",
                "category": category,
                "prompt_strategy": "question_gen",
            }
            print(f"  [{i}/{total}] Generating question for {category}... ", end="", flush=True)
            try:
                stem: _QuestionStem = instructor_complete(
                    messages,
                    response_model=_QuestionStem,
                    model=generation_model,
                    temperature=0.9,
                    max_tokens=300,
                    obs_context=obs_context,
                )
                results.append(SharedQuestion(
                    trace_id=trace_id,
                    category=category,
                    question=stem.question,
                    equipment_problem=stem.equipment_problem,
                ))
                print("OK")
            except Exception as e:
                print(f"FAIL ({e})")

    out_file.write_text(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} shared questions → {out_file}")
    return results


def run_answer_generation_phase(
    shared_questions: list[SharedQuestion],
    generation_model: str,
    output_dir: Path,
    strategy: str,
    batch_label: str,
    additional_context: str = "",
    overwrite: bool = False,
) -> list[GenerationResult]:
    """Ph1b: generate answers for a fixed shared question set.

    Uses the same trace_ids as the shared questions so results can be
    joined across baselines by trace_id for apples-to-apples comparison.
    """
    batch_id = str(uuid.uuid4())
    out_file = output_dir / "generation_results.json"

    existing_ids: set[str] = set()
    existing_dicts: list[dict] = []
    if not overwrite and out_file.exists():
        existing_dicts = json.loads(out_file.read_text())
        existing_ids = {r["trace_id"] for r in existing_dicts}

    generator = DIYDatasetGenerator(
        generation_model=generation_model,
        strategy=strategy,
        batch_id=batch_id,
        batch_label=batch_label,
        additional_context=additional_context,
    )

    pending = [q for q in shared_questions if q.trace_id not in existing_ids]
    if not pending:
        print(f"Resume: all {len(shared_questions)} questions already answered. Skipping.")
        return [GenerationResult(**r) for r in existing_dicts]
    if existing_ids:
        print(f"Resume: {len(existing_ids)} existing, generating {len(pending)} more.")

    results: list[GenerationResult] = []
    for i, shared_q in enumerate(pending):
        print(f"  [{i+1}/{len(pending)}] Answering {shared_q.category} ({shared_q.trace_id[:8]})... ", end="", flush=True)
        result = generator.generate_single_from_question(shared_q)
        status = "OK" if result.parse_error is None else f"FAIL ({result.parse_error[:60]})"
        print(status)
        results.append(result)

    parsed = sum(1 for r in results if r.parse_error is None)
    print(f"\nAnswer generation complete: {parsed}/{len(results)} parsed ({parsed/len(results)*100:.1f}% new)")

    merged = existing_dicts + [r.model_dump() for r in results]
    out_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    suffix = f"  ({len(existing_dicts)} existing + {len(results)} new = {len(merged)} total)" if existing_dicts else ""
    print(f"Saved → {out_file}{suffix}")
    return [GenerationResult(**r) for r in merged]


def run_generation_phase(
    num_samples: int,
    generation_model: str,
    output_dir: Path,
    strategy: str = "zero_shot",
    batch_label: str = "",
    additional_context: str = "",
    seed: int = 42,
    output_base: Path | None = None,
    overwrite: bool = False,
    samples_per_category: int | None = None,
) -> list[GenerationResult]:
    batch_id = str(uuid.uuid4())
    out_file = output_dir / "generation_results.json"

    # Load existing results unless overwriting
    existing_dicts: list[dict] = []
    existing_ids: set[str] = set()
    if not overwrite and out_file.exists():
        existing_dicts = json.loads(out_file.read_text())
        existing_ids = {r["trace_id"] for r in existing_dicts}

    if generation_model == "mock":
        generator: BenchmarkGenerator = BenchmarkGenerator(
            batch_id=batch_id,
            batch_label=batch_label,
            seed=seed,
            output_base=output_base or output_dir.parent,
        )
        results = generator.generate_batch(num_samples)
    else:
        generator = DIYDatasetGenerator(
            generation_model=generation_model,
            strategy=strategy,
            batch_id=batch_id,
            batch_label=batch_label,
            additional_context=additional_context,
        )

        # Resume: compute how many more needed per category
        remaining_per_category: dict[str, int] | None = None
        if existing_dicts and not overwrite:
            from collections import Counter
            existing_counts = Counter(r["category"] for r in existing_dicts if r.get("parse_error") is None)
            target_per_cat = samples_per_category or (num_samples // len(generator.templates))
            remaining_per_category = {
                t["category"]: max(0, target_per_cat - existing_counts.get(t["category"], 0))
                for t in generator.templates
            }
            still_needed = sum(remaining_per_category.values())
            if still_needed == 0:
                print(f"Resume: all targets met in existing file ({len(existing_dicts)} records). Skipping generation.")
                return [GenerationResult(**r) for r in existing_dicts]
            print(f"Resume: {len(existing_dicts)} existing, generating {still_needed} more.")

        results = generator.generate_batch(
            num_samples=num_samples,
            samples_per_category=samples_per_category,
            remaining_per_category=remaining_per_category,
        )

    # Dedup: drop new results whose trace_id already exists
    new_results = [r for r in results if r.trace_id not in existing_ids]
    dupes = len(results) - len(new_results)
    if dupes:
        print(f"Dedup: dropped {dupes} duplicate trace_id(s)")

    parsed = sum(1 for r in new_results if r.parse_error is None)
    print(f"\nGeneration complete: {parsed}/{len(new_results)} parsed ({parsed/len(new_results)*100:.1f}% new)")

    merged = existing_dicts + [r.model_dump() for r in new_results]
    out_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    suffix = f"  ({len(existing_dicts)} existing + {len(new_results)} new = {len(merged)} total)" if existing_dicts else ""
    print(f"Saved → {out_file}{suffix}")
    return [GenerationResult(**r) for r in merged]
