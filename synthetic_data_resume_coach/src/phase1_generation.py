"""Phase 1 — Job description and resume pair generation with checkpoint recovery.

Progress is written incrementally: each generated job and pair is appended to
the output JSONL files immediately after it is created. If the pipeline crashes
or is killed, re-running with --resume <run_label> picks up exactly where it
left off.

Resume logic:
  - Jobs:  skip the first N that already exist in jobs_<run_label>.jsonl
  - Pairs: for each job, count how many pairs already exist for that job's
           trace_id. Generate only the missing fit levels.
"""

from pathlib import Path
from typing import Optional

from openai import RateLimitError

from .generators import JobDescriptionGenerator, ResumeGenerator
from .schema import FitLevel, JobDescription, Resume, ResumeJobPair

_CIRCUIT_BREAKER_THRESHOLD = 2  # consecutive rate-limit failures before stopping

FIT_LEVELS = [
    FitLevel.EXCELLENT,
    FitLevel.GOOD,
    FitLevel.PARTIAL,
    FitLevel.POOR,
    FitLevel.MISMATCH,
]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_jsonl(path: Path, model_cls) -> list:
    """Load records from an existing JSONL checkpoint; skip malformed lines."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(model_cls.model_validate_json(line))
        except Exception:
            pass
    return out


def _append_jsonl(path: Path, record) -> None:
    """Append one Pydantic record to a JSONL file immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")


def _pairs_by_job(pairs: list[ResumeJobPair]) -> dict[str, list[ResumeJobPair]]:
    """Index existing pairs by the job trace_id stored in resume metadata."""
    index: dict[str, list[ResumeJobPair]] = {}
    for pair in pairs:
        jid = (
            pair.resume.metadata.target_job_trace_id
            if pair.resume and pair.resume.metadata
            else ""
        )
        index.setdefault(jid, []).append(pair)
    return index


# ── Main phase runner ──────────────────────────────────────────────────────────

def run_generation_phase(
    num_jobs: int,
    resumes_per_job: int,
    model: str,
    output_dir: str,
    run_label: str,
    industries: Optional[list[str]] = None,
) -> dict:
    """Generate job descriptions and matched resumes with incremental checkpointing.

    On first run: generates everything from scratch, appending each item
    immediately so progress is never lost.

    On resumed run (same run_label, output files already exist): loads
    existing progress and skips already-completed items.

    Args:
        num_jobs:        Total number of job descriptions to generate.
        resumes_per_job: Number of resumes per job (one per fit level, cycling).
        model:           LLM model name.
        output_dir:      Base output directory.
        run_label:       Timestamp label; must match prior run when resuming.
        industries:      Optional industry filter.

    Returns:
        Dict with keys: jobs, pairs, resumes, files.
    """
    job_gen = JobDescriptionGenerator(model=model)
    resume_gen = ResumeGenerator(model=model)
    generated_dir = Path(output_dir) / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    jobs_file    = generated_dir / f"jobs_{run_label}.jsonl"
    pairs_file   = generated_dir / f"pairs_{run_label}.jsonl"
    resumes_file = generated_dir / f"resumes_{run_label}.jsonl"

    # ── Load existing progress ─────────────────────────────────────────────────
    jobs: list[JobDescription] = _load_jsonl(jobs_file, JobDescription)
    all_pairs: list[ResumeJobPair] = _load_jsonl(pairs_file, ResumeJobPair)

    if jobs or all_pairs:
        print(f"  Resuming: {len(jobs)} jobs + {len(all_pairs)} pairs already on disk")

    # ── Step 1.1: Generate remaining job descriptions ──────────────────────────
    jobs_needed = num_jobs - len(jobs)
    if jobs_needed > 0:
        print(f"  Generating {jobs_needed} job descriptions "
              f"({len(jobs)}/{num_jobs} already done)...")
        consecutive_rl = 0
        for i in range(len(jobs), num_jobs):
            try:
                job = job_gen.generate_single(
                    industry=industries[i % len(industries)] if industries else None
                )
                consecutive_rl = 0
                jobs.append(job)
                _append_jsonl(jobs_file, job)
                print(f"  [{i+1}/{num_jobs}] {job.title} @ {job.company.name}... OK")
            except RateLimitError as e:
                consecutive_rl += 1
                print(f"  [{i+1}/{num_jobs}] rate limit ({consecutive_rl}/{_CIRCUIT_BREAKER_THRESHOLD}): {e!s:.60}")
                if consecutive_rl >= _CIRCUIT_BREAKER_THRESHOLD:
                    print(f"  Circuit breaker tripped — resume with --resume {run_label}")
                    break
            except Exception as e:
                print(f"  [{i+1}/{num_jobs}] generation failed: {e!s:.80}")
    else:
        print(f"  Jobs: {len(jobs)}/{num_jobs} already complete — skipping job generation")

    # ── Step 1.2: Generate missing resumes for each job ────────────────────────
    job_pairs = _pairs_by_job(all_pairs)

    # Count how many pairs are still needed across all jobs.
    total_needed = sum(
        max(0, resumes_per_job - len(job_pairs.get(
            job.metadata.trace_id if job.metadata else "", []
        )))
        for job in jobs
    )
    print(f"\n  Generating resumes ({total_needed} needed across {len(jobs)} jobs)...")

    consecutive_rl = 0
    for i, job in enumerate(jobs):
        job_trace_id = job.metadata.trace_id if job.metadata else ""
        existing = job_pairs.get(job_trace_id, [])
        already_done = len(existing)

        if already_done >= resumes_per_job:
            fit_summary = ", ".join(
                p.metadata.fit_level for p in existing if p.metadata
            )
            print(f"  [{i+1}/{len(jobs)}] {job.title} → already complete "
                  f"({already_done} resumes: {fit_summary})")
            continue

        # Determine which fit levels are still missing.
        done_fit_levels = {
            p.metadata.fit_level for p in existing if p.metadata
        }
        remaining = [
            fl for fl in FIT_LEVELS if fl.value not in done_fit_levels
        ][: resumes_per_job - already_done]

        try:
            new_pairs = job_gen.generate_with_multiple_resumes(
                job=job,
                resume_generator=resume_gen,
                resumes_per_job=len(remaining),
                fit_levels=remaining,
            )
            consecutive_rl = 0
            for pair in new_pairs:
                all_pairs.append(pair)
                _append_jsonl(pairs_file, pair)
                _append_jsonl(resumes_file, pair.resume)

            fit_summary = ", ".join(
                p.metadata.fit_level for p in (existing + new_pairs) if p.metadata
            )
            print(f"  [{i+1}/{len(jobs)}] {job.title} → "
                  f"{already_done + len(new_pairs)} resumes [{fit_summary}]")
        except RateLimitError as e:
            consecutive_rl += 1
            print(f"  [{i+1}/{len(jobs)}] rate limit ({consecutive_rl}/{_CIRCUIT_BREAKER_THRESHOLD}): {e!s:.60}")
            if consecutive_rl >= _CIRCUIT_BREAKER_THRESHOLD:
                print(f"  Circuit breaker tripped — resume with --resume {run_label}")
                break
        except Exception as e:
            print(f"  [{i+1}/{len(jobs)}] resume generation failed: {e!s:.80}")

    resumes: list[Resume] = [p.resume for p in all_pairs]
    print(f"\n  Phase 1 complete: {len(jobs)} jobs, {len(all_pairs)} pairs generated")

    return {
        "jobs": jobs,
        "pairs": all_pairs,
        "resumes": resumes,
        "files": {
            "jobs":    str(jobs_file),
            "pairs":   str(pairs_file),
            "resumes": str(resumes_file),
        },
    }
