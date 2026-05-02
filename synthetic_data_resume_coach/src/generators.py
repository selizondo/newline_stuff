"""Resume and job description generators (jobs-first pipeline flow)."""

import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import logfire
from openai import RateLimitError
from llm_utils import instructor_complete

from .prompts import load_resume_prompt_templates
from .schema import (
    FitLevel,
    JobDescription,
    JobDescriptionMetadata,
    Resume,
    ResumeJobPair,
    ResumeJobPairMetadata,
    ResumeMetadata,
)
from .utils.storage import get_timestamped_filename, save_jsonl
from .utils.trace import generate_trace_id


# ── Shared constants ───────────────────────────────────────────────────────────

INDUSTRIES = [
    "Technology",
    "Healthcare",
    "Finance",
    "Education",
    "Marketing",
    "Engineering",
    "Data Science",
    "Design",
    "Sales",
    "Operations",
    "E-commerce",
    "Manufacturing",
    "Consulting",
    "Media",
]

SENIORITY_LEVELS = ["Entry", "Mid", "Senior", "Lead", "Executive"]

REMOTE_POLICIES = ["Remote", "Hybrid", "On-site"]

JOB_TYPES = ["Full-time", "Part-time", "Contract", "Internship"]

EXPERIENCE_LEVELS = [
    "Entry-level (0-2 years)",
    "Mid-level (3-5 years)",
    "Senior (6-10 years)",
    "Lead/Principal (10+ years)",
]

# Maps required experience years → EXPERIENCE_LEVELS label
def _years_to_exp_level(years: int) -> str:
    if years <= 2:
        return EXPERIENCE_LEVELS[0]
    if years <= 5:
        return EXPERIENCE_LEVELS[1]
    if years <= 10:
        return EXPERIENCE_LEVELS[2]
    return EXPERIENCE_LEVELS[3]


# Maps required experience years → SENIORITY_LEVELS label
def _years_to_seniority(years: int) -> str:
    if years <= 2:
        return "Entry"
    if years <= 5:
        return "Mid"
    if years <= 10:
        return "Senior"
    return "Lead"


NICHE_ROLES = {
    "blockchain", "quantum computing", "ai ethics", "robotics",
    "bioinformatics", "cryptography", "embedded systems", "fpga",
    "compiler", "kernel", "hpc", "mlops", "devsecops", "sre", "reliability",
}

FIT_INSTRUCTIONS: dict[FitLevel, str] = {
    FitLevel.EXCELLENT: "Include ALL required skills with Expert proficiency.",
    FitLevel.GOOD:      "Include 80% of required skills with Advanced proficiency.",
    FitLevel.PARTIAL:   "Include about 50% of required skills with varying proficiency.",
    FitLevel.POOR:      "Include only 20-30% of required skills, mostly at Beginner level.",
    FitLevel.MISMATCH:  "Include skills from a completely different field.",
}


def _is_niche_role(title: str) -> bool:
    title_lower = title.lower()
    return any(niche in title_lower for niche in NICHE_ROLES)


# ── ResumeGenerator ────────────────────────────────────────────────────────────

class ResumeGenerator:
    """Generate synthetic resumes using LLM with structured output."""

    _prompt_templates_cache: dict[str, dict] | None = None

    @classmethod
    def _templates(cls) -> dict[str, dict]:
        if cls._prompt_templates_cache is None:
            cls._prompt_templates_cache = load_resume_prompt_templates()
        return cls._prompt_templates_cache

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        logfire.configure()
        logfire.info("ResumeGenerator initialized", model=model)

    def generate_single(
        self,
        industry: Optional[str] = None,
        experience_level: Optional[str] = None,
        specific_role: Optional[str] = None,
        prompt_template: Optional[str] = None,
        target_job_trace_id: Optional[str] = None,
        fit_level: Optional[FitLevel] = None,
    ) -> Resume:
        """Generate a single synthetic resume from a named prompt template."""
        industry = industry or random.choice(INDUSTRIES)
        experience_level = experience_level or random.choice(EXPERIENCE_LEVELS)
        templates = self._templates()
        prompt_template = prompt_template or random.choice(list(templates.keys()))

        tmpl = templates.get(prompt_template, templates["formal"])
        role_context = f" specifically for a {specific_role} role" if specific_role else ""

        trace_id = generate_trace_id("resume")
        with logfire.span("generate_resume", trace_id=trace_id):
            resume = instructor_complete(
                messages=[
                    {"role": "system", "content": tmpl["system"]},
                    {"role": "user", "content": tmpl["user"].format(
                        industry=industry,
                        experience_level=experience_level,
                        role_context=role_context,
                    )},
                ],
                response_model=Resume,
                model=self.model,
            )
            resume.metadata = ResumeMetadata(
                trace_id=trace_id,
                generated_at=datetime.utcnow(),
                prompt_template=prompt_template,
                target_industry=industry,
                target_seniority=experience_level,
                fit_level=fit_level,
                target_job_trace_id=target_job_trace_id,
            )
            logfire.info("Resume generated", trace_id=trace_id, industry=industry,
                         experience_level=experience_level, prompt_template=prompt_template)
        return resume

    def generate_for_job(
        self,
        job_trace_id: str,
        job_title: str,
        required_skills: list[str],
        industry: str,
        experience_years: int,
        fit_level: FitLevel = FitLevel.GOOD,
    ) -> Resume:
        """Generate a resume tailored to a specific job with a controlled fit level."""
        exp_level = _years_to_exp_level(experience_years)
        fit_match = (
            "matches well with" if fit_level in (FitLevel.EXCELLENT, FitLevel.GOOD)
            else "partially matches" if fit_level == FitLevel.PARTIAL
            else "does not match"
        )
        prompt_template = random.choice(list(self._templates().keys()))
        trace_id = generate_trace_id("resume")

        user_prompt = (
            f"Generate a resume for a candidate applying for a {job_title} position\n"
            f"in the {industry} industry requiring {experience_years} years of experience.\n\n"
            f"Required skills for this job: {', '.join(required_skills)}\n\n"
            f"{FIT_INSTRUCTIONS.get(fit_level, FIT_INSTRUCTIONS[FitLevel.GOOD])}\n\n"
            f"Create a realistic resume that {fit_match} this job.\n\n"
            "Include:\n"
            "1. Contact information (realistic but fake)\n"
            "2. Professional summary\n"
            "3. Education (1-2 entries)\n"
            "4. Work experience (2-4 entries)\n"
            "5. Skills (with proficiency levels)\n"
            "6. Certifications (if applicable)\n\n"
            "Use ISO date format (YYYY-MM-DD). For current positions, set end_date to null."
        )

        with logfire.span("generate_resume_for_job", trace_id=trace_id, job_trace_id=job_trace_id):
            resume = instructor_complete(
                messages=[
                    {"role": "system", "content": (
                        "You are a professional resume writer creating realistic synthetic "
                        "resumes. Use ISO date format (YYYY-MM-DD)."
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=Resume,
                model=self.model,
            )
            resume.metadata = ResumeMetadata(
                trace_id=trace_id,
                generated_at=datetime.utcnow(),
                prompt_template=prompt_template,
                target_industry=industry,
                target_seniority=exp_level,
                fit_level=fit_level,
                target_job_trace_id=job_trace_id,
            )
            logfire.info("Resume generated for job", trace_id=trace_id,
                         job_trace_id=job_trace_id, fit_level=fit_level.value)
        return resume

    def generate_batch(
        self,
        count: int = 10,
        industries: Optional[list[str]] = None,
        experience_levels: Optional[list[str]] = None,
    ) -> list[Resume]:
        """Generate standalone resumes (not tied to a specific job)."""
        industries = industries or INDUSTRIES
        experience_levels = experience_levels or EXPERIENCE_LEVELS
        template_names = list(self._templates().keys())

        resumes = []
        with logfire.span("generate_resume_batch", count=count):
            for i in range(count):
                try:
                    resume = self.generate_single(
                        industry=random.choice(industries),
                        experience_level=random.choice(experience_levels),
                        prompt_template=random.choice(template_names),
                    )
                    resumes.append(resume)
                    logfire.info(f"Generated resume {i + 1}/{count}")
                except Exception as e:
                    logfire.error(f"Failed to generate resume {i + 1}", error=str(e))
        return resumes

    def save_resumes(
        self,
        resumes: list[Resume],
        output_dir: str = "data/generated",
        filename: Optional[str] = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / (filename or get_timestamped_filename("resumes", "jsonl"))
        save_jsonl(resumes, file_path)
        logfire.info(f"Saved {len(resumes)} resumes to {file_path}")
        return file_path


# ── JobDescriptionGenerator ────────────────────────────────────────────────────

class JobDescriptionGenerator:
    """Generate synthetic job descriptions using LLM with structured output."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        logfire.configure()
        logfire.info("JobDescriptionGenerator initialized", model=model)

    def generate_single(
        self,
        title: Optional[str] = None,
        industry: Optional[str] = None,
        seniority_level: Optional[str] = None,
        remote_policy: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> JobDescription:
        """Generate a single synthetic job description."""
        seniority_level = seniority_level or random.choice(SENIORITY_LEVELS)
        remote_policy = remote_policy or random.choice(REMOTE_POLICIES)
        industry = industry or random.choice(INDUSTRIES)

        title_context = f"for a {title} position" if title else "for a relevant position"
        trace_id = generate_trace_id("job")

        user_prompt = (
            f"Generate a realistic, detailed job description {title_context} in the {industry} industry.\n\n"
            f"The position should be at the {seniority_level} level with a {remote_policy} work arrangement.\n\n"
            "Include:\n"
            "1. Company information (realistic but fictional company)\n"
            "2. Detailed job description (at least 100 words)\n"
            "3. Requirements (required skills, preferred skills, education, years of experience)\n"
            "4. Key responsibilities (5-8 bullet points)\n"
            "5. Benefits and perks (4-6 items)\n"
            "6. Salary range (realistic for the role and level)\n\n"
            "Make the job posting professional and appealing to candidates.\n"
            f"The company size should be one of: Startup, Small, Medium, Large, Enterprise.\n"
            f"Experience level should be: {seniority_level}"
        )

        with logfire.span("generate_job_description", trace_id=trace_id):
            job = instructor_complete(
                messages=[
                    {"role": "system", "content": (
                        "You are a professional HR specialist creating realistic job postings "
                        "for training data. Do not include a metadata field."
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=JobDescription,
                model=self.model,
            )
            job.metadata = JobDescriptionMetadata(
                trace_id=trace_id,
                generated_at=datetime.utcnow(),
                prompt_template=prompt_template or "default",
                is_niche_role=_is_niche_role(job.title),
            )
            logfire.info("Job description generated", trace_id=trace_id, title=job.title,
                         company=job.company.name, seniority=seniority_level,
                         is_niche=job.metadata.is_niche_role)
        return job

    def generate_batch(
        self,
        count: int = 10,
        seniority_levels: Optional[list[str]] = None,
        industries: Optional[list[str]] = None,
    ) -> list[JobDescription]:
        """Generate a batch of job descriptions, stratified by seniority and industry."""
        seniority_levels = seniority_levels or SENIORITY_LEVELS
        industries = industries or INDUSTRIES

        jobs = []
        with logfire.span("generate_job_batch", count=count):
            for i in range(count):
                try:
                    job = self.generate_single(
                        seniority_level=random.choice(seniority_levels),
                        industry=random.choice(industries),
                    )
                    jobs.append(job)
                    logfire.info(f"Generated job {i + 1}/{count}")
                except Exception as e:
                    logfire.error(f"Failed to generate job {i + 1}", error=str(e))
        return jobs

    def generate_with_multiple_resumes(
        self,
        job: JobDescription,
        resume_generator: ResumeGenerator,
        resumes_per_job: int = 5,
        fit_levels: Optional[list[FitLevel]] = None,
    ) -> list[ResumeJobPair]:
        """Generate resumes at each fit level for a single job (jobs-first flow)."""
        fit_levels = fit_levels or list(FitLevel)
        job_trace_id = job.metadata.trace_id if job.metadata else generate_trace_id("job")
        pairs = []

        for i in range(resumes_per_job):
            fit_level = fit_levels[i % len(fit_levels)]
            try:
                resume = resume_generator.generate_for_job(
                    job_trace_id=job_trace_id,
                    job_title=job.title,
                    required_skills=job.requirements.required_skills,
                    industry=job.company.industry,
                    experience_years=job.requirements.experience_years,
                    fit_level=fit_level,
                )
                pair = ResumeJobPair(
                    resume=resume,
                    job_description=job,
                    metadata=ResumeJobPairMetadata(
                        trace_id=generate_trace_id("pair"),
                        generated_at=datetime.utcnow(),
                        fit_level=fit_level.value,
                    ),
                )
                pairs.append(pair)
                logfire.info(f"Generated resume {i + 1}/{resumes_per_job} for job",
                             job_trace_id=job_trace_id, fit_level=fit_level.value)
            except RateLimitError:
                raise
            except Exception as e:
                logfire.error(f"Failed to generate resume {i + 1} for job",
                              error=str(e), job_trace_id=job_trace_id)
        return pairs

    def save_jobs(
        self,
        jobs: list[JobDescription],
        output_dir: str = "data/generated",
        filename: Optional[str] = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / (filename or get_timestamped_filename("jobs", "jsonl"))
        save_jsonl(jobs, file_path)
        logfire.info(f"Saved {len(jobs)} job descriptions to {file_path}")
        return file_path

    def save_pairs(
        self,
        pairs: list[ResumeJobPair],
        output_dir: str = "data/generated",
        filename: Optional[str] = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / (filename or get_timestamped_filename("pairs", "jsonl"))
        save_jsonl(pairs, file_path)
        logfire.info(f"Saved {len(pairs)} pairs to {file_path}")
        return file_path
