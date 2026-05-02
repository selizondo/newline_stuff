"""LLM Judge for subtle failure detection in resume-job pairs."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import logfire
from pydantic import BaseModel, Field
from llm_utils import instructor_complete

from ..schema import JobDescription, Resume
from ..utils.trace import generate_trace_id


class JudgmentResult(BaseModel):
    """Structured result from LLM judge evaluation."""

    has_hallucinations: bool = Field(
        description="Whether the resume contains hallucinated or unverifiable claims"
    )
    hallucination_details: Optional[str] = Field(
        None, description="Details about detected hallucinations"
    )
    has_awkward_language: bool = Field(
        description="Whether the resume contains awkward or unnatural language"
    )
    awkward_language_details: Optional[str] = Field(
        None, description="Details about awkward language"
    )
    overall_quality_score: float = Field(
        ge=0.0, le=1.0, description="Overall quality score from 0 to 1"
    )
    fit_assessment: str = Field(
        description="Assessment of how well the resume fits the job"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    red_flags: list[str] = Field(
        default_factory=list, description="Potential red flags identified"
    )


@dataclass
class LLMJudgment:
    """Complete judgment for a resume-job pair."""

    trace_id: str
    resume_trace_id: Optional[str]
    job_trace_id: Optional[str]
    result: JudgmentResult
    judged_at: datetime
    model_used: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "resume_trace_id": self.resume_trace_id,
            "job_trace_id": self.job_trace_id,
            "has_hallucinations": self.result.has_hallucinations,
            "hallucination_details": self.result.hallucination_details,
            "has_awkward_language": self.result.has_awkward_language,
            "awkward_language_details": self.result.awkward_language_details,
            "overall_quality_score": self.result.overall_quality_score,
            "fit_assessment": self.result.fit_assessment,
            "recommendations": self.result.recommendations,
            "red_flags": self.result.red_flags,
            "judged_at": self.judged_at.isoformat(),
            "model_used": self.model_used,
        }


class LLMJudge:
    """LLM-based judge for evaluating resume-job pairs."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
    ):
        """Initialize the LLM judge.

        Args:
            api_key: Groq API key. If not provided, reads from GROQ_API_KEY env var.
            model: Model to use for judgment.
        """
        self.model = model
        logfire.configure()
        logfire.info("LLMJudge initialized", model=model)

        self.judgments: list[LLMJudgment] = []

    def _format_resume_for_judge(self, resume: Resume) -> str:
        """Format a resume for LLM evaluation.

        Args:
            resume: Resume to format.

        Returns:
            Formatted resume text.
        """
        parts = [
            f"Name: {resume.contact.name}",
            f"Location: {resume.contact.location}",
            "",
            "SUMMARY:",
            resume.summary or "No summary provided",
            "",
            "EDUCATION:",
        ]

        for edu in resume.education:
            parts.append(f"- {edu.degree} from {edu.institution} ({edu.graduation_date})")

        parts.append("")
        parts.append("EXPERIENCE:")

        for exp in resume.experience:
            end_date = exp.end_date or "Present"
            parts.append(f"- {exp.title} at {exp.company} ({exp.start_date} to {end_date})")
            for resp in exp.responsibilities[:3]:  # Limit to first 3
                parts.append(f"  * {resp}")

        parts.append("")
        parts.append("SKILLS:")
        for skill in resume.skills:
            parts.append(f"- {skill.name} ({skill.proficiency_level})")

        return "\n".join(parts)

    def _format_job_for_judge(self, job: JobDescription) -> str:
        """Format a job description for LLM evaluation.

        Args:
            job: Job description to format.

        Returns:
            Formatted job text.
        """
        parts = [
            f"Title: {job.title}",
            f"Company: {job.company.name} ({job.company.industry}, {job.company.size})",
            f"Location: {job.company.location}",
            "",
            "DESCRIPTION:",
            job.description[:500],  # Truncate if too long
            "",
            "REQUIREMENTS:",
            f"- Education: {job.requirements.education_requirements}",
            f"- Experience: {job.requirements.experience_years} years",
            f"- Level: {job.requirements.experience_level}",
            "",
            "Required Skills:",
        ]

        for skill in job.requirements.required_skills:
            parts.append(f"- {skill}")

        if job.requirements.preferred_skills:
            parts.append("")
            parts.append("Preferred Skills:")
            for skill in job.requirements.preferred_skills[:5]:  # Limit
                parts.append(f"- {skill}")

        return "\n".join(parts)

    def judge_pair(
        self,
        resume: Resume,
        job: JobDescription,
        pair_trace_id: Optional[str] = None,
    ) -> LLMJudgment:
        """Judge a resume-job pair using LLM.

        Args:
            resume: Candidate resume.
            job: Target job description.
            pair_trace_id: Optional trace ID for the pair.

        Returns:
            LLMJudgment with evaluation results.
        """
        trace_id = pair_trace_id or generate_trace_id("judge")
        resume_trace_id = resume.metadata.trace_id if resume.metadata else None
        job_trace_id = job.metadata.trace_id if job.metadata else None

        resume_text = self._format_resume_for_judge(resume)
        job_text = self._format_job_for_judge(job)

        prompt = f"""You are an expert HR analyst and resume reviewer. Analyze the following resume against the job description and provide a detailed evaluation.

## JOB DESCRIPTION:
{job_text}

## RESUME:
{resume_text}

## EVALUATION CRITERIA:

1. **Hallucinations**: Look for claims that seem exaggerated, unverifiable, or inconsistent:
   - Unrealistic skill claims (e.g., "expert in 50+ technologies")
   - Inconsistent timeline or experience
   - Claims that don't match the education/experience level
   - Generic or templated-sounding achievements

2. **Awkward Language**: Identify unnatural or problematic writing:
   - Excessive buzzwords or jargon
   - Grammatically incorrect sentences
   - Repetitive phrases or word patterns
   - Overly formal or stilted language
   - AI-generated sounding text

3. **Fit Assessment**: Evaluate how well the candidate matches the job:
   - Skills alignment
   - Experience level match
   - Industry relevance

4. **Red Flags**: Note any concerning patterns:
   - Gaps in employment
   - Inconsistent career progression
   - Mismatched expectations

Provide your evaluation in structured format."""

        with logfire.span("llm_judge", trace_id=trace_id):
            result = instructor_complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR analyst providing objective, "
                        "thorough resume evaluations. Be specific and constructive.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=JudgmentResult,
                model=self.model,
            )

            judgment = LLMJudgment(
                trace_id=trace_id,
                resume_trace_id=resume_trace_id,
                job_trace_id=job_trace_id,
                result=result,
                judged_at=datetime.utcnow(),
                model_used=self.model,
            )

            self.judgments.append(judgment)

            logfire.info(
                "LLM judgment complete",
                trace_id=trace_id,
                has_hallucinations=result.has_hallucinations,
                has_awkward_language=result.has_awkward_language,
                quality_score=result.overall_quality_score,
            )

        return judgment

    def judge_batch(
        self,
        pairs: list[tuple[Resume, JobDescription]],
    ) -> list[LLMJudgment]:
        """Judge multiple resume-job pairs.

        Args:
            pairs: List of (resume, job) tuples.

        Returns:
            List of LLMJudgment results.
        """
        results = []

        with logfire.span("judge_batch", count=len(pairs)):
            for i, (resume, job) in enumerate(pairs):
                try:
                    judgment = self.judge_pair(resume, job)
                    results.append(judgment)
                    logfire.info(f"Judged pair {i + 1}/{len(pairs)}")
                except Exception as e:
                    logfire.error(f"Failed to judge pair {i + 1}", error=str(e))

        return results

    def get_statistics(self) -> dict:
        """Get statistics on judgments.

        Returns:
            Dictionary of statistics.
        """
        if not self.judgments:
            return {
                "total_judged": 0,
                "hallucination_rate": 0.0,
                "awkward_language_rate": 0.0,
                "average_quality_score": 0.0,
            }

        total = len(self.judgments)

        return {
            "total_judged": total,
            "hallucination_rate": sum(
                1 for j in self.judgments if j.result.has_hallucinations
            )
            / total,
            "awkward_language_rate": sum(
                1 for j in self.judgments if j.result.has_awkward_language
            )
            / total,
            "average_quality_score": sum(
                j.result.overall_quality_score for j in self.judgments
            )
            / total,
            "average_red_flags": sum(len(j.result.red_flags) for j in self.judgments)
            / total,
        }

    def reset(self) -> None:
        """Reset all judgments."""
        self.judgments = []
