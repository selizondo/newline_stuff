"""Failure labeling module for resume-job pair analysis."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import logfire
import pandas as pd

from ..schema import JobDescription, Resume, ResumeJobPair
from ..utils.storage import save_jsonl, get_timestamped_filename


@dataclass
class FailureLabels:
    """Failure labels for a resume-job pair."""

    trace_id: str
    resume_trace_id: Optional[str]
    job_trace_id: Optional[str]

    # Core metrics (0 = pass, 1 = fail)
    skills_overlap_ratio: float  # 0.0 to 1.0 (Jaccard similarity)
    experience_mismatch: int  # 0 or 1
    seniority_mismatch: int  # 0 or 1
    missing_core_skill: int  # 0 or 1
    hallucinated_skill: int  # 0 or 1 (skill claimed but not demonstrable)
    awkward_language_flag: int  # 0 or 1

    # Additional context
    missing_skills: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    experience_gap: int = 0  # Difference in years
    seniority_gap: str = ""  # e.g., "Entry vs Senior"

    # Metadata
    labeled_at: Optional[datetime] = None
    prompt_template: Optional[str] = None
    fit_level: Optional[str] = None
    is_niche_role: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "resume_trace_id": self.resume_trace_id,
            "job_trace_id": self.job_trace_id,
            "skills_overlap_ratio": self.skills_overlap_ratio,
            "experience_mismatch": self.experience_mismatch,
            "seniority_mismatch": self.seniority_mismatch,
            "missing_core_skill": self.missing_core_skill,
            "hallucinated_skill": self.hallucinated_skill,
            "awkward_language_flag": self.awkward_language_flag,
            "missing_skills": self.missing_skills,
            "matched_skills": self.matched_skills,
            "experience_gap": self.experience_gap,
            "seniority_gap": self.seniority_gap,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
            "prompt_template": self.prompt_template,
            "fit_level": self.fit_level,
            "is_niche_role": self.is_niche_role,
        }

    @property
    def overall_pass(self) -> bool:
        """Check if all failure checks pass."""
        return (
            self.skills_overlap_ratio >= 0.5
            and self.experience_mismatch == 0
            and self.seniority_mismatch == 0
            and self.missing_core_skill == 0
            and self.hallucinated_skill == 0
            and self.awkward_language_flag == 0
        )

    @property
    def failure_count(self) -> int:
        """Count total number of failures."""
        return (
            (1 if self.skills_overlap_ratio < 0.5 else 0)
            + self.experience_mismatch
            + self.seniority_mismatch
            + self.missing_core_skill
            + self.hallucinated_skill
            + self.awkward_language_flag
        )


class FailureLabeler:
    """Label failure modes for resume-job pairs."""

    # Seniority level ordering for comparison
    SENIORITY_ORDER = {
        "entry": 0,
        "junior": 0,
        "mid": 1,
        "intermediate": 1,
        "senior": 2,
        "lead": 3,
        "principal": 3,
        "executive": 4,
        "director": 4,
        "vp": 5,
        "c-level": 5,
    }

    # Patterns for awkward language detection
    AWKWARD_PATTERNS = [
        r"\b(synergy|synergize|synergistic)\b",
        r"\b(leverage|leveraging)\s+\w+\s+to\s+\w+",
        r"\b(utilize|utilization)\b",
        r"\b(paradigm|paradigm shift)\b",
        r"\b(proactive|proactively)\b.*\b(proactive|proactively)\b",  # Repeated
        r"\b(thinking outside the box)\b",
        r"\b(move the needle)\b",
        r"\b(low-hanging fruit)\b",
        r"\b(circle back)\b",
        r"\b(deep dive)\b.*\b(deep dive)\b",  # Repeated
        r"(\w+)\s+\1\s+\1",  # Triple word repetition
        r"[.!?]{2,}",  # Multiple punctuation
        r"\b(responsible for being responsible)\b",
        r"\b(managed management)\b",
        r"\b(led leadership)\b",
    ]

    # Common hallucination indicators
    HALLUCINATION_INDICATORS = [
        "certified in everything",
        "expert in all",
        "mastered every",
        "proficient in 50+",
        "fluent in 10+",
        "20+ years experience",  # In entry-level context
    ]

    def __init__(self):
        """Initialize the failure labeler."""
        logfire.configure()
        self.labels: list[FailureLabels] = []

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets.

        Args:
            set1: First set.
            set2: Second set.

        Returns:
            Jaccard similarity coefficient (0.0 to 1.0).
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill name for comparison.

        Args:
            skill: Skill name to normalize.

        Returns:
            Normalized skill name.
        """
        # Lowercase and remove common variations
        normalized = skill.lower().strip()
        # Remove version numbers
        normalized = re.sub(r"\s*\d+(\.\d+)*\s*", "", normalized)
        # Remove common suffixes
        for suffix in [".js", ".py", " developer", " engineer", " programming"]:
            normalized = normalized.replace(suffix, "")
        return normalized.strip()

    def extract_resume_skills(self, resume: Resume) -> set[str]:
        """Extract normalized skills from a resume.

        Args:
            resume: Resume to extract skills from.

        Returns:
            Set of normalized skill names.
        """
        skills = set()
        for skill in resume.skills:
            skills.add(self.normalize_skill(skill.name))
        return skills

    def extract_job_skills(self, job: JobDescription) -> tuple[set[str], set[str]]:
        """Extract normalized skills from a job description.

        Args:
            job: Job description to extract skills from.

        Returns:
            Tuple of (required_skills, preferred_skills) as normalized sets.
        """
        required = {self.normalize_skill(s) for s in job.requirements.required_skills}
        preferred = {self.normalize_skill(s) for s in job.requirements.preferred_skills}
        return required, preferred

    def calculate_skills_overlap(
        self, resume: Resume, job: JobDescription
    ) -> tuple[float, list[str], list[str]]:
        """Calculate skills overlap ratio using Jaccard similarity.

        Args:
            resume: Candidate resume.
            job: Target job description.

        Returns:
            Tuple of (overlap_ratio, missing_skills, matched_skills).
        """
        resume_skills = self.extract_resume_skills(resume)
        required_skills, preferred_skills = self.extract_job_skills(job)

        # Calculate overlap with required skills
        all_job_skills = required_skills | preferred_skills
        matched = resume_skills & all_job_skills
        missing = required_skills - resume_skills

        # Jaccard similarity on required skills
        overlap_ratio = self.jaccard_similarity(resume_skills, required_skills)

        return overlap_ratio, list(missing), list(matched)

    def detect_experience_mismatch(
        self, resume: Resume, job: JobDescription
    ) -> tuple[bool, int]:
        """Detect if there's an experience mismatch.

        Args:
            resume: Candidate resume.
            job: Target job description.

        Returns:
            Tuple of (is_mismatch, gap_in_years).
        """
        # Estimate resume experience years
        resume_years = 0
        for exp in resume.experience:
            if exp.end_date:
                years = (exp.end_date - exp.start_date).days / 365
            else:
                # Current job - estimate to now
                from datetime import date

                years = (date.today() - exp.start_date).days / 365
            resume_years += years

        required_years = job.requirements.experience_years

        # Calculate gap
        gap = required_years - int(resume_years)

        # Allow some flexibility (±1 year is acceptable)
        is_mismatch = gap > 1 or resume_years < 0.5 * required_years

        return is_mismatch, max(0, gap)

    def detect_seniority_mismatch(
        self, resume: Resume, job: JobDescription
    ) -> tuple[bool, str]:
        """Detect if there's a seniority level mismatch.

        Args:
            resume: Candidate resume.
            job: Target job description.

        Returns:
            Tuple of (is_mismatch, gap_description).
        """
        # Get job seniority level
        job_level = job.requirements.experience_level.lower()
        job_order = self.SENIORITY_ORDER.get(job_level, 1)

        # Estimate resume seniority from most recent job title
        resume_level = "mid"  # Default
        if resume.experience:
            title = resume.experience[0].title.lower()
            for level, order in self.SENIORITY_ORDER.items():
                if level in title:
                    resume_level = level
                    break
            # Also check for junior/senior keywords
            if "junior" in title or "jr" in title:
                resume_level = "entry"
            elif "senior" in title or "sr" in title:
                resume_level = "senior"
            elif "lead" in title or "principal" in title:
                resume_level = "lead"

        resume_order = self.SENIORITY_ORDER.get(resume_level, 1)

        # Check if gap is more than 1 level
        gap = abs(job_order - resume_order)
        is_mismatch = gap > 1

        gap_description = ""
        if is_mismatch:
            gap_description = f"{resume_level.capitalize()} vs {job_level.capitalize()}"

        return is_mismatch, gap_description

    def detect_missing_core_skill(
        self, resume: Resume, job: JobDescription
    ) -> bool:
        """Detect if resume is missing a core/critical skill.

        Core skills are typically the first 3 required skills or
        skills mentioned in the job title.

        Args:
            resume: Candidate resume.
            job: Target job description.

        Returns:
            True if missing a core skill.
        """
        resume_skills = self.extract_resume_skills(resume)
        required_skills, _ = self.extract_job_skills(job)

        # First 3 required skills are considered core
        core_skills = list(required_skills)[:3]

        # Check if any core skill is missing
        for core_skill in core_skills:
            if core_skill not in resume_skills:
                # Check for partial matches
                partial_match = any(
                    core_skill in rs or rs in core_skill for rs in resume_skills
                )
                if not partial_match:
                    return True

        return False

    def detect_hallucinated_skill(self, resume: Resume) -> bool:
        """Detect potentially hallucinated skills.

        A hallucinated skill is one that seems unrealistic or
        inconsistent with the rest of the resume.

        Args:
            resume: Resume to check.

        Returns:
            True if potential hallucination detected.
        """
        # Check for obvious hallucination patterns
        resume_text = (resume.summary or "") + " ".join(
            " ".join(exp.responsibilities) for exp in resume.experience
        )
        resume_text_lower = resume_text.lower()

        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator in resume_text_lower:
                return True

        # Check for unrealistic skill counts
        if len(resume.skills) > 20:
            # Too many skills might indicate hallucination
            expert_count = sum(
                1 for s in resume.skills if s.proficiency_level.lower() == "expert"
            )
            if expert_count > 10:
                return True

        # Check for skills with unrealistic years of experience
        for skill in resume.skills:
            if skill.years_experience and skill.years_experience > 25:
                return True

        # Check if entry-level resume claims too many advanced skills
        total_exp_years = sum(
            (
                (exp.end_date or datetime.now().date()) - exp.start_date
            ).days
            / 365
            for exp in resume.experience
        )

        if total_exp_years < 2:  # Entry-level
            expert_count = sum(
                1 for s in resume.skills if s.proficiency_level.lower() == "expert"
            )
            if expert_count > 2:
                return True

        return False

    def detect_awkward_language(self, resume: Resume) -> bool:
        """Detect awkward or overly buzzword-heavy language.

        Args:
            resume: Resume to check.

        Returns:
            True if awkward language detected.
        """
        # Combine all text from resume
        text_parts = [resume.summary or ""]
        for exp in resume.experience:
            text_parts.extend(exp.responsibilities)
            text_parts.extend(exp.achievements)

        full_text = " ".join(text_parts)

        # Check for awkward patterns
        for pattern in self.AWKWARD_PATTERNS:
            if re.search(pattern, full_text, re.IGNORECASE):
                return True

        # Check for excessive buzzword density
        buzzwords = [
            "leverage",
            "synergy",
            "paradigm",
            "innovative",
            "disruptive",
            "scalable",
            "robust",
            "cutting-edge",
            "best-in-class",
            "world-class",
            "game-changing",
        ]
        buzzword_count = sum(
            1 for word in buzzwords if word in full_text.lower()
        )

        # More than 5 buzzwords in a single resume is suspicious
        if buzzword_count > 5:
            return True

        return False

    def label_pair(
        self, resume: Resume, job: JobDescription, pair_trace_id: Optional[str] = None
    ) -> FailureLabels:
        """Label failure modes for a resume-job pair.

        Args:
            resume: Candidate resume.
            job: Target job description.
            pair_trace_id: Optional trace ID for the pair.

        Returns:
            FailureLabels with all metrics calculated.
        """
        from ..utils.trace import generate_trace_id

        trace_id = pair_trace_id or generate_trace_id("label")
        resume_trace_id = resume.metadata.trace_id if resume.metadata else None
        job_trace_id = job.metadata.trace_id if job.metadata else None

        with logfire.span("label_failure_modes", trace_id=trace_id):
            # Calculate all metrics
            skills_overlap, missing_skills, matched_skills = self.calculate_skills_overlap(
                resume, job
            )
            exp_mismatch, exp_gap = self.detect_experience_mismatch(resume, job)
            seniority_mismatch, seniority_gap = self.detect_seniority_mismatch(resume, job)
            missing_core = self.detect_missing_core_skill(resume, job)
            hallucinated = self.detect_hallucinated_skill(resume)
            awkward = self.detect_awkward_language(resume)

            labels = FailureLabels(
                trace_id=trace_id,
                resume_trace_id=resume_trace_id,
                job_trace_id=job_trace_id,
                skills_overlap_ratio=skills_overlap,
                experience_mismatch=1 if exp_mismatch else 0,
                seniority_mismatch=1 if seniority_mismatch else 0,
                missing_core_skill=1 if missing_core else 0,
                hallucinated_skill=1 if hallucinated else 0,
                awkward_language_flag=1 if awkward else 0,
                missing_skills=missing_skills,
                matched_skills=matched_skills,
                experience_gap=exp_gap,
                seniority_gap=seniority_gap,
                labeled_at=datetime.utcnow(),
                prompt_template=resume.metadata.prompt_template if resume.metadata else None,
                fit_level=resume.metadata.fit_level.value
                if resume.metadata and resume.metadata.fit_level
                else None,
                is_niche_role=job.metadata.is_niche_role if job.metadata else False,
            )

            self.labels.append(labels)

            logfire.info(
                "Labeled failure modes",
                trace_id=trace_id,
                overall_pass=labels.overall_pass,
                failure_count=labels.failure_count,
            )

        return labels

    def label_pairs(self, pairs: list[ResumeJobPair]) -> list[FailureLabels]:
        """Label failure modes for multiple resume-job pairs.

        Args:
            pairs: List of ResumeJobPair objects.

        Returns:
            List of FailureLabels.
        """
        results = []

        with logfire.span("label_pairs", count=len(pairs)):
            for i, pair in enumerate(pairs):
                pair_trace_id = pair.metadata.trace_id if pair.metadata else None
                labels = self.label_pair(pair.resume, pair.job_description, pair_trace_id)
                results.append(labels)
                logfire.info(f"Labeled pair {i + 1}/{len(pairs)}")

        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all labels to a pandas DataFrame.

        Returns:
            DataFrame with all failure labels.
        """
        if not self.labels:
            return pd.DataFrame()

        data = [label.to_dict() for label in self.labels]
        return pd.DataFrame(data)

    def get_statistics(self) -> dict:
        """Get statistics on failure labels.

        Returns:
            Dictionary of statistics.
        """
        if not self.labels:
            return {
                "total_pairs": 0,
                "overall_pass_rate": 0.0,
                "failure_rates": {},
            }

        total = len(self.labels)
        pass_count = sum(1 for l in self.labels if l.overall_pass)

        # Calculate failure rates for each category
        failure_rates = {
            "low_skills_overlap": sum(1 for l in self.labels if l.skills_overlap_ratio < 0.5)
            / total,
            "experience_mismatch": sum(l.experience_mismatch for l in self.labels) / total,
            "seniority_mismatch": sum(l.seniority_mismatch for l in self.labels) / total,
            "missing_core_skill": sum(l.missing_core_skill for l in self.labels) / total,
            "hallucinated_skill": sum(l.hallucinated_skill for l in self.labels) / total,
            "awkward_language": sum(l.awkward_language_flag for l in self.labels) / total,
        }

        # Average skills overlap
        avg_overlap = sum(l.skills_overlap_ratio for l in self.labels) / total

        # Stats by prompt template
        template_stats = {}
        for label in self.labels:
            template = label.prompt_template or "unknown"
            if template not in template_stats:
                template_stats[template] = {"count": 0, "failures": 0}
            template_stats[template]["count"] += 1
            if not label.overall_pass:
                template_stats[template]["failures"] += 1

        for template in template_stats:
            template_stats[template]["failure_rate"] = (
                template_stats[template]["failures"] / template_stats[template]["count"]
            )

        # Stats by fit level
        fit_level_stats = {}
        for label in self.labels:
            fit = label.fit_level or "unknown"
            if fit not in fit_level_stats:
                fit_level_stats[fit] = {"count": 0, "avg_overlap": 0.0}
            fit_level_stats[fit]["count"] += 1
            fit_level_stats[fit]["avg_overlap"] += label.skills_overlap_ratio

        for fit in fit_level_stats:
            fit_level_stats[fit]["avg_overlap"] /= fit_level_stats[fit]["count"]

        # Niche role stats
        niche_labels = [l for l in self.labels if l.is_niche_role]
        niche_failure_rate = (
            sum(1 for l in niche_labels if not l.overall_pass) / len(niche_labels)
            if niche_labels
            else 0.0
        )

        return {
            "total_pairs": total,
            "overall_pass_rate": pass_count / total,
            "average_skills_overlap": avg_overlap,
            "failure_rates": failure_rates,
            "by_template": template_stats,
            "by_fit_level": fit_level_stats,
            "niche_role_failure_rate": niche_failure_rate,
        }

    def save_labels(
        self,
        output_dir: str = "data/labeled",
        filename: Optional[str] = None,
    ) -> Path:
        """Save labels to a JSONL file.

        Args:
            output_dir: Output directory path.
            filename: Output filename. If None, generates timestamped name.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = get_timestamped_filename("failure_labels", "jsonl")

        file_path = output_path / filename
        data = [label.to_dict() for label in self.labels]

        save_jsonl(data, file_path)

        logfire.info(f"Saved {len(self.labels)} failure labels to {file_path}")
        return file_path

    def reset(self) -> None:
        """Reset all labels."""
        self.labels = []
