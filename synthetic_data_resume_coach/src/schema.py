"""All Pydantic data models for the Resume Coach pipeline.

Organized in dependency order:
  1. Enums and shared constants
  2. Resume models (ContactInfo → Education → Experience → Skill → ResumeMetadata → Resume)
  3. Job description models (Company → Requirements → JobDescriptionMetadata → JobDescription)
  4. Pair model (ResumeJobPairMetadata → ResumeJobPair)
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── 1. Enums ───────────────────────────────────────────────────────────────────

class FitLevel(str, Enum):
    """Controlled fit levels between a resume and a job description."""

    EXCELLENT = "excellent"   # 80%+ skill overlap
    GOOD = "good"             # 60–80%
    PARTIAL = "partial"       # 40–60%
    POOR = "poor"             # 20–40%
    MISMATCH = "mismatch"     # <20%


# ── 2. Resume models ───────────────────────────────────────────────────────────

class ContactInfo(BaseModel):
    name: str = Field(..., min_length=1, description="Full name of the candidate")
    email: EmailStr = Field(..., description="Email address")
    phone: str = Field(..., min_length=10, description="Phone number")
    location: str = Field(..., description="City, State or City, Country")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio or personal website URL")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        cleaned = "".join(c for c in v if c.isdigit() or c in "+-(). ")
        if len("".join(c for c in cleaned if c.isdigit())) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return v


class Education(BaseModel):
    degree: str = Field(..., description="Degree type and major (e.g., 'B.S. Computer Science')")
    institution: str = Field(..., description="Name of the educational institution")
    graduation_date: date = Field(..., description="Graduation date")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA on 4.0 scale")
    relevant_coursework: Optional[list[str]] = Field(default_factory=list)

    @field_validator("graduation_date", mode="before")
    @classmethod
    def parse_graduation_date(cls, v):
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%B %Y", "%b %Y"]:
                    try:
                        return datetime.strptime(v, fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse date: {v}")
        return v


class Experience(BaseModel):
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    start_date: date = Field(..., description="Start date of employment")
    end_date: Optional[date] = Field(None, description="End date (None if current)")
    responsibilities: list[str] = Field(default_factory=list, min_length=1)
    achievements: list[str] = Field(default_factory=list)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%B %Y", "%b %Y"]:
                    try:
                        return datetime.strptime(v, fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse date: {v}")
        return v

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v, info):
        if v is not None and "start_date" in info.data:
            start = info.data["start_date"]
            if start and v < start:
                raise ValueError("end_date must be after start_date")
        return v


class Skill(BaseModel):
    name: str = Field(..., description="Skill name")
    proficiency_level: str = Field(
        ..., description="Proficiency: Beginner, Intermediate, Advanced, or Expert"
    )
    years_experience: Optional[float] = Field(None, ge=0)

    @field_validator("proficiency_level")
    @classmethod
    def validate_proficiency(cls, v: str) -> str:
        valid_levels = {"beginner", "intermediate", "advanced", "expert"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Proficiency level must be one of: {valid_levels}")
        return v.capitalize()


class ResumeMetadata(BaseModel):
    trace_id: Optional[str] = Field(None, description="Unique trace ID")
    generated_at: Optional[datetime] = Field(None, description="Generation timestamp")
    prompt_template: Optional[str] = Field(None, description="Prompt template used")
    target_industry: Optional[str] = None
    target_seniority: Optional[str] = None
    fit_level: Optional[FitLevel] = Field(None, description="Fit level vs target job")
    target_job_trace_id: Optional[str] = None


class Resume(BaseModel):
    contact: ContactInfo = Field(..., description="Contact information")
    summary: Optional[str] = Field(None, max_length=500, description="Professional summary")
    education: list[Education] = Field(default_factory=list, min_length=1)
    experience: list[Experience] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list, min_length=1)
    certifications: Optional[list[str]] = Field(default_factory=list)
    languages: Optional[list[str]] = Field(default_factory=list)
    metadata: Optional[ResumeMetadata] = Field(default=None)


# ── 3. Job description models ──────────────────────────────────────────────────

class Company(BaseModel):
    name: str = Field(..., min_length=1, description="Company name")
    industry: str = Field(..., description="Industry sector")
    size: str = Field(..., description="Startup, Small, Medium, Large, or Enterprise")
    location: str = Field(..., description="Company location or 'Remote'")

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: str) -> str:
        valid_sizes = {"startup", "small", "medium", "large", "enterprise"}
        if v.lower() not in valid_sizes:
            raise ValueError(f"Company size must be one of: {valid_sizes}")
        return v.capitalize()


class Requirements(BaseModel):
    required_skills: list[str] = Field(default_factory=list, min_length=1)
    preferred_skills: list[str] = Field(default_factory=list)
    education_requirements: str = Field(..., description="Minimum education requirement")
    experience_years: int = Field(..., ge=0, le=30)
    experience_level: str = Field(..., description="Entry, Mid, Senior, Lead, or Executive")

    @field_validator("experience_level")
    @classmethod
    def validate_experience_level(cls, v: str) -> str:
        valid_levels = {"entry", "mid", "senior", "lead", "executive"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Experience level must be one of: {valid_levels}")
        return v.capitalize()


class JobDescriptionMetadata(BaseModel):
    trace_id: Optional[str] = Field(None, description="Unique trace ID")
    generated_at: Optional[datetime] = Field(None, description="Generation timestamp")
    prompt_template: Optional[str] = Field(None, description="Prompt template used")
    is_niche_role: Optional[bool] = Field(None, description="Niche/specialized role flag")


class JobDescription(BaseModel):
    title: str = Field(..., min_length=1, description="Job title")
    company: Company
    description: str = Field(..., min_length=50, description="Detailed job description")
    requirements: Requirements
    responsibilities: list[str] = Field(default_factory=list, min_length=1)
    benefits: list[str] = Field(default_factory=list)
    salary_range: Optional[str] = None
    remote_policy: str = Field("On-site", description="Remote, Hybrid, or On-site")
    employment_type: str = Field("Full-time", description="Full-time, Part-time, Contract, Internship")
    metadata: Optional[JobDescriptionMetadata] = Field(default=None)

    @field_validator("remote_policy")
    @classmethod
    def validate_remote_policy(cls, v: str) -> str:
        valid = {"remote", "hybrid", "on-site", "onsite"}
        if v.lower().replace("-", "") not in {p.replace("-", "") for p in valid}:
            raise ValueError("Remote policy must be one of: Remote, Hybrid, On-site")
        return v.capitalize()

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        valid = {"full-time", "fulltime", "part-time", "parttime", "contract", "internship"}
        if v.lower().replace("-", "") not in {t.replace("-", "") for t in valid}:
            raise ValueError("Employment type must be one of: Full-time, Part-time, Contract, Internship")
        return v.capitalize()


# ── 4. Pair model ──────────────────────────────────────────────────────────────

class ResumeJobPairMetadata(BaseModel):
    trace_id: Optional[str] = Field(None, description="Unique trace ID for this pair")
    generated_at: Optional[datetime] = None
    fit_level: Optional[str] = Field(None, description="excellent, good, partial, poor, mismatch")


class ResumeJobPair(BaseModel):
    """A matched resume + job description with controlled fit level for training data."""

    resume: Resume
    job_description: JobDescription
    match_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    match_analysis: Optional[str] = None
    metadata: Optional[ResumeJobPairMetadata] = Field(default=None)
