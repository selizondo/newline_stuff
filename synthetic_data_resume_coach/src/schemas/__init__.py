# Re-exports for backward compatibility — all models now live in src/schema.py
from ..schema import (
    Company,
    ContactInfo,
    Education,
    Experience,
    FitLevel,
    JobDescription,
    JobDescriptionMetadata,
    Requirements,
    Resume,
    ResumeJobPair,
    ResumeJobPairMetadata,
    ResumeMetadata,
    Skill,
)

__all__ = [
    "Company",
    "ContactInfo",
    "Education",
    "Experience",
    "FitLevel",
    "JobDescription",
    "JobDescriptionMetadata",
    "Requirements",
    "Resume",
    "ResumeJobPair",
    "ResumeJobPairMetadata",
    "ResumeMetadata",
    "Skill",
]
