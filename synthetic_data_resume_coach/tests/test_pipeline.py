"""Tests for the pipeline components."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.schemas.resume import (
    ContactInfo,
    Education,
    Experience,
    Skill,
    Resume,
)
from src.schemas.job_description import (
    Company,
    Requirements,
    JobDescription,
)
from src.validators.schema_validator import SchemaValidator, ValidationResult
from src.analysis.failure_modes import FailureModeAnalyzer, FailureCategory


class TestResumeSchema:
    """Tests for Resume schema validation."""

    def test_valid_contact_info(self):
        """Test valid contact info creation."""
        contact = ContactInfo(
            name="John Doe",
            email="john.doe@email.com",
            phone="555-123-4567",
            location="San Francisco, CA",
            linkedin="https://linkedin.com/in/johndoe",
        )
        assert contact.name == "John Doe"
        assert contact.email == "john.doe@email.com"

    def test_invalid_email(self):
        """Test that invalid email raises error."""
        with pytest.raises(ValueError):
            ContactInfo(
                name="John Doe",
                email="invalid-email",
                phone="555-123-4567",
                location="San Francisco, CA",
            )

    def test_invalid_phone(self):
        """Test that short phone number raises error."""
        with pytest.raises(ValueError):
            ContactInfo(
                name="John Doe",
                email="john@email.com",
                phone="123",
                location="San Francisco, CA",
            )

    def test_valid_education(self):
        """Test valid education entry."""
        edu = Education(
            degree="Bachelor of Science in Computer Science",
            institution="MIT",
            graduation_date=date(2020, 5, 15),
            gpa=3.8,
            relevant_coursework=["Data Structures", "Algorithms"],
        )
        assert edu.degree == "Bachelor of Science in Computer Science"
        assert edu.gpa == 3.8

    def test_invalid_gpa(self):
        """Test that GPA outside range raises error."""
        with pytest.raises(ValueError):
            Education(
                degree="BS Computer Science",
                institution="MIT",
                graduation_date=date(2020, 5, 15),
                gpa=5.0,  # Invalid: > 4.0
            )

    def test_valid_experience(self):
        """Test valid experience entry."""
        exp = Experience(
            company="Google",
            title="Software Engineer",
            start_date=date(2020, 6, 1),
            end_date=date(2023, 12, 31),
            responsibilities=["Developed features", "Code reviews"],
            achievements=["Increased efficiency by 20%"],
        )
        assert exp.company == "Google"
        assert exp.end_date > exp.start_date

    def test_experience_end_before_start(self):
        """Test that end date before start date raises error."""
        with pytest.raises(ValueError):
            Experience(
                company="Google",
                title="Software Engineer",
                start_date=date(2023, 1, 1),
                end_date=date(2020, 1, 1),  # Before start
                responsibilities=["Developed features"],
            )

    def test_valid_skill(self):
        """Test valid skill entry."""
        skill = Skill(
            name="Python",
            proficiency_level="Advanced",
            years_experience=5,
        )
        assert skill.name == "Python"
        assert skill.proficiency_level == "Advanced"

    def test_invalid_proficiency_level(self):
        """Test that invalid proficiency level raises error."""
        with pytest.raises(ValueError):
            Skill(
                name="Python",
                proficiency_level="Super Expert",  # Invalid
                years_experience=5,
            )

    def test_valid_resume(self):
        """Test complete valid resume creation."""
        resume = Resume(
            contact=ContactInfo(
                name="John Doe",
                email="john@email.com",
                phone="555-123-4567",
                location="San Francisco, CA",
            ),
            summary="Experienced software engineer",
            education=[
                Education(
                    degree="BS Computer Science",
                    institution="Stanford",
                    graduation_date=date(2018, 6, 15),
                )
            ],
            experience=[
                Experience(
                    company="Google",
                    title="Software Engineer",
                    start_date=date(2018, 7, 1),
                    responsibilities=["Development"],
                )
            ],
            skills=[
                Skill(name="Python", proficiency_level="Expert"),
            ],
        )
        assert resume.contact.name == "John Doe"
        assert len(resume.education) == 1
        assert len(resume.skills) == 1


class TestJobDescriptionSchema:
    """Tests for JobDescription schema validation."""

    def test_valid_company(self):
        """Test valid company creation."""
        company = Company(
            name="TechCorp",
            industry="Technology",
            size="Large",
            location="San Francisco, CA",
        )
        assert company.name == "TechCorp"
        assert company.size == "Large"

    def test_invalid_company_size(self):
        """Test that invalid company size raises error."""
        with pytest.raises(ValueError):
            Company(
                name="TechCorp",
                industry="Technology",
                size="Huge",  # Invalid
                location="San Francisco, CA",
            )

    def test_valid_requirements(self):
        """Test valid requirements creation."""
        reqs = Requirements(
            required_skills=["Python", "SQL"],
            preferred_skills=["AWS"],
            education_requirements="Bachelor's degree",
            experience_years=3,
            experience_level="Mid",
        )
        assert len(reqs.required_skills) == 2
        assert reqs.experience_years == 3

    def test_invalid_experience_level(self):
        """Test that invalid experience level raises error."""
        with pytest.raises(ValueError):
            Requirements(
                required_skills=["Python"],
                education_requirements="Bachelor's",
                experience_years=3,
                experience_level="Super Senior",  # Invalid
            )

    def test_valid_job_description(self):
        """Test complete valid job description creation."""
        job = JobDescription(
            title="Software Engineer",
            company=Company(
                name="TechCorp",
                industry="Technology",
                size="Large",
                location="Remote",
            ),
            description="We are looking for a talented software engineer to join our team. " * 5,
            requirements=Requirements(
                required_skills=["Python", "JavaScript"],
                education_requirements="Bachelor's in CS",
                experience_years=3,
                experience_level="Mid",
            ),
            responsibilities=["Develop features", "Code review"],
            benefits=["Health insurance", "401k"],
            remote_policy="Remote",
            employment_type="Full-time",
        )
        assert job.title == "Software Engineer"
        assert job.remote_policy == "Remote"


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = SchemaValidator()

    def test_validate_valid_resume(self):
        """Test validation of valid resume data."""
        valid_data = {
            "contact": {
                "name": "John Doe",
                "email": "john@email.com",
                "phone": "555-123-4567",
                "location": "San Francisco, CA",
            },
            "education": [
                {
                    "degree": "BS Computer Science",
                    "institution": "Stanford",
                    "graduation_date": "2018-06-15",
                }
            ],
            "skills": [
                {"name": "Python", "proficiency_level": "Expert"},
            ],
        }
        result = self.validator.validate_resume(valid_data)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_resume(self):
        """Test validation of invalid resume data."""
        invalid_data = {
            "contact": {
                "name": "John Doe",
                "email": "invalid-email",  # Invalid
                "phone": "123",  # Too short
                "location": "San Francisco",
            },
            "education": [],  # Missing required education
            "skills": [],  # Missing required skills
        }
        result = self.validator.validate_resume(invalid_data)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_batch_validation(self):
        """Test batch validation."""
        data_list = [
            {
                "contact": {
                    "name": "John",
                    "email": "john@email.com",
                    "phone": "555-123-4567",
                    "location": "SF",
                },
                "education": [
                    {"degree": "BS", "institution": "MIT", "graduation_date": "2020-05-01"}
                ],
                "skills": [{"name": "Python", "proficiency_level": "Advanced"}],
            },
            {
                "contact": {"name": "Jane", "email": "invalid"},  # Invalid
                "education": [],
                "skills": [],
            },
        ]
        results, summary = self.validator.validate_batch(data_list, data_type="resume")
        assert len(results) == 2
        assert summary["valid"] == 1
        assert summary["invalid"] == 1


class TestFailureModeAnalyzer:
    """Tests for FailureModeAnalyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = FailureModeAnalyzer()
        self.validator = SchemaValidator()

    def test_categorize_missing_error(self):
        """Test categorization of missing field error."""
        from src.validators.schema_validator import ValidationError_

        error = ValidationError_(
            field="contact.name",
            error_type="missing",
            message="Field required",
        )
        category = self.analyzer.categorize_error(error)
        assert category == FailureCategory.MISSING_REQUIRED

    def test_categorize_date_error(self):
        """Test categorization of date-related error."""
        from src.validators.schema_validator import ValidationError_

        error = ValidationError_(
            field="experience.0.end_date",
            error_type="value_error",
            message="end_date must be after start_date",
        )
        category = self.analyzer.categorize_error(error)
        assert category == FailureCategory.LOGICAL_INCONSISTENCY

    def test_analyze_results(self):
        """Test analysis of validation results."""
        invalid_data = {
            "contact": {
                "name": "",  # Empty
                "email": "invalid",
                "phone": "123",
                "location": "SF",
            },
            "education": [],
            "skills": [],
        }
        result = self.validator.validate_resume(invalid_data)
        self.analyzer.analyze_results([result])

        stats = self.analyzer.get_statistics()
        assert stats["total_errors"] > 0
        assert stats["unique_modes"] > 0

    def test_statistics_calculation(self):
        """Test statistics calculation."""
        # Empty analyzer
        stats = self.analyzer.get_statistics()
        assert stats["total_errors"] == 0
        assert stats["unique_modes"] == 0


class TestIntegration:
    """Integration tests for the pipeline components."""

    def test_validation_and_analysis_flow(self):
        """Test the flow from validation to failure analysis."""
        validator = SchemaValidator()
        analyzer = FailureModeAnalyzer()

        # Create test data with various errors
        test_data = [
            {
                "contact": {
                    "name": "Valid User",
                    "email": "valid@email.com",
                    "phone": "555-123-4567",
                    "location": "NYC",
                },
                "education": [
                    {"degree": "BS", "institution": "MIT", "graduation_date": "2020-01-01"}
                ],
                "skills": [{"name": "Python", "proficiency_level": "Expert"}],
            },
            {
                "contact": {
                    "name": "",
                    "email": "bad-email",
                    "phone": "123",
                    "location": "LA",
                },
                "education": [],
                "skills": [],
            },
        ]

        # Validate
        results, summary = validator.validate_batch(test_data, data_type="resume")

        # Analyze failures
        analyzer.analyze_results(results)
        stats = analyzer.get_statistics()

        # Verify flow
        assert summary["total"] == 2
        assert summary["valid"] == 1
        assert summary["invalid"] == 1
        assert stats["total_errors"] > 0


class TestDataPersistence:
    """Tests for data saving and loading."""

    def test_validation_result_serialization(self):
        """Test that ValidationResult can be serialized."""
        result = ValidationResult(
            is_valid=False,
            data=None,
            raw_data={"test": "data"},
            errors=[],
        )
        serialized = result.to_dict()
        assert "is_valid" in serialized
        assert serialized["is_valid"] is False

    def test_failure_mode_serialization(self):
        """Test that FailureMode can be serialized."""
        from src.analysis.failure_modes import FailureMode

        mode = FailureMode(
            category=FailureCategory.MISSING_REQUIRED,
            field="contact.name",
            error_type="missing",
            message="Field required",
            count=5,
            examples=["example1"],
        )
        serialized = mode.to_dict()
        assert "category" in serialized
        assert serialized["count"] == 5


class TestFailureLabeler:
    """Tests for FailureLabeler (resume-job pair analysis)."""

    def setup_method(self):
        """Setup test fixtures."""
        from src.analysis.failure_labeler import FailureLabeler

        self.labeler = FailureLabeler()

        # Create test resume
        self.resume = Resume(
            contact=ContactInfo(
                name="John Doe",
                email="john@email.com",
                phone="555-123-4567",
                location="San Francisco, CA",
            ),
            education=[
                Education(
                    degree="BS Computer Science",
                    institution="Stanford University",
                    graduation_date=date(2018, 6, 15),
                )
            ],
            experience=[
                Experience(
                    company="TechCorp",
                    title="Software Engineer",
                    start_date=date(2018, 7, 1),
                    end_date=date(2022, 1, 1),
                    responsibilities=["Developed features", "Code reviews"],
                    achievements=["Improved performance by 50%"],
                )
            ],
            skills=[
                Skill(name="Python", proficiency_level="Expert", years_experience=5),
                Skill(name="JavaScript", proficiency_level="Advanced", years_experience=3),
            ],
        )

        # Create test job
        self.job = JobDescription(
            title="Senior Software Engineer",
            company=Company(
                name="BigTech Inc",
                industry="Technology",
                size="Large",
                location="San Francisco, CA",
            ),
            description="We are looking for a senior software engineer to join our team. " * 5,
            requirements=Requirements(
                required_skills=["Python", "JavaScript", "AWS"],
                preferred_skills=["Docker", "Kubernetes"],
                education_requirements="Bachelor's in CS or related field",
                experience_years=5,
                experience_level="Senior",
            ),
            responsibilities=["Lead development", "Mentor juniors"],
            benefits=["Health insurance", "401k"],
        )

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"python", "javascript", "java"}
        set2 = {"python", "javascript", "go"}
        similarity = self.labeler.jaccard_similarity(set1, set2)
        # Intersection: python, javascript (2)
        # Union: python, javascript, java, go (4)
        assert similarity == 0.5

    def test_jaccard_similarity_empty(self):
        """Test Jaccard similarity with empty sets."""
        assert self.labeler.jaccard_similarity(set(), set()) == 1.0
        assert self.labeler.jaccard_similarity({"a"}, set()) == 0.0

    def test_normalize_skill(self):
        """Test skill normalization."""
        assert self.labeler.normalize_skill("Python 3.10") == "python"
        assert self.labeler.normalize_skill("JavaScript Developer") == "javascript"
        assert self.labeler.normalize_skill("  REACT.js  ") == "react"

    def test_calculate_skills_overlap(self):
        """Test skills overlap calculation."""
        overlap, missing, matched = self.labeler.calculate_skills_overlap(
            self.resume, self.job
        )
        assert 0 <= overlap <= 1
        assert isinstance(missing, list)
        assert isinstance(matched, list)

    def test_detect_experience_mismatch(self):
        """Test experience mismatch detection."""
        is_mismatch, gap = self.labeler.detect_experience_mismatch(
            self.resume, self.job
        )
        assert isinstance(is_mismatch, bool)
        assert isinstance(gap, int)
        assert gap >= 0

    def test_detect_seniority_mismatch(self):
        """Test seniority mismatch detection."""
        is_mismatch, gap_desc = self.labeler.detect_seniority_mismatch(
            self.resume, self.job
        )
        assert isinstance(is_mismatch, bool)
        assert isinstance(gap_desc, str)

    def test_label_pair(self):
        """Test labeling a resume-job pair."""
        labels = self.labeler.label_pair(self.resume, self.job)

        assert labels.trace_id is not None
        assert 0 <= labels.skills_overlap_ratio <= 1
        assert labels.experience_mismatch in [0, 1]
        assert labels.seniority_mismatch in [0, 1]
        assert labels.missing_core_skill in [0, 1]
        assert labels.hallucinated_skill in [0, 1]
        assert labels.awkward_language_flag in [0, 1]

    def test_labels_to_dict(self):
        """Test converting labels to dictionary."""
        labels = self.labeler.label_pair(self.resume, self.job)
        labels_dict = labels.to_dict()

        assert "trace_id" in labels_dict
        assert "skills_overlap_ratio" in labels_dict
        assert "labeled_at" in labels_dict

    def test_labeler_statistics(self):
        """Test labeler statistics."""
        # Label multiple pairs
        self.labeler.label_pair(self.resume, self.job)

        stats = self.labeler.get_statistics()
        assert stats["total_pairs"] == 1
        assert "overall_pass_rate" in stats
        assert "failure_rates" in stats


class TestAPIRoutes:
    """Tests for FastAPI routes."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_review_resume_validation(self, client):
        """Test review-resume endpoint input validation."""
        # Missing required fields should fail
        response = client.post("/review-resume", json={})
        assert response.status_code == 422  # Validation error

    def test_failure_rates_endpoint(self, client):
        """Test failure rates endpoint."""
        response = client.get("/analysis/failure-rates")
        assert response.status_code == 200
        data = response.json()
        assert "example_data" in data


class TestBraintrustEvaluator:
    """Tests for BraintrustEvaluator."""

    def test_evaluator_without_api_key(self):
        """Test evaluator initializes without API key (disabled mode)."""
        from src.evaluation.braintrust_eval import BraintrustEvaluator

        # Clear env var temporarily
        import os
        original = os.environ.pop("BRAINTRUST_API_KEY", None)
        try:
            evaluator = BraintrustEvaluator()
            assert evaluator.enabled is False
        finally:
            if original:
                os.environ["BRAINTRUST_API_KEY"] = original

    def test_log_batch_disabled(self):
        """Test log_batch returns 0 when disabled."""
        from src.evaluation.braintrust_eval import BraintrustEvaluator

        evaluator = BraintrustEvaluator()
        evaluator.enabled = False
        result = evaluator.log_batch([])
        assert result == 0


class TestStorageUtils:
    """Tests for storage utilities."""

    def test_get_timestamped_filename(self):
        """Test timestamped filename generation."""
        from src.utils.storage import get_timestamped_filename

        filename = get_timestamped_filename("test", "jsonl")
        assert filename.startswith("test_")
        assert filename.endswith(".jsonl")

    def test_save_and_load_jsonl(self, tmp_path):
        """Test saving and loading JSONL files."""
        from src.utils.storage import save_jsonl, load_jsonl

        data = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"},
        ]

        file_path = tmp_path / "test.jsonl"
        save_jsonl(data, file_path)

        loaded = load_jsonl(file_path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["name"] == "test2"

    def test_jsonl_writer(self, tmp_path):
        """Test JSONLWriter context manager."""
        from src.utils.storage import JSONLWriter

        file_path = tmp_path / "writer_test.jsonl"

        with JSONLWriter(file_path) as writer:
            writer.write({"a": 1})
            writer.write({"b": 2})

        # Verify file contents
        with open(file_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
