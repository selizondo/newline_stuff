"""Pipeline configuration. LLM settings live in llm_utils.config."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    num_jobs: int = 10
    resumes_per_job: int = 5
    max_correction_retries: int = 3
    model: str = "llama-3.3-70b-versatile"
    output_dir: str = "data"
    generate_heatmaps: bool = True
    enable_correction: bool = True
    enable_llm_judge: bool = False
    enable_braintrust: bool = False
