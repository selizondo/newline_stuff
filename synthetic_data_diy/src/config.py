"""
Configuration management — reads from environment variables or .env file.
Supports both OpenAI and local Ollama (OpenAI-compatible API).

Generation and judge models can point at different providers/endpoints, e.g.
OpenAI for generation and a local Ollama instance for low-cost bulk judging.
"""

import functools
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Generation (Phase 1, Phase 6 generation)
    base_url: str
    api_key: str
    generation_model: str
    rate_limit_delay: float

    # LLM-as-Judge (Phases 3, 4, 7 and Phase 6 re-evaluation)
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    judge_rate_limit_delay: float


@functools.lru_cache(maxsize=None)
def get_settings() -> Settings:
    # Generation settings
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    generation_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    rate_limit_delay = float(os.getenv("LLM_RATE_LIMIT_DELAY", "0.5"))

    # Judge settings — default to generation values when not explicitly set
    judge_base_url = os.getenv("LLM_JUDGE_BASE_URL", base_url)
    judge_api_key = os.getenv("LLM_JUDGE_API_KEY", api_key)
    judge_model = os.getenv("LLM_JUDGE_MODEL", generation_model)
    judge_rate_limit_delay = float(os.getenv("LLM_JUDGE_RATE_LIMIT_DELAY", rate_limit_delay))

    if not api_key:
        raise ValueError(
            "LLM_API_KEY is not set. Copy .env.example to .env and fill in your credentials."
        )

    return Settings(
        base_url=base_url,
        api_key=api_key,
        generation_model=generation_model,
        rate_limit_delay=rate_limit_delay,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
        judge_rate_limit_delay=judge_rate_limit_delay,
    )
