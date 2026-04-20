"""
Configuration management — reads from environment variables or .env file.
Supports both OpenAI and local Ollama (OpenAI-compatible API).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    base_url: str
    api_key: str
    generation_model: str  # generation model (Phase 1, Phase 6 generation)
    judge_model: str       # LLM-as-Judge model (Phases 3, 4, 7 and Phase 6 re-evaluation)
    rate_limit_delay: float = 2.0  # seconds between LLM calls


def get_settings() -> Settings:
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # Default judge model to the generation model when not explicitly set
    judge_model = os.getenv("LLM_JUDGE_MODEL", model)
    rate_limit_delay = float(os.getenv("LLM_RATE_LIMIT_DELAY", "2.0"))

    if not api_key:
        raise ValueError(
            "LLM_API_KEY is not set. Copy .env.example to .env and fill in your credentials."
        )

    return Settings(
        base_url=base_url,
        api_key=api_key,
        generation_model=model,
        judge_model=judge_model,
        rate_limit_delay=rate_limit_delay,
    )
