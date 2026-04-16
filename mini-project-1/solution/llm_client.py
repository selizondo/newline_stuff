"""
Configurable LLM client — wraps the OpenAI Python SDK.
Works with OpenAI, Ollama (http://localhost:11434/v1), or any OpenAI-compatible endpoint.
"""

import time
from typing import TypeVar, Type

import instructor
from instructor.exceptions import InstructorRetryException
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None
_instructor_client: instructor.Instructor | None = None


def get_client(settings: Settings | None = None) -> OpenAI:
    """Return a cached OpenAI-compatible client."""
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = OpenAI(base_url=s.base_url, api_key=s.api_key)
    return _client


def get_instructor_client(settings: Settings | None = None) -> instructor.Instructor:
    """Return a cached Instructor-patched OpenAI client for structured outputs."""
    global _instructor_client
    if _instructor_client is None:
        s = settings or get_settings()
        _instructor_client = instructor.from_openai(OpenAI(base_url=s.base_url, api_key=s.api_key))
    return _instructor_client


def instructor_complete(
    messages: list[dict],
    response_model: Type[T],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 4,
) -> T:
    """Use Instructor to generate a structured Pydantic object from an LLM.

    Retries on 429 RateLimitError with exponential backoff (2s, 4s, 8s, 16s).
    """
    client = get_instructor_client()
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=3,
            )
        except InstructorRetryException as e:
            print(f"\n  [validation failed] {e.n_attempts} attempt(s), model={model}")
            print(f"  errors: {e.errors()}")
            print(f"  last response: {e.last_completion}")
            e.validation_errors = e.errors()
            e.validation_attempts = e.n_attempts
            raise
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...", end="", flush=True)
            time.sleep(delay)
            delay *= 2


def chat_complete(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 4,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Retries on 429 RateLimitError with exponential backoff (2s, 4s, 8s, 16s).
    """
    client = get_client()
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...", end="", flush=True)
            time.sleep(delay)
            delay *= 2
