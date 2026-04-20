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
_rate_limit_delay: float | None = None

# Defaults shared by instructor_complete and chat_complete
_DEFAULT_TEMPERATURE: float = 0.7
_DEFAULT_MAX_TOKENS: int = 1500
_DEFAULT_MAX_RETRIES: int = 4
_DEFAULT_BACKOFF_DELAY: float = 2.0   # initial RateLimitError retry delay in seconds
_BACKOFF_MULTIPLIER: float = 2.0      # multiplier applied each retry
_INTER_CYCLE_SLEEP: float = 60.0      # sleep after exhausting all retries before re-raising; gives the API time to recover between outer-loop calls
_INSTRUCTOR_INTERNAL_RETRIES: int = 3  # Instructor-level validation retries


def get_client(settings: Settings | None = None) -> OpenAI:
    """Return a cached OpenAI-compatible client."""
    global _client, _rate_limit_delay
    if _client is None:
        s = settings or get_settings()
        _client = OpenAI(base_url=s.base_url, api_key=s.api_key)
        _rate_limit_delay = s.rate_limit_delay
    return _client


def _get_rate_limit_delay() -> float:
    """Return the cached rate_limit_delay, initialising the client if needed."""
    if _rate_limit_delay is None:
        get_client()
    return _rate_limit_delay  # type: ignore[return-value]


def get_instructor_client(settings: Settings | None = None) -> instructor.Instructor:
    """Return a cached Instructor-patched OpenAI client for structured outputs."""
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_openai(get_client(settings))
    return _instructor_client


def instructor_complete(
    messages: list[dict],
    response_model: Type[T],
    model: str,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> T:
    """Use Instructor to generate a structured Pydantic object from an LLM.

    Sleeps rate_limit_delay seconds after each successful call.
    Retries on 429 RateLimitError with exponential backoff (_DEFAULT_BACKOFF_DELAY * _BACKOFF_MULTIPLIER^n).
    """
    client = get_instructor_client()
    rate_limit_delay = _get_rate_limit_delay()
    delay = _DEFAULT_BACKOFF_DELAY
    for attempt in range(max_retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=_INSTRUCTOR_INTERNAL_RETRIES,
            )
            time.sleep(rate_limit_delay)
            return result
        except InstructorRetryException as e:
            # .errors() is a tenacity method in some versions, a property or absent in others
            _errors = e.errors() if callable(getattr(e, "errors", None)) else getattr(e, "errors", None)
            print(f"\n  [validation failed] {e.n_attempts} attempt(s), model={model}")
            print(f"  errors: {_errors}")
            print(f"  last response: {e.last_completion}")
            e.validation_errors = _errors
            e.validation_attempts = e.n_attempts
            raise
        except RateLimitError:
            if attempt == max_retries:
                print(f"\n  [rate limit] all {max_retries} retries exhausted — sleeping {_INTER_CYCLE_SLEEP:.0f}s before raising", flush=True)
                time.sleep(_INTER_CYCLE_SLEEP)
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...", end="", flush=True)
            time.sleep(delay)
            delay *= _BACKOFF_MULTIPLIER


def chat_complete(
    messages: list[dict],
    model: str,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Sleeps rate_limit_delay seconds after each successful call.
    Retries on 429 RateLimitError with exponential backoff (_DEFAULT_BACKOFF_DELAY * _BACKOFF_MULTIPLIER^n).
    """
    client = get_client()
    rate_limit_delay = _get_rate_limit_delay()
    delay = _DEFAULT_BACKOFF_DELAY
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(rate_limit_delay)
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries:
                print(f"\n  [rate limit] all {max_retries} retries exhausted — sleeping {_INTER_CYCLE_SLEEP:.0f}s before raising", flush=True)
                time.sleep(_INTER_CYCLE_SLEEP)
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...", end="", flush=True)
            time.sleep(delay)
            delay *= _BACKOFF_MULTIPLIER
