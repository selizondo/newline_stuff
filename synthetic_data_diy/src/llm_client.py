"""
LLM client for the synthetic_data_diy pipeline.

Thin wrapper around llm_utils.client that adds per-call Logfire + Langfuse
observability via record_llm_generation. Client caching, rate-limit backoff,
and retry logic all live in llm_utils.
"""

import time
from typing import TypeVar, Type

import instructor
from instructor.exceptions import InstructorRetryException
from openai import RateLimitError
from pydantic import BaseModel

from llm_utils.client import (
    get_client,
    get_instructor_client,
    get_judge_client,
    get_judge_instructor_client,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BACKOFF_DELAY,
    BACKOFF_MULTIPLIER,
    INTER_CYCLE_SLEEP,
    INSTRUCTOR_INTERNAL_RETRIES,
    _gen_delay,
    _judge_delay,
    _is_rate_limit,
)
from llm_utils.config import get_settings
from observability import record_llm_generation

T = TypeVar("T", bound=BaseModel)

_JUDGE_SYSTEM_PROMPT = (
    "You are a quality evaluator for DIY repair content. "
    "Respond with exactly one digit: 0 or 1."
)

_JUDGE_BATCH_SYSTEM_PROMPT = (
    "You are a quality evaluator for DIY repair content. "
    "For each criterion, score 1 (pass/present) or 0 (fail/absent). "
    "Return a JSON object with the exact keys specified."
)


def instructor_complete(
    messages: list[dict],
    response_model: Type[T],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    obs_context: dict | None = None,
) -> T:
    """Structured generation with observability. Retries on 429."""
    client = get_instructor_client()
    delay = DEFAULT_BACKOFF_DELAY
    _t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=INSTRUCTOR_INTERNAL_RETRIES,
            )
            record_llm_generation(
                obs_context=obs_context,
                name="phase1.instructor_complete",
                model=model,
                input_messages=messages,
                output=result.model_dump() if hasattr(result, "model_dump") else str(result),
                duration_ms=(time.monotonic() - _t0) * 1000,
            )
            time.sleep(_gen_delay())
            return result
        except InstructorRetryException as exc:
            if _is_rate_limit(exc):
                if attempt == max_retries:
                    print(f"\n  [rate limit] retries exhausted — sleeping {INTER_CYCLE_SLEEP:.0f}s", flush=True)
                    time.sleep(INTER_CYCLE_SLEEP)
                    raise
                print(f"\n  [rate limit] waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
                time.sleep(delay)
                delay *= BACKOFF_MULTIPLIER
                continue
            _errors = exc.errors() if callable(getattr(exc, "errors", None)) else getattr(exc, "errors", None)
            print(f"\n  [validation failed] {exc.n_attempts} attempt(s), model={model}")
            print(f"  errors: {_errors}")
            record_llm_generation(
                obs_context=obs_context,
                name="phase1.instructor_complete",
                model=model,
                input_messages=messages,
                output=None,
                duration_ms=(time.monotonic() - _t0) * 1000,
                error=exc,
                extra_attributes={"validation_attempts": exc.n_attempts},
            )
            raise
        except RateLimitError:
            if attempt == max_retries:
                print(f"\n  [rate limit] retries exhausted — sleeping {INTER_CYCLE_SLEEP:.0f}s", flush=True)
                time.sleep(INTER_CYCLE_SLEEP)
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER


def chat_complete(
    messages: list[dict],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_judge_client: bool = False,
) -> str:
    """Chat completion with rate-limit backoff. Routes to judge endpoint when flagged."""
    client = get_judge_client() if use_judge_client else get_client()
    rate_delay = _judge_delay() if use_judge_client else _gen_delay()
    delay = DEFAULT_BACKOFF_DELAY
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(rate_delay)
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries:
                print(f"\n  [rate limit] retries exhausted — sleeping {INTER_CYCLE_SLEEP:.0f}s", flush=True)
                time.sleep(INTER_CYCLE_SLEEP)
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER


def judge_binary(
    prompt: str,
    model: str,
    default_on_error: int = 0,
    obs_context: dict | None = None,
    name: str = "judge_binary",
) -> int:
    """Call judge endpoint and return 0 or 1, with observability."""
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    _t0 = time.monotonic()
    try:
        raw = chat_complete(messages, model=model, temperature=0.1, max_tokens=10, use_judge_client=True)
        digit = raw.strip()[0] if raw.strip() else ""
        result = int(digit) if digit in ("0", "1") else default_on_error
        record_llm_generation(
            obs_context=obs_context,
            name=name,
            model=model,
            input_messages=messages,
            output=str(result),
            duration_ms=(time.monotonic() - _t0) * 1000,
        )
        return result
    except Exception as exc:
        record_llm_generation(
            obs_context=obs_context,
            name=name,
            model=model,
            input_messages=messages,
            output=None,
            duration_ms=(time.monotonic() - _t0) * 1000,
            error=exc,
        )
        return default_on_error


def judge_batch(
    prompt: str,
    response_model: Type[T],
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    obs_context: dict | None = None,
) -> T:
    """Batch judge call with instructor, observability, and rate-limit backoff."""
    client = get_judge_instructor_client()
    messages = [
        {"role": "system", "content": _JUDGE_BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    delay = DEFAULT_BACKOFF_DELAY
    _t0 = time.monotonic()
    _gen_name = f"phase{obs_context.get('phase', '?')}.judge_batch" if obs_context else "judge_batch"
    for attempt in range(max_retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=0.0,
                max_tokens=500,
                max_retries=INSTRUCTOR_INTERNAL_RETRIES,
            )
            record_llm_generation(
                obs_context=obs_context,
                name=_gen_name,
                model=model,
                input_messages=messages,
                output=result.model_dump() if hasattr(result, "model_dump") else str(result),
                duration_ms=(time.monotonic() - _t0) * 1000,
            )
            time.sleep(_judge_delay())
            return result
        except InstructorRetryException as exc:
            if _is_rate_limit(exc):
                if attempt == max_retries:
                    print(f"\n  [rate limit] retries exhausted — sleeping {INTER_CYCLE_SLEEP:.0f}s", flush=True)
                    time.sleep(INTER_CYCLE_SLEEP)
                    raise
                print(f"\n  [rate limit] waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
                time.sleep(delay)
                delay *= BACKOFF_MULTIPLIER
                continue
            record_llm_generation(
                obs_context=obs_context,
                name=_gen_name,
                model=model,
                input_messages=messages,
                output=None,
                duration_ms=(time.monotonic() - _t0) * 1000,
                error=exc,
            )
            raise
        except RateLimitError:
            if attempt == max_retries:
                print(f"\n  [rate limit] retries exhausted — sleeping {INTER_CYCLE_SLEEP:.0f}s", flush=True)
                time.sleep(INTER_CYCLE_SLEEP)
                raise
            print(f"\n  [rate limit] waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER
