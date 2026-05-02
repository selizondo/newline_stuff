"""
Shared LLM client — wraps the OpenAI Python SDK with instructor.

Works with OpenAI, Ollama (http://localhost:11434/v1), Groq
(https://api.groq.com/openai/v1), or any OpenAI-compatible endpoint.

Two independent cached clients:
  generation client  — LLM_BASE_URL / LLM_API_KEY
  judge client       — LLM_JUDGE_BASE_URL / LLM_JUDGE_API_KEY

Rate-limit backoff parses the provider-supplied retry-after header.
Daily quota exhaustion raises RuntimeError immediately rather than sleeping.

Observability hook: pass obs_fn to instructor_complete / judge_binary /
judge_batch. The hook is called with keyword args on both success and error:
  obs_fn(model, input_messages, output, duration_ms, error, extra_attributes)
output is the raw result object on success, None on error.
"""

import re
import time
from typing import Callable, TypeVar, Type

import instructor
from instructor.exceptions import InstructorRetryException
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

# ── Cached clients ─────────────────────────────────────────────────────────────

_gen_client: OpenAI | None = None
_gen_instructor_client: instructor.Instructor | None = None
_gen_rate_limit_delay: float | None = None

_judge_client: OpenAI | None = None
_judge_instructor_client: instructor.Instructor | None = None
_judge_rate_limit_delay: float | None = None

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1500
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_WAIT: float = 60.0      # fallback wait when provider gives no retry-after
TPD_THRESHOLD: float = 300.0          # retry-after above this = daily limit; raise immediately
INSTRUCTOR_INTERNAL_RETRIES: int = 3  # instructor-level validation retries

# ── Judge system prompts ───────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a quality evaluator. "
    "Respond with exactly one digit: 0 or 1."
)

JUDGE_BATCH_SYSTEM = (
    "You are a quality evaluator. "
    "For each criterion, score 1 (pass/present) or 0 (fail/absent). "
    "Return a JSON object with the exact keys specified."
)


def _parse_retry_after(exc: Exception) -> float:
    """Extract the provider-suggested retry-after from a 429 error message.

    Handles two formats:
      - Plain seconds: 'try again in 15.03s'  (Groq TPM, OpenAI)
      - Minutes+seconds: 'try again in 1m21.9s'  (Groq TPD rolling window)

    Returns DEFAULT_RETRY_WAIT if no match.
    Raises immediately if the resolved wait exceeds TPD_THRESHOLD.
    """
    msg = str(exc)
    m = re.search(r"(?:try again in|retry after)\s*(\d+)m([\d.]+)s", msg, re.IGNORECASE)
    if m:
        wait = int(m.group(1)) * 60 + float(m.group(2))
    else:
        m = re.search(r"(?:try again in|retry after)\s*([\d.]+)s", msg, re.IGNORECASE)
        wait = float(m.group(1)) if m else DEFAULT_RETRY_WAIT

    if wait > TPD_THRESHOLD:
        raise RuntimeError(
            f"Daily token quota exhausted — provider requests {wait:.0f}s wait. "
            "Retry after UTC midnight or switch to a different model/key."
        ) from exc
    return wait


def _instructor_mode(base_url: str) -> instructor.Mode:
    """JSON mode for Ollama; TOOLS mode for cloud providers."""
    if "localhost" in base_url or "11434" in base_url:
        return instructor.Mode.JSON
    return instructor.Mode.TOOLS


def get_client(settings: Settings | None = None) -> OpenAI:
    """Return the cached generation OpenAI-compatible client."""
    global _gen_client, _gen_rate_limit_delay
    if _gen_client is None:
        s = settings or get_settings()
        _gen_client = OpenAI(base_url=s.base_url, api_key=s.api_key)
        _gen_rate_limit_delay = s.rate_limit_delay
    return _gen_client


def get_judge_client(settings: Settings | None = None) -> OpenAI:
    """Return the cached judge OpenAI-compatible client."""
    global _judge_client, _judge_rate_limit_delay
    if _judge_client is None:
        s = settings or get_settings()
        _judge_client = OpenAI(base_url=s.judge_base_url, api_key=s.judge_api_key)
        _judge_rate_limit_delay = s.judge_rate_limit_delay
    return _judge_client


def get_instructor_client(settings: Settings | None = None) -> instructor.Instructor:
    """Return the cached instructor-patched generation client."""
    global _gen_instructor_client
    if _gen_instructor_client is None:
        s = settings or get_settings()
        _gen_instructor_client = instructor.from_openai(
            get_client(settings), mode=_instructor_mode(s.base_url)
        )
    return _gen_instructor_client


def get_judge_instructor_client(settings: Settings | None = None) -> instructor.Instructor:
    """Return the cached instructor-patched judge client."""
    global _judge_instructor_client
    if _judge_instructor_client is None:
        s = settings or get_settings()
        _judge_instructor_client = instructor.from_openai(
            get_judge_client(settings), mode=_instructor_mode(s.judge_base_url)
        )
    return _judge_instructor_client


def _gen_delay() -> float:
    if _gen_rate_limit_delay is None:
        get_client()
    return _gen_rate_limit_delay  # type: ignore[return-value]


def _judge_delay() -> float:
    if _judge_rate_limit_delay is None:
        get_judge_client()
    return _judge_rate_limit_delay  # type: ignore[return-value]


def _is_rate_limit(exc: Exception) -> bool:
    """True for both raw RateLimitError and instructor-wrapped 429s."""
    return isinstance(exc, RateLimitError) or "429" in str(exc)


def _call_obs(obs_fn, *, model, input_messages, output, duration_ms, error=None, extra_attributes=None):
    if obs_fn is not None:
        obs_fn(
            model=model,
            input_messages=input_messages,
            output=output,
            duration_ms=duration_ms,
            error=error,
            extra_attributes=extra_attributes or {},
        )


# ── Public call functions ──────────────────────────────────────────────────────

def instructor_complete(
    messages: list[dict],
    response_model: Type[T],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    obs_fn: Callable | None = None,
) -> T:
    """Generate a structured Pydantic object via the generation LLM.

    Retries on 429 using the provider-supplied retry-after time.
    Raises RuntimeError immediately if the provider signals a daily quota exhaustion.
    obs_fn is called with keyword args on success and on non-rate-limit error.
    """
    client = get_instructor_client()
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
            _call_obs(obs_fn, model=model, input_messages=messages, output=result,
                      duration_ms=(time.monotonic() - _t0) * 1000)
            time.sleep(_gen_delay())
            return result
        except InstructorRetryException as exc:
            if not _is_rate_limit(exc):
                _call_obs(obs_fn, model=model, input_messages=messages, output=None,
                          duration_ms=(time.monotonic() - _t0) * 1000, error=exc,
                          extra_attributes={"validation_attempts": getattr(exc, "n_attempts", None)})
                raise
            wait = _parse_retry_after(exc)
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(wait)
        except RateLimitError as exc:
            wait = _parse_retry_after(exc)
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(wait)


def chat_complete(
    messages: list[dict],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_judge_client: bool = False,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Set use_judge_client=True to route through the judge endpoint.
    Retries on 429 using the provider-supplied retry-after time.
    """
    client = get_judge_client() if use_judge_client else get_client()
    rate_delay = _judge_delay() if use_judge_client else _gen_delay()
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
        except RateLimitError as exc:
            wait = _parse_retry_after(exc)
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(wait)


def judge_binary(
    prompt: str,
    model: str,
    default_on_error: int = 0,
    obs_fn: Callable | None = None,
) -> int:
    """Call the judge endpoint and return 0 or 1.

    default_on_error=0 → fail safe (unknown → fail).
    default_on_error=1 → conservative (unknown → assume failure present).
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    _t0 = time.monotonic()
    try:
        raw = chat_complete(messages, model=model, temperature=0.1, max_tokens=10, use_judge_client=True)
        digit = raw.strip()[0] if raw.strip() else ""
        result = int(digit) if digit in ("0", "1") else default_on_error
        _call_obs(obs_fn, model=model, input_messages=messages, output=str(result),
                  duration_ms=(time.monotonic() - _t0) * 1000)
        return result
    except Exception as exc:
        _call_obs(obs_fn, model=model, input_messages=messages, output=None,
                  duration_ms=(time.monotonic() - _t0) * 1000, error=exc)
        return default_on_error


def judge_batch(
    prompt: str,
    response_model: Type[T],
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    obs_fn: Callable | None = None,
) -> T:
    """Call the judge endpoint with instructor to score all criteria in one call.

    Raises InstructorRetryException if the model cannot produce a valid response
    after internal retries — callers should catch and apply a default.
    """
    client = get_judge_instructor_client()
    messages = [
        {"role": "system", "content": JUDGE_BATCH_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    _t0 = time.monotonic()
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
            _call_obs(obs_fn, model=model, input_messages=messages, output=result,
                      duration_ms=(time.monotonic() - _t0) * 1000)
            time.sleep(_judge_delay())
            return result
        except InstructorRetryException as exc:
            if not _is_rate_limit(exc):
                _call_obs(obs_fn, model=model, input_messages=messages, output=None,
                          duration_ms=(time.monotonic() - _t0) * 1000, error=exc)
                raise
            wait = _parse_retry_after(exc)
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(wait)
        except RateLimitError as exc:
            wait = _parse_retry_after(exc)
            if attempt == max_retries:
                raise
            print(f"\n  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", end="", flush=True)
            time.sleep(wait)
