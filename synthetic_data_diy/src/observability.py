"""
Observability for the synthetic_data_diy pipeline.

Provides Logfire phase-level tracing and Langfuse per-sample LLM generation
tracking. Both tools degrade gracefully when not configured — the pipeline
never crashes due to observability failures.

Usage:
  from observability import configure_observability, record_llm_generation
  configure_observability()   # call once at startup in main.py
  record_llm_generation(obs_context, ...)  # called from llm_client.py

obs_context dict keys:
  trace_id        - Q&A pair UUID (same value across Phases 1, 4, 5)
  batch_label     - run label used as Langfuse session_id
  phase           - int (1 | 3 | 4 | 5)
  category        - repair category string
  prompt_strategy - strategy name, empty string for judge phases
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_langfuse_client = None
_langfuse_initialized: bool = False

_LOGS_DIR = Path(__file__).parent / "logs"


# ── Langfuse ──────────────────────────────────────────────────────────────────

def flush_langfuse() -> None:
    """Flush any queued Langfuse events. Call once before process exit."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:
        pass


def get_langfuse():
    """Lazy Langfuse singleton. Returns None when keys are missing or placeholder."""
    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client
    _langfuse_initialized = True

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    if (
        not pk or pk.startswith("pk-lf-...")
        or not sk or sk.startswith("sk-lf-...")
    ):
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=pk,
            secret_key=sk,
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
    except Exception:
        pass
    return _langfuse_client


# ── Logfire ───────────────────────────────────────────────────────────────────

def configure_observability(send_to_logfire: bool = False) -> None:
    """Configure Logfire for the pipeline process. Call once at main() startup."""
    try:
        import logfire
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
        from typing import Sequence

        _LOGS_DIR.mkdir(exist_ok=True)
        _trace_file = _LOGS_DIR / "traces.jsonl"

        class _JsonlExporter(SpanExporter):
            def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
                try:
                    with _trace_file.open("a", encoding="utf-8") as f:
                        for span in spans:
                            record = {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "name": span.name,
                                "status": span.status.status_code.name,
                                "duration_ms": round(
                                    (span.end_time - span.start_time) / 1_000_000, 1
                                ) if span.end_time and span.start_time else None,
                                "attributes": dict(span.attributes or {}),
                            }
                            f.write(json.dumps(record) + "\n")
                except Exception:
                    pass
                return SpanExportResult.SUCCESS

            def shutdown(self) -> None:
                pass

        logfire.configure(
            service_name="synthetic-data-pipeline",
            service_version="0.1.0",
            environment=os.getenv("PIPELINE_ENV", "development"),
            send_to_logfire=send_to_logfire,
            console=logfire.ConsoleOptions(
                colors="auto",
                span_style="show-parents",
                include_timestamps=True,
                min_log_level="info",
            ),
            scrubbing=logfire.ScrubbingOptions(
                extra_patterns=["api_key", "secret", "token", "authorization"],
            ),
            additional_span_processors=[SimpleSpanProcessor(_JsonlExporter())],
        )

        lf = get_langfuse()
        logfire.info(
            "observability configured langfuse={langfuse}",
            langfuse=lf is not None,
            langfuse_url=os.getenv("LANGFUSE_BASE_URL", ""),
            send_to_logfire=send_to_logfire,
        )
    except Exception:
        pass


# ── Per-call recording ────────────────────────────────────────────────────────

def record_llm_generation(
    obs_context: dict | None,
    name: str,
    model: str,
    input_messages: list[dict],
    output: Any,
    duration_ms: float,
    error: Exception | None = None,
    extra_attributes: dict | None = None,
) -> None:
    """Emit one Logfire log + one Langfuse generation for a completed LLM call.

    No-ops immediately when obs_context is None — zero overhead for callers that
    don't pass context (e.g. chat_complete used by Phase 3 judge_binary).
    """
    if obs_context is None:
        return

    ctx = obs_context
    phase = ctx.get("phase")
    trace_id = ctx.get("trace_id", "")
    category = ctx.get("category", "")
    batch_label = ctx.get("batch_label", "")
    strategy = ctx.get("prompt_strategy", "")

    # ── Logfire structured log ────────────────────────────────────────────────
    try:
        import logfire
        attrs: dict = {
            "phase": phase,
            "trace_id": trace_id,
            "category": category,
            "batch_label": batch_label,
            "prompt_strategy": strategy,
            "model": model,
            "duration_ms": round(duration_ms, 1),
            "generation_name": name,
            **(extra_attributes or {}),
        }
        if error:
            logfire.error(
                "llm.call.error {generation_name} model={model} {duration_ms}ms",
                **attrs,
            )
        else:
            logfire.info(
                "llm.call {generation_name} model={model} {duration_ms}ms",
                **attrs,
            )
    except Exception:
        pass

    # ── Langfuse generation (v4 ingestion API) ───────────────────────────────
    try:
        lf = get_langfuse()
        if lf is None:
            return

        if trace_id.startswith("benchmark-"):
            return

        import uuid as _uuid
        from datetime import datetime, timezone
        from langfuse.api.ingestion.types.ingestion_event import (
            IngestionEvent_TraceCreate,
            IngestionEvent_GenerationCreate,
        )
        from langfuse.api.ingestion.types.trace_body import TraceBody
        from langfuse.api.ingestion.types.create_generation_body import CreateGenerationBody
        from langfuse.api.commons.types.observation_level import ObservationLevel

        # v4 requires 32-char lowercase hex IDs (UUID without dashes)
        lf_trace_id = trace_id.replace("-", "") if trace_id else _uuid.uuid4().hex
        lf_gen_id = _uuid.uuid4().hex  # unique per generation event
        from datetime import timedelta
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(milliseconds=duration_ms)
        end_iso = end_time.isoformat()

        lf.api.ingestion.batch(batch=[
            IngestionEvent_TraceCreate(
                id=_uuid.uuid4().hex,
                timestamp=end_iso,
                body=TraceBody(
                    id=lf_trace_id,
                    name=name,
                    session_id=batch_label or None,
                    input=input_messages,
                    output=output,
                    metadata={
                        "phase": phase,
                        "category": category,
                        "prompt_strategy": strategy,
                        "batch_label": batch_label,
                    },
                ),
            ),
            IngestionEvent_GenerationCreate(
                id=_uuid.uuid4().hex,
                timestamp=end_iso,
                body=CreateGenerationBody(
                    id=lf_gen_id,
                    trace_id=lf_trace_id,
                    name=name,
                    model=model,
                    input=input_messages,
                    output=output,
                    metadata={
                        "phase": phase,
                        "duration_ms": round(duration_ms, 1),
                        **(extra_attributes or {}),
                    },
                    level=ObservationLevel.ERROR if error else ObservationLevel.DEFAULT,
                    status_message=str(error) if error else None,
                    start_time=start_time,
                    end_time=end_time,
                ),
            ),
        ])
    except Exception:
        pass
