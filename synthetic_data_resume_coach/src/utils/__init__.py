"""Utility modules for the Synthetic Data Resume Coach."""

from .trace import generate_trace_id, TraceableMixin, generate_batch_trace_ids
from .storage import (
    save_jsonl,
    load_jsonl,
    iter_jsonl,
    load_jsonl_as_models,
    save_invalid_records,
    get_timestamped_filename,
    JSONLWriter,
)

__all__ = [
    # Trace utilities
    "generate_trace_id",
    "TraceableMixin",
    "generate_batch_trace_ids",
    # Storage utilities
    "save_jsonl",
    "load_jsonl",
    "iter_jsonl",
    "load_jsonl_as_models",
    "save_invalid_records",
    "get_timestamped_filename",
    "JSONLWriter",
]
