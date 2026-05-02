"""Trace ID generation and tracking utilities."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


def generate_trace_id(prefix: str = "trace") -> str:
    """Generate a unique trace ID.

    Args:
        prefix: Optional prefix for the trace ID (e.g., 'resume', 'job', 'pair').

    Returns:
        A unique trace ID string in format: prefix_timestamp_uuid
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


class TraceableMixin(BaseModel):
    """Mixin class that adds trace_id to any Pydantic model."""

    trace_id: Optional[str] = Field(
        default=None,
        description="Unique trace ID for tracking this record through the pipeline",
    )

    def ensure_trace_id(self, prefix: str = "trace") -> str:
        """Ensure the record has a trace_id, generating one if needed.

        Args:
            prefix: Prefix for the generated trace ID.

        Returns:
            The trace_id (existing or newly generated).
        """
        if self.trace_id is None:
            self.trace_id = generate_trace_id(prefix)
        return self.trace_id


def generate_batch_trace_ids(count: int, prefix: str = "trace") -> list[str]:
    """Generate a batch of unique trace IDs.

    Args:
        count: Number of trace IDs to generate.
        prefix: Prefix for all trace IDs.

    Returns:
        List of unique trace IDs.
    """
    return [generate_trace_id(prefix) for _ in range(count)]
