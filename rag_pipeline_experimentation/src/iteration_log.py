"""Structured iteration log for P4 experiment runs.

Each log entry records one configuration change with before/after metrics
and auto-computed deltas. Entries are appended to a JSONL file so the full
history is preserved across runs.

Entry format:
    {
        "iteration":  2,
        "date":       "2026-04-30T10:00:00+00:00",
        "change":     "Switched fixed-512 → recursive-512 chunking",
        "reason":     "Recall@5=0.72 with fixed; mid-paragraph splits visible",
        "config":     {...},
        "before":     {"recall@5": 0.72, "ndcg@5": 0.68},
        "after":      {"recall@5": 0.83, "ndcg@5": 0.77},
        "delta":      {"recall@5": 0.11, "ndcg@5": 0.09}
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_LOG = Path("experiments/iteration_log.jsonl")


def _load_last_after(log_path: Path) -> dict[str, float]:
    """Return the last entry's 'after' metrics, or {} if log is new/empty."""
    if not log_path.exists():
        return {}
    for line in reversed(log_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line).get("after", {})
        except json.JSONDecodeError:
            continue
    return {}


def _compute_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        k: round(after[k] - before[k], 4)
        for k in after
        if k in before
        and isinstance(after.get(k), (int, float))
        and isinstance(before.get(k), (int, float))
    }


def _count_entries(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return sum(1 for line in log_path.read_text().splitlines() if line.strip())


def log_iteration(
    change: str,
    reason: str,
    after_metrics: dict[str, float],
    config: dict | None = None,
    log_path: Path = _DEFAULT_LOG,
) -> None:
    """Append one iteration entry to the JSONL log.

    Args:
        change:        Human-readable description of what changed.
        reason:        The metric or observation that motivated the change.
        after_metrics: Metric values after the change.
        config:        Optional serialised experiment config dict.
        log_path:      Path to the JSONL log file (created if absent).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    before = _load_last_after(log_path)
    delta = _compute_delta(before, after_metrics)
    iteration = _count_entries(log_path) + 1

    entry = {
        "iteration": iteration,
        "date":      datetime.now(timezone.utc).isoformat(),
        "change":    change,
        "reason":    reason,
        "config":    config or {},
        "before":    before,
        "after":     after_metrics,
        "delta":     delta,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def print_log(log_path: Path = _DEFAULT_LOG) -> None:
    """Print all iteration log entries in human-readable format."""
    if not log_path.exists():
        print("No iteration log found.")
        return

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        print(f"\n── Iteration {e.get('iteration', '?')} [{e.get('date', '?')[:10]}] ──")
        print(f"  Change : {e.get('change', '?')}")
        print(f"  Reason : {e.get('reason', '?')}")
        before = e.get("before", {})
        after  = e.get("after",  {})
        delta  = e.get("delta",  {})
        for k in after:
            b = before.get(k, "n/a")
            a = after[k]
            d = delta.get(k, "n/a")
            sign = "+" if isinstance(d, (int, float)) and d >= 0 else ""
            print(f"  {k:22s}: {b!s:8s} → {a!s:8s}  ({sign}{d})")
