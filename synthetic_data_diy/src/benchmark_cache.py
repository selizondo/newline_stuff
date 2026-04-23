"""
Benchmark row cache for dipenbhuva/home-diy-repair-qa.

HuggingFace is contacted at most once per machine. After that, all sampling is
served from two local JSON caches:

  Low-fidelity  (_benchmark_cache/raw_rows.json)
    All rows from the HF training split, with the category column decoded from
    integer back to string. Every HF field is preserved.

  High-fidelity  (_benchmark_cache/validated_rows.json)
    Subset of raw rows that pass QAPair.model_validate(). Only QAPair fields
    plus 'category' are stored (no trace_id — callers assign fresh ones per run).

Public API
----------
sample_raw_rows(num_samples, seed, output_base, oversample_factor=1) -> list[dict]
sample_validated_rows(num_samples, seed, output_base) -> list[dict]

Both return a stratified sample (preserving category proportions). The caches are
stored under output_base/_benchmark_cache/ so every pipeline run shares them.
"""

import json
from collections import defaultdict
from pathlib import Path

BENCHMARK_DATASET = "dipenbhuva/home-diy-repair-qa"
_CACHE_SUBDIR = "_benchmark_cache"


# ---------------------------------------------------------------------------
# Cache directory helpers
# ---------------------------------------------------------------------------

def _cache_dir(output_base: Path) -> Path:
    return output_base / _CACHE_SUBDIR


# ---------------------------------------------------------------------------
# Low-fidelity cache: raw HF rows
# ---------------------------------------------------------------------------

def _load_or_fetch_raw_rows(output_base: Path) -> list[dict]:
    """Return all rows from the HF training split; cached after the first fetch."""
    cache_path = _cache_dir(output_base) / "raw_rows.json"

    if cache_path.exists():
        print(f"Benchmark cache hit (raw rows): {cache_path}")
        return json.loads(cache_path.read_text())

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install 'datasets': pip install datasets"
        )

    print(f"Loading benchmark dataset from HuggingFace: {BENCHMARK_DATASET} ...")
    try:
        dataset = load_dataset(BENCHMARK_DATASET, split="train")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{BENCHMARK_DATASET}'. "
            f"Check network connectivity, or set HF_DATASETS_OFFLINE=1 if cached locally. "
            f"Error: {e}"
        ) from e

    encoded = dataset.class_encode_column("category")
    label_names = encoded.features["category"].names
    rows = [{**row, "category": label_names[row["category"]]} for row in encoded]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    print(f"Cached {len(rows)} raw rows → {cache_path}")
    return rows


# ---------------------------------------------------------------------------
# High-fidelity cache: schema-validated rows (QAPair fields + category)
# ---------------------------------------------------------------------------

def _load_or_build_validated_rows(output_base: Path) -> list[dict]:
    """Return raw rows that pass QAPair.model_validate(); cached after first build."""
    cache_path = _cache_dir(output_base) / "validated_rows.json"

    if cache_path.exists():
        print(f"Benchmark cache hit (validated rows): {cache_path}")
        return json.loads(cache_path.read_text())

    from schema import QAPair

    qa_fields = frozenset(QAPair.model_fields)
    raw_rows = _load_or_fetch_raw_rows(output_base)

    validated: list[dict] = []
    skipped = 0
    for row in raw_rows:
        try:
            qa = QAPair.model_validate({k: row[k] for k in qa_fields if k in row})
            validated.append({"category": row["category"], **qa.model_dump()})
        except Exception:
            skipped += 1

    cache_path.write_text(json.dumps(validated))
    print(f"Cached {len(validated)} validated rows ({skipped} skipped) → {cache_path}")
    return validated


# ---------------------------------------------------------------------------
# Stratified sampling (pure Python, reproducible)
# ---------------------------------------------------------------------------

def _stratified_sample(rows: list[dict], target: int, seed: int) -> list[dict]:
    """Stratified sample from rows preserving category proportions."""
    import random as _rnd

    rng = _rnd.Random(seed)

    if target >= len(rows):
        result = list(rows)
        rng.shuffle(result)
        return result

    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_cat[row.get("category", "")].append(i)

    n = len(rows)
    selected: list[int] = []

    for cat in sorted(by_cat.keys()):
        idxs = list(by_cat[cat])
        rng.shuffle(idxs)
        take = max(1, round(len(idxs) * target / n))
        selected.extend(idxs[:min(take, len(idxs))])

    selected = selected[:target]

    # Top-up if proportional rounding left us short
    if len(selected) < target:
        extras = list(set(range(n)) - set(selected))
        rng.shuffle(extras)
        selected.extend(extras[:target - len(selected)])

    rng.shuffle(selected)
    return [rows[i] for i in selected]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_raw_rows(
    num_samples: int,
    seed: int,
    output_base: Path,
    oversample_factor: int = 1,
) -> list[dict]:
    """Return a stratified sample of raw HF rows.

    Pass oversample_factor > 1 when downstream filtering (e.g. Phase 2 heuristic
    gates) is expected to reject a fraction of rows.
    """
    rows = _load_or_fetch_raw_rows(output_base)
    target = min(oversample_factor * num_samples, len(rows))
    return _stratified_sample(rows, target, seed)


def sample_validated_rows(
    num_samples: int,
    seed: int,
    output_base: Path,
    oversample_factor: int = 1,
) -> list[dict]:
    """Return a stratified sample of schema-valid rows (QAPair fields + category).

    Rows in this cache have already passed QAPair.model_validate(), so callers
    only need to run Phase 2 heuristic gates. Pass oversample_factor if those
    gates are expected to reject a meaningful fraction.
    """
    rows = _load_or_build_validated_rows(output_base)
    target = min(oversample_factor * num_samples, len(rows))
    return _stratified_sample(rows, target, seed)
