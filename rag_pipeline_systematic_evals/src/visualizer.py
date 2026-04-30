"""
Visualizer for P3 grid-search results.

Produces 6 charts that cover the spec's required output:
  1. MRR leaderboard     — all configs ranked by MRR
  2. Recall@K curves     — K in [1,3,5,10] per retrieval method
  3. Chunking comparison — grouped bar (chunk strategy × metric)
  4. Embedding comparison — small vs large across key metrics
  5. Retrieval method comparison — vector / BM25 / hybrid
  6. MRR heatmap         — chunk config × retrieval method

All charts are saved to `output_dir` as PNG files and returned as
a dict mapping name → Path so the caller can log or display them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EvaluationResult, RetrievalMethod

# Retrieval method colours — used in leaderboard, curves, and comparison chart.
_METHOD_COLORS = {
    RetrievalMethod.VECTOR.value:  "#4C72B0",
    RetrievalMethod.BM25.value:    "#DD8452",
    RetrievalMethod.HYBRID.value:  "#55A868",
}
_COLOR_FALLBACK = "#8c8c8c"

_K_VALUES = [1, 3, 5, 10]
_CORE_METRICS  = ["mrr", "map_score", "recall@5", "ndcg@5"]
_CORE_LABELS   = ["MRR", "MAP", "Recall@5", "NDCG@5"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_charts(
    results: list[EvaluationResult],
    output_dir: Path = Path("visualizations"),
) -> dict[str, Path]:
    """
    Generate all 6 visualisation charts from a list of EvaluationResult objects.

    Returns dict mapping chart name → saved Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _results_to_dataframe(results)

    saved: dict[str, Path] = {}
    for name, fn in [
        ("mrr_leaderboard",      _chart_mrr_leaderboard),
        ("recall_at_k_curves",   _chart_recall_at_k_curves),
        ("chunking_comparison",  _chart_chunking_comparison),
        ("embedding_comparison", _chart_embedding_comparison),
        ("retrieval_comparison", _chart_retrieval_comparison),
        ("mrr_heatmap",          _chart_mrr_heatmap),
    ]:
        path = output_dir / f"{name}.png"
        fn(df, path)
        saved[name] = path

    return saved


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        cfg = r.config
        row = {
            "experiment_id":    r.experiment_id,
            "chunk_label":      cfg.chunk.label(),
            "embed_label":      cfg.embed.label(),
            "retrieval_label":  cfg.retrieval.label(),
            "retrieval_method": cfg.retrieval.method.value,
            "mrr":              r.metrics.mrr,
            "map_score":        r.metrics.map_score,
            "avg_time_s":       r.metrics.avg_retrieval_time_s,
            "total_queries":    r.metrics.total_queries,
        }
        for k in _K_VALUES:
            row[f"recall@{k}"]    = r.metrics.recall_at_k.get(str(k), 0.0)
            row[f"precision@{k}"] = r.metrics.precision_at_k.get(str(k), 0.0)
            row[f"ndcg@{k}"]      = r.metrics.ndcg_at_k.get(str(k), 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart 1: MRR leaderboard
# ---------------------------------------------------------------------------

def _chart_mrr_leaderboard(df: pd.DataFrame, path: Path) -> None:
    sorted_df = df.sort_values("mrr", ascending=True)
    colors = [_method_color(m) for m in sorted_df["retrieval_method"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    bars = ax.barh(sorted_df["experiment_id"], sorted_df["mrr"], color=colors)

    ax.set_xlabel("MRR (Mean Reciprocal Rank)", fontsize=11)
    ax.set_title("MRR Leaderboard — All Configurations", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)

    for bar, val in zip(bars, sorted_df["mrr"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    method_colors = {m: _method_color(m) for m in df["retrieval_method"].unique()}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=m) for m, c in method_colors.items()]
    ax.legend(handles=handles, title="Retrieval", loc="lower right", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Recall@K curves
# ---------------------------------------------------------------------------

def _chart_recall_at_k_curves(df: pd.DataFrame, path: Path) -> None:
    k_cols = [f"recall@{k}" for k in _K_VALUES]
    grouped = df.groupby("retrieval_method")[k_cols].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, row in grouped.iterrows():
        ax.plot(_K_VALUES, row.values, marker="o", label=method,
                color=_method_color(str(method)), linewidth=2)

    ax.set_xlabel("K", fontsize=11)
    ax.set_ylabel("Mean Recall@K", fontsize=11)
    ax.set_title("Recall@K Curves by Retrieval Method", fontsize=13, fontweight="bold")
    ax.set_xticks(_K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Retrieval method", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Chunking comparison (x-axis = chunk strategy, bars = metric)
# ---------------------------------------------------------------------------

def _chart_chunking_comparison(df: pd.DataFrame, path: Path) -> None:
    grouped = df.groupby("chunk_label")[_CORE_METRICS].mean().reset_index()
    x = np.arange(len(grouped))
    n = len(_CORE_METRICS)
    width = 0.8 / n
    palette = sns.color_palette("muted", n)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, label) in enumerate(zip(_CORE_METRICS, _CORE_LABELS)):
        ax.bar(x + i * width - (n - 1) * width / 2, grouped[col], width,
               label=label, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["chunk_label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Chunking Strategy Comparison (averaged over embed & retrieval)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Charts 4 & 5: Grouped-bar by dimension (embed model / retrieval method)
# ---------------------------------------------------------------------------

def _grouped_bar(
    df: pd.DataFrame,
    groupby_col: str,
    title: str,
    path: Path,
    bar_width: float = 0.3,
    color_fn: Callable[[str], str] | None = None,
) -> None:
    """Grouped-bar chart: x = metric, bars = one per group value."""
    grouped = df.groupby(groupby_col)[_CORE_METRICS].mean().reset_index()
    x = np.arange(len(_CORE_METRICS))
    n_groups = len(grouped)
    palette = sns.color_palette("Set2", n_groups) if color_fn is None else None

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(grouped.iterrows()):
        offset = (i - (n_groups - 1) / 2) * bar_width
        color = color_fn(str(row[groupby_col])) if color_fn else palette[i]
        ax.bar(x + offset, [row[m] for m in _CORE_METRICS], bar_width,
               label=str(row[groupby_col]), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(_CORE_LABELS, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_embedding_comparison(df: pd.DataFrame, path: Path) -> None:
    _grouped_bar(
        df, "embed_label",
        "Embedding Model Comparison (averaged over chunk & retrieval)",
        path,
    )


def _chart_retrieval_comparison(df: pd.DataFrame, path: Path) -> None:
    _grouped_bar(
        df, "retrieval_method",
        "Retrieval Method Comparison (averaged over chunk & embed)",
        path,
        bar_width=0.25,
        color_fn=_method_color,
    )


# ---------------------------------------------------------------------------
# Chart 6: MRR heatmap
# ---------------------------------------------------------------------------

def _chart_mrr_heatmap(df: pd.DataFrame, path: Path) -> None:
    pivot = df.pivot_table(
        index="chunk_label",
        columns="retrieval_method",
        values="mrr",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.9)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "MRR"},
    )
    ax.set_title("MRR Heatmap: Chunk Config × Retrieval Method",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Retrieval Method", fontsize=10)
    ax.set_ylabel("Chunk Config", fontsize=10)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _method_color(method: str) -> str:
    return _METHOD_COLORS.get(method, _COLOR_FALLBACK)
