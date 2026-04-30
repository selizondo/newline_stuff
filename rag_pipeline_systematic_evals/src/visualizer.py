"""
Visualizer for P3 grid-search results.

Produces 9 charts covering all spec-required output:
  1. MRR leaderboard            — all configs ranked by MRR
  2. Recall@K curves            — K in [1,3,5,10] per retrieval method
  3. Chunking comparison        — grouped bar (chunk strategy × metric)
  4. Embedding comparison       — small vs large across key metrics
  5. Retrieval method comparison — vector / BM25 / hybrid
  6. MRR heatmap                — chunk config × embedding model
  7. Recall vs Precision scatter — Recall@5 vs Precision@5, top-5 labelled
  8. Metric correlation matrix  — Pearson r across all IR metrics
  9. Response time vs quality   — avg latency vs MRR, Pareto front annotated

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
        ("mrr_leaderboard",           _chart_mrr_leaderboard),
        ("recall_at_k_curves",        _chart_recall_at_k_curves),
        ("chunking_comparison",       _chart_chunking_comparison),
        ("embedding_comparison",      _chart_embedding_comparison),
        ("retrieval_comparison",      _chart_retrieval_comparison),
        ("mrr_heatmap",               _chart_mrr_heatmap),
        ("recall_precision_scatter",  _chart_recall_precision_scatter),
        ("metric_correlation",        _chart_metric_correlation),
        ("response_time_vs_quality",  _chart_response_time_vs_quality),
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
        columns="embed_label",
        values="mrr",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(7, max(4, len(pivot) * 0.9)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "MRR"},
    )
    ax.set_title("MRR Heatmap: Chunk Config × Embedding Model",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Embedding Model", fontsize=10)
    ax.set_ylabel("Chunk Config", fontsize=10)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 7: Recall@5 vs Precision@5 scatter
# ---------------------------------------------------------------------------

def _chart_recall_precision_scatter(df: pd.DataFrame, path: Path) -> None:
    colors = [_method_color(m) for m in df["retrieval_method"]]
    top5 = df.nlargest(5, "mrr")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["recall@5"], df["precision@5"], c=colors, s=80,
               alpha=0.8, edgecolors="white", linewidths=0.5)

    for _, row in top5.iterrows():
        label = row["experiment_id"].split("__")[-1]
        ax.annotate(label, (row["recall@5"], row["precision@5"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 3))

    ax.set_xlabel("Recall@5", fontsize=11)
    ax.set_ylabel("Precision@5", fontsize=11)
    ax.set_title("Recall@5 vs Precision@5 — All Configurations", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.01, max(df["precision@5"].max() * 1.3, 0.25))

    method_colors = {m: _method_color(m) for m in df["retrieval_method"].unique()}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=m) for m, c in method_colors.items()]
    ax.legend(handles=handles, title="Retrieval", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 8: Metric correlation matrix
# ---------------------------------------------------------------------------

_CORR_COLS = [
    "mrr", "map_score",
    "recall@1", "recall@3", "recall@5", "recall@10",
    "precision@1", "precision@3", "precision@5", "precision@10",
    "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
]
_CORR_LABELS = [
    "MRR", "MAP",
    "R@1", "R@3", "R@5", "R@10",
    "P@1", "P@3", "P@5", "P@10",
    "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10",
]


def _chart_metric_correlation(df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in _CORR_COLS if c in df.columns]
    labels = [_CORR_LABELS[_CORR_COLS.index(c)] for c in cols]
    corr = df[cols].corr()
    corr.index = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"},
        annot_kws={"size": 7},
    )
    ax.set_title("Metric Correlation Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 9: Response time vs quality (Pareto front annotated)
# ---------------------------------------------------------------------------

def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where no other row has lower time AND higher MRR."""
    mask = []
    for i, row in df.iterrows():
        dominated = any(
            (other["avg_time_s"] <= row["avg_time_s"] and other["mrr"] >= row["mrr"])
            and (other["avg_time_s"] < row["avg_time_s"] or other["mrr"] > row["mrr"])
            for j, other in df.iterrows() if i != j
        )
        mask.append(not dominated)
    return df[mask]


def _chart_response_time_vs_quality(df: pd.DataFrame, path: Path) -> None:
    colors = [_method_color(m) for m in df["retrieval_method"]]
    pareto = _pareto_front(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["avg_time_s"] * 1000, df["mrr"], c=colors, s=80,
               alpha=0.8, edgecolors="white", linewidths=0.5)

    for _, row in pareto.iterrows():
        label = row["experiment_id"].split("__")[-1]
        ax.annotate(
            label, (row["avg_time_s"] * 1000, row["mrr"]),
            fontsize=7, textcoords="offset points", xytext=(5, 3),
            color="darkred", fontweight="bold",
        )

    ax.set_xlabel("Avg Retrieval Time (ms)", fontsize=11)
    ax.set_ylabel("MRR", fontsize=11)
    ax.set_title("Response Time vs Quality Trade-off", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)

    method_colors = {m: _method_color(m) for m in df["retrieval_method"].unique()}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=m) for m, c in method_colors.items()]
    ax.legend(handles=handles, title="Retrieval", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _method_color(method: str) -> str:
    return _METHOD_COLORS.get(method, _COLOR_FALLBACK)
