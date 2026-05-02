"""Experiment visualization for P4.

Six required charts:
  1. Retrieval metrics comparison heatmap
  2. Configuration dimension impact (grouped bar, avg metric per dim value)
  3. Before/after improvement comparison with delta labels
  4. Generation quality radar chart (4-dim LLM-as-Judge)
  5. Query latency distribution (box plot by retrieval method)
  6. Hybrid fusion weight sweep (NDCG@5 vs alpha)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models import ExperimentResult, JudgeScore

_VIZ_DIR = Path("visualizations")


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 1. Retrieval Metrics Heatmap ───────────────────────────────────────────────

def plot_metrics_heatmap(
    results: list[ExperimentResult],
    metrics: list[str] | None = None,
    out_dir: Path = _VIZ_DIR,
    filename: str = "metrics_heatmap.png",
) -> Path:
    """Heatmap: experiment configs × retrieval metrics."""
    metrics = metrics or ["recall@5", "precision@5", "mrr", "ndcg@5"]

    row_labels = [r.experiment_id for r in results]
    data = np.array([[r.metrics.get(m, 0.0) for m in metrics] for r in results])

    fig_h = max(6, len(results) * 0.45 + 2)
    fig_w = max(8, len(metrics) * 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=metrics,
        yticklabels=row_labels,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Score"},
        linewidths=0.5,
    )
    ax.set_title("Retrieval Metrics Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Configuration", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    return _save(fig, out_dir, filename)


# ── 2. Configuration Dimension Impact ─────────────────────────────────────────

def plot_dimension_impact(
    results: list[ExperimentResult],
    primary_metric: str = "ndcg@5",
    out_dir: Path = _VIZ_DIR,
    filename: str = "dimension_impact.png",
) -> Path:
    """Grouped bar: average metric per value of each config dimension."""
    rows = []
    for r in results:
        cfg = r.config
        rows.append({
            "chunking":  cfg.get("chunk", {}).get("strategy", "?"),
            "embedding": cfg.get("embed", {}).get("model", "?"),
            "retrieval": cfg.get("retrieval", {}).get("method", "?"),
            "metric":    r.metrics.get(primary_metric, 0.0),
        })
    df = pd.DataFrame(rows)

    dimensions = ["chunking", "embedding", "retrieval"]
    colors = ["#4C78A8", "#F58518", "#E45756"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, dim, color in zip(axes, dimensions, colors):
        grp = df.groupby(dim)["metric"].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(grp)), grp.values, color=color, alpha=0.85, width=0.55)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{dim.capitalize()} Impact", fontsize=11, fontweight="bold")
        ax.set_ylabel(primary_metric if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(grp.values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Configuration Dimension Impact on {primary_metric}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return _save(fig, out_dir, filename)


# ── 3. Before / After Improvement ─────────────────────────────────────────────

def plot_before_after(
    before: dict[str, float],
    after: dict[str, float],
    label_before: str = "Before",
    label_after: str = "After",
    out_dir: Path = _VIZ_DIR,
    filename: str = "before_after.png",
) -> Path:
    """Side-by-side bars comparing metrics before and after an improvement."""
    common = [k for k in before if k in after]
    x = np.arange(len(common))
    b_vals = [before[m] for m in common]
    a_vals = [after[m] for m in common]
    deltas = [a - b for a, b in zip(a_vals, b_vals)]

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(common) * 1.5), 5))
    ax.bar(x - width / 2, b_vals, width, label=label_before, color="#4C78A8", alpha=0.85)
    ax.bar(x + width / 2, a_vals, width, label=label_after,  color="#54A24B", alpha=0.85)

    for i, (bv, av, d) in enumerate(zip(b_vals, a_vals, deltas)):
        top = max(bv, av) + 0.03
        sign = "+" if d >= 0 else ""
        ax.text(i, top, f"{sign}{d:.3f}", ha="center", va="bottom", fontsize=9,
                color="#2ca02c" if d >= 0 else "#d62728", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=25, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Before / After Improvement Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, out_dir, filename)


# ── 4. Generation Quality Radar ───────────────────────────────────────────────

def plot_radar(
    scores: list[JudgeScore],
    config_label: str = "Best Config",
    out_dir: Path = _VIZ_DIR,
    filename: str = "radar_generation.png",
) -> Path:
    """Radar chart of average LLM-as-Judge scores across 4 dimensions."""
    dims = ["Relevance", "Accuracy", "Completeness", "Citation Quality"]
    n = max(len(scores), 1)
    avgs = [
        sum(s.relevance        for s in scores) / n,
        sum(s.accuracy         for s in scores) / n,
        sum(s.completeness     for s in scores) / n,
        sum(s.citation_quality for s in scores) / n,
    ]

    # Close the polygon
    angles = [k / len(dims) * 2 * np.pi for k in range(len(dims))]
    avgs_c  = avgs  + [avgs[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_c, avgs_c, "o-", linewidth=2, color="#4C78A8")
    ax.fill(angles_c, avgs_c, alpha=0.2, color="#4C78A8")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="grey")
    ax.set_xticks(angles)
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_title(
        f"Generation Quality — {config_label}",
        fontsize=13, fontweight="bold", pad=20,
    )
    for angle, val in zip(angles, avgs):
        ax.text(angle, val + 0.3, f"{val:.2f}", ha="center", va="center", fontsize=10)
    plt.tight_layout()
    return _save(fig, out_dir, filename)


# ── 5. Latency Distribution ───────────────────────────────────────────────────

def plot_latency(
    results: list[ExperimentResult],
    out_dir: Path = _VIZ_DIR,
    filename: str = "latency_distribution.png",
) -> Path:
    """Box plot of per-query retrieval latency grouped by retrieval method."""
    rows = []
    for r in results:
        method = r.config.get("retrieval", {}).get("method", "?")
        for qr in r.query_results:
            rows.append({"method": method, "latency_ms": qr.retrieval_time_s * 1000})

    fig, ax = plt.subplots(figsize=(10, 5))

    if not rows:
        ax.text(0.5, 0.5, "No per-query latency data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("Query Latency Distribution", fontsize=13, fontweight="bold")
        return _save(fig, out_dir, filename)

    df = pd.DataFrame(rows)
    methods = sorted(df["method"].unique())
    data_groups = [df[df["method"] == m]["latency_ms"].values for m in methods]

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
    bp = ax.boxplot(data_groups, labels=methods, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_ylabel("Query Latency (ms)", fontsize=11)
    ax.set_title("Query Latency Distribution by Retrieval Method",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, out_dir, filename)


# ── 6. Hybrid Fusion Weight Sweep ─────────────────────────────────────────────

def plot_fusion_sweep(
    alpha_scores: dict[float, float],
    metric_name: str = "ndcg@5",
    out_dir: Path = _VIZ_DIR,
    filename: str = "fusion_sweep.png",
) -> Path:
    """Line chart of retrieval metric vs. hybrid fusion weight alpha."""
    alphas = sorted(alpha_scores)
    score_vals = [alpha_scores[a] for a in alphas]

    best_idx = int(np.argmax(score_vals))
    best_alpha = alphas[best_idx]
    best_score = score_vals[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, score_vals, "o-", linewidth=2, markersize=7, color="#4C78A8")
    ax.axvline(
        best_alpha, color="red", linestyle="--", alpha=0.7,
        label=f"Best α={best_alpha:.2f}  ({metric_name}={best_score:.3f})",
    )
    ax.set_xlabel("Hybrid Fusion Weight α  (dense fraction)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"Hybrid Fusion Weight Sweep — {metric_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, out_dir, filename)
