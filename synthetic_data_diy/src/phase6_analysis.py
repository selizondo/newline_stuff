"""
Phase 6: Failure & Quality Analysis
Produces all required visualizations and a structured JSON report.

Auto-loads benchmark_eval.csv (written by Phase 3) from the same output_dir
to compute an apples-to-apples quality gap: both the benchmark and generated
data are compared on the same metric — overall_quality_pass rate — so the
comparison is methodologically valid.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend

from schema import AnalysisSummary, FAILURE_MODE_FIELDS as FAILURE_MODE_NAMES
from phase5_quality_eval import QUALITY_DIMENSIONS

QUALITY_DIM_NAMES = [d.name for d in QUALITY_DIMENSIONS]
REPAIR_CATEGORIES = [
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_maintenance",
    "general_home_repair",
]


def _label(s: str) -> str:
    return s.replace("_", " ").title()


class FailureAnalyzer:
    def __init__(self, failure_df: pd.DataFrame, quality_df: pd.DataFrame):
        self.fdf = failure_df
        self.qdf = quality_df
        # Merge on trace_id for combined analysis
        self.combined = failure_df.merge(quality_df, on=["trace_id", "category"], how="inner")

    # ------------------------------------------------------------------
    # Visualization 1: failure mode heatmap (samples × modes)
    # ------------------------------------------------------------------
    def plot_failure_heatmap(self, save_path: Path) -> None:
        mode_cols = [m for m in FAILURE_MODE_NAMES if m in self.fdf.columns]
        matrix = self.fdf[mode_cols].T  # modes as rows
        fig, ax = plt.subplots(figsize=(max(10, len(self.fdf) * 0.3), 6))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Failure (1) / Pass (0)"},
            linewidths=0.3,
        )
        ax.set_title("Failure Mode Heatmap (Samples × Modes)", fontsize=13)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Failure Mode")
        ax.set_yticklabels([_label(m) for m in mode_cols], rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Visualization 2: failure rates by repair category
    # ------------------------------------------------------------------
    def plot_failure_rates_by_category(self, save_path: Path) -> None:
        cat_rates = (
            self.fdf.groupby("category")[FAILURE_MODE_NAMES]
            .mean()
            .reindex([c for c in REPAIR_CATEGORIES if c in self.fdf["category"].unique()])
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        cat_rates.plot(kind="bar", ax=ax, colormap="tab10")
        ax.set_title("Failure Rates by Repair Category", fontsize=13)
        ax.set_xlabel("Category")
        ax.set_ylabel("Failure Rate")
        ax.set_xticklabels([_label(c) for c in cat_rates.index], rotation=30, ha="right")
        ax.legend(
            [_label(m) for m in FAILURE_MODE_NAMES],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Visualization 3: before vs after trend (populated by Phase 7)
    # ------------------------------------------------------------------
    def plot_failure_mode_trends(
        self,
        save_path: Path,
        corrected_df: pd.DataFrame | None = None,
    ) -> None:
        baseline_rates = self.fdf[FAILURE_MODE_NAMES].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(FAILURE_MODE_NAMES))
        labels = [_label(m) for m in FAILURE_MODE_NAMES]

        ax.bar([i - 0.2 for i in x], baseline_rates, width=0.4, label="Baseline", color="#e74c3c")
        if corrected_df is not None and not corrected_df.empty:
            corrected_rates = corrected_df[FAILURE_MODE_NAMES].mean()
            ax.bar([i + 0.2 for i in x], corrected_rates, width=0.4, label="Corrected", color="#2ecc71")

        ax.set_title("Failure Mode Rates: Before vs After Correction", fontsize=13)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Failure Rate")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Visualization 4: quality dimension pass rates
    # ------------------------------------------------------------------
    def plot_quality_dimensions(self, save_path: Path) -> None:
        pass_rates = self.qdf[QUALITY_DIM_NAMES].mean()
        thresholds = {d.name: d.threshold for d in QUALITY_DIMENSIONS}

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            range(len(QUALITY_DIM_NAMES)),
            pass_rates,
            color=["#2ecc71" if pass_rates[n] >= thresholds[n] else "#e74c3c" for n in QUALITY_DIM_NAMES],
        )
        for i, name in enumerate(QUALITY_DIM_NAMES):
            ax.hlines(thresholds[name], i - 0.4, i + 0.4, colors="navy", linestyles="--", linewidth=1.2)

        ax.set_title("Quality Dimension Pass Rates (green = meets threshold)", fontsize=13)
        ax.set_xticks(range(len(QUALITY_DIM_NAMES)))
        ax.set_xticklabels([_label(n) for n in QUALITY_DIM_NAMES], rotation=30, ha="right")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Visualization 5: benchmark vs generated overall_quality_pass
    # Apples-to-apples: both use the same overall_quality_pass metric.
    # ------------------------------------------------------------------
    def plot_benchmark_comparison(
        self,
        save_path: Path,
        benchmark_df: pd.DataFrame | None = None,
    ) -> None:
        generated_rates = self.qdf[QUALITY_DIM_NAMES].mean()
        labels = [_label(n) for n in QUALITY_DIM_NAMES]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(QUALITY_DIM_NAMES))
        ax.bar([i - 0.2 for i in x], generated_rates, width=0.4, label="Generated", color="#3498db")
        if benchmark_df is not None and not benchmark_df.empty:
            bench_vals = [
                float(benchmark_df[n].mean()) if n in benchmark_df.columns else 0.0
                for n in QUALITY_DIM_NAMES
            ]
            ax.bar([i + 0.2 for i in x], bench_vals, width=0.4, label="Benchmark", color="#f39c12")

        ax.set_title("Generated vs Benchmark Quality Dimensions", fontsize=13)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Visualization 6: failure mode correlation heatmap
    # ------------------------------------------------------------------
    def plot_failure_correlations(self, save_path: Path) -> None:
        corr = self.fdf[FAILURE_MODE_NAMES].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
        )
        ax.set_title("Failure Mode Correlations (Pearson)", fontsize=13)
        ax.set_xticklabels([_label(m) for m in FAILURE_MODE_NAMES], rotation=30, ha="right")
        ax.set_yticklabels([_label(m) for m in FAILURE_MODE_NAMES], rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved → {save_path}")

    # ------------------------------------------------------------------
    # Identify most problematic items (3+ failures)
    # ------------------------------------------------------------------
    def most_problematic_items(self) -> list[str]:
        return list(self.fdf[self.fdf["failure_count"] >= 3]["trace_id"].values)

    # ------------------------------------------------------------------
    # Build AnalysisSummary — optionally includes benchmark gap
    # ------------------------------------------------------------------
    def build_summary(self, benchmark_df: pd.DataFrame | None = None) -> AnalysisSummary:
        failure_rates_by_mode = {m: float(self.fdf[m].mean()) for m in FAILURE_MODE_NAMES if m in self.fdf.columns}
        failure_rates_by_cat = {
            cat: float(self.fdf[self.fdf["category"] == cat]["overall_failure"].mean())
            for cat in self.fdf["category"].unique()
        }
        quality_rates = {d.name: float(self.qdf[d.name].mean()) for d in QUALITY_DIMENSIONS if d.name in self.qdf.columns}
        thresholds_met = {d.name: quality_rates.get(d.name, 0) >= d.threshold for d in QUALITY_DIMENSIONS}
        generated_pass_rate = float(self.qdf["overall_quality_pass"].mean())

        # Benchmark gap — apples-to-apples: compare overall_quality_pass rates
        overall_benchmark_gap = None
        benchmark_dimension_gaps = None
        if benchmark_df is not None and not benchmark_df.empty and "overall_quality_pass" in benchmark_df.columns:
            bench_pass_rate = float(benchmark_df["overall_quality_pass"].mean())
            overall_benchmark_gap = round(bench_pass_rate - generated_pass_rate, 4)
            benchmark_dimension_gaps = {
                d.name: round(float(benchmark_df[d.name].mean()) - quality_rates.get(d.name, 0.0), 4)
                for d in QUALITY_DIMENSIONS
                if d.name in benchmark_df.columns
            }

        return AnalysisSummary(
            total_samples=len(self.fdf),
            overall_failure_rate=float(self.fdf["overall_failure"].mean()),
            failure_rates_by_mode=failure_rates_by_mode,
            failure_rates_by_category=failure_rates_by_cat,
            quality_pass_rates_by_dimension=quality_rates,
            overall_quality_pass_rate=generated_pass_rate,
            thresholds_met=thresholds_met,
            most_problematic_items=self.most_problematic_items(),
            overall_benchmark_gap=overall_benchmark_gap,
            benchmark_dimension_gaps=benchmark_dimension_gaps,
        )


def plot_strategy_comparison(
    strategy_dirs: dict[str, Path],
    save_path: Path,
) -> None:
    """Visualization 7: quality pass rate per strategy, per dimension."""
    strategy_rates: dict[str, list[float]] = {}
    for label, d in strategy_dirs.items():
        csv = d / "quality_eval_data.csv"
        if not csv.exists():
            print(f"  [strategy comparison] skipping '{label}' — quality_eval_data.csv not found")
            continue
        df = pd.read_csv(csv)
        strategy_rates[label] = [float(df[dim].mean()) if dim in df.columns else 0.0 for dim in QUALITY_DIM_NAMES]

    if not strategy_rates:
        print("  [strategy comparison] no data found, skipping chart")
        return

    labels = list(strategy_rates.keys())
    n_strategies = len(labels)
    n_dims = len(QUALITY_DIM_NAMES)
    x = range(n_dims)
    width = 0.8 / n_strategies
    colours = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(labels):
        offsets = [xi + (i - n_strategies / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, strategy_rates[label], width=width, label=label, color=colours[i % len(colours)])

    ax.set_title("Quality Pass Rate by Prompt Strategy", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(n) for n in QUALITY_DIM_NAMES], rotation=30, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def run_analysis_phase(
    output_dir: Path,
    corrected_dir: Path | None = None,
) -> AnalysisSummary:
    """Run analysis and produce visualizations + JSON report.

    Auto-loads benchmark_eval.csv from output_dir (written by Phase 3) when
    present. Both the benchmark and generated data are scored on overall_quality_pass
    so the gap comparison is apples-to-apples.
    """
    failure_csv = output_dir / "failure_labeled_data.csv"
    quality_csv = output_dir / "quality_eval_data.csv"

    if not failure_csv.exists():
        raise FileNotFoundError(f"Not found: {failure_csv}. Run Phase 4 first.")
    if not quality_csv.exists():
        raise FileNotFoundError(f"Not found: {quality_csv}. Run Phase 5 first.")

    fdf = pd.read_csv(failure_csv)
    qdf = pd.read_csv(quality_csv)

    corrected_fdf = None
    if corrected_dir and (corrected_dir / "failure_labeled_data.csv").exists():
        corrected_fdf = pd.read_csv(corrected_dir / "failure_labeled_data.csv")

    benchmark_df = None
    benchmark_csv = output_dir / "benchmark_eval.csv"
    if benchmark_csv.exists():
        benchmark_df = pd.read_csv(benchmark_csv)
        print(f"  Loaded benchmark data: {len(benchmark_df)} items from {benchmark_csv}")
    else:
        print("  No benchmark_eval.csv found — gap analysis skipped (run Phase 3 first).")

    analyzer = FailureAnalyzer(fdf, qdf)

    print("Generating visualizations...")
    analyzer.plot_failure_heatmap(output_dir / "failure_heatmap.png")
    analyzer.plot_failure_rates_by_category(output_dir / "failure_rates_by_category.png")
    analyzer.plot_failure_mode_trends(output_dir / "failure_mode_trends.png", corrected_df=corrected_fdf)
    analyzer.plot_quality_dimensions(output_dir / "quality_dimensions.png")
    analyzer.plot_benchmark_comparison(output_dir / "benchmark_comparison.png", benchmark_df=benchmark_df)
    analyzer.plot_failure_correlations(output_dir / "failure_correlations.png")

    summary = analyzer.build_summary(benchmark_df=benchmark_df)
    report_path = output_dir / "analysis_report.json"
    report_path.write_text(json.dumps(summary.model_dump(), indent=2))
    print(f"  Saved → {report_path}")

    print(f"\nAnalysis summary:")
    print(f"  Overall failure rate : {summary.overall_failure_rate*100:.1f}%")
    print(f"  Overall quality pass : {summary.overall_quality_pass_rate*100:.1f}%")
    print(f"  Problematic items    : {len(summary.most_problematic_items)} (≥3 failures)")
    dims_not_met = [k for k, v in summary.thresholds_met.items() if not v]
    if dims_not_met:
        print(f"  Dimensions below threshold: {', '.join(dims_not_met)}")
    if summary.overall_benchmark_gap is not None:
        gap = summary.overall_benchmark_gap
        direction = "benchmark leads" if gap > 0 else "generated leads"
        print(f"  Benchmark gap (overall_quality_pass): {gap*100:+.1f}pp ({direction})")
    return summary
