"""
Phase 6: Failure & Quality Analysis
Produces all required visualizations, a structured JSON report, and a
self-contained HTML summary page (analysis_summary.html) that embeds all
charts with auto-generated observations.

Auto-loads benchmark_eval.csv (written by Phase 3) from the same output_dir
to compute an apples-to-apples quality gap: both the benchmark and generated
data are compared on the same metric — overall_quality_pass rate — so the
comparison is methodologically valid.
"""

import base64
import json
from datetime import datetime
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


    # ------------------------------------------------------------------
    # HTML summary page: self-contained, base64-embedded charts
    # ------------------------------------------------------------------
    def generate_summary_page(
        self,
        output_dir: Path,
        summary: AnalysisSummary,
        batch_label: str = "",
        ph7_report: dict | None = None,
        iteration_log: list[dict] | None = None,
    ) -> Path:
        """Write analysis_summary.html to output_dir with embedded charts and observations."""

        charts = [
            ("failure_heatmap.png", "Failure Mode Heatmap"),
            ("failure_rates_by_category.png", "Failure Rates by Category"),
            ("failure_mode_trends.png", "Failure Mode Trends: Before vs After Correction"),
            ("quality_dimensions.png", "Quality Dimension Pass Rates"),
            ("benchmark_comparison.png", "Generated vs Benchmark Quality"),
            ("failure_correlations.png", "Failure Mode Correlations"),
        ]

        def _embed(fname: str) -> str:
            p = output_dir / fname
            return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

        def _obs_heatmap() -> list[str]:
            mode_rates = {m: float(self.fdf[m].mean()) for m in FAILURE_MODE_NAMES if m in self.fdf.columns}
            if not mode_rates:
                return ["No failure mode data available."]
            top = max(mode_rates, key=mode_rates.__getitem__)
            low = min(mode_rates, key=mode_rates.__getitem__)
            n_any = int((self.fdf.get("overall_failure", pd.Series(dtype=int)) == 1).sum())
            return [
                f"Most prevalent failure mode: <strong>{_label(top)}</strong> at {mode_rates[top]*100:.1f}%.",
                f"{n_any} of {len(self.fdf)} samples ({n_any/len(self.fdf)*100:.1f}%) have at least one failure.",
                f"Least common failure mode: <strong>{_label(low)}</strong> at {mode_rates[low]*100:.1f}%.",
            ]

        def _obs_by_category() -> list[str]:
            if "category" not in self.fdf.columns or "overall_failure" not in self.fdf.columns:
                return ["No category data available."]
            cat_rates = self.fdf.groupby("category")["overall_failure"].mean()
            worst, best = cat_rates.idxmax(), cat_rates.idxmin()
            obs = [
                f"<strong>{_label(worst)}</strong> has the highest failure rate ({cat_rates[worst]*100:.1f}%).",
                f"<strong>{_label(best)}</strong> has the lowest failure rate ({cat_rates[best]*100:.1f}%).",
            ]
            worst_cat_df = self.fdf[self.fdf["category"] == worst]
            mode_rates = {m: float(worst_cat_df[m].mean()) for m in FAILURE_MODE_NAMES if m in worst_cat_df.columns}
            if mode_rates:
                top = max(mode_rates, key=mode_rates.__getitem__)
                obs.append(f"Dominant failure in {_label(worst)}: <strong>{_label(top)}</strong> ({mode_rates[top]*100:.1f}%).")
            return obs

        def _obs_trends() -> list[str]:
            rate = float(self.fdf["overall_failure"].mean()) if "overall_failure" in self.fdf.columns else 0.0
            obs = [f"Baseline overall failure rate: <strong>{rate*100:.1f}%</strong>."]
            if rate <= 0.15:
                obs.append("Rate is at or below the Phase 7 stopping threshold (15%) — correction may not be needed.")
            elif rate <= 0.30:
                obs.append("Moderate failure rate — Phase 7 correction is recommended.")
            else:
                obs.append("High failure rate — Phase 7 correction is strongly recommended.")
            obs.append("Corrected bars appear only after Phase 7 completes and Phase 6 re-runs.")
            return obs

        def _obs_quality() -> list[str]:
            thresholds = {d.name: d.threshold for d in QUALITY_DIMENSIONS}
            pass_rates = {d.name: float(self.qdf[d.name].mean()) for d in QUALITY_DIMENSIONS if d.name in self.qdf.columns}
            if not pass_rates:
                return ["No quality data available."]
            passed = [n for n in pass_rates if pass_rates[n] >= thresholds.get(n, 0.8)]
            failed = [n for n in pass_rates if pass_rates[n] < thresholds.get(n, 0.8)]
            overall_qp = float(self.qdf["overall_quality_pass"].mean()) if "overall_quality_pass" in self.qdf.columns else 0.0
            obs = [f"Overall quality pass rate: <strong>{overall_qp*100:.1f}%</strong> (Phase 7 target ≥80%)."]
            if passed:
                obs.append(f"{len(passed)} dimension(s) meet threshold: {', '.join(_label(n) for n in passed)}.")
            if failed:
                worst_dim = min(failed, key=lambda n: pass_rates[n])
                obs.append(f"Below threshold: {', '.join('<strong>' + _label(n) + '</strong>' for n in failed)}.")
                obs.append(f"Weakest dimension: <strong>{_label(worst_dim)}</strong> at {pass_rates[worst_dim]*100:.1f}%.")
            return obs

        def _obs_benchmark() -> list[str]:
            if summary.overall_benchmark_gap is None:
                return ["Benchmark data not available — run Phase 3 first to enable gap analysis."]
            gap = summary.overall_benchmark_gap
            obs = [
                (f"Benchmark leads by <strong>{gap*100:.1f}pp</strong> on overall quality pass rate." if gap > 0
                 else f"Generated data leads benchmark by <strong>{abs(gap)*100:.1f}pp</strong>."),
            ]
            if summary.benchmark_dimension_gaps:
                largest = max(summary.benchmark_dimension_gaps, key=lambda k: abs(summary.benchmark_dimension_gaps[k]))
                gv = summary.benchmark_dimension_gaps[largest]
                obs.append(f"Largest gap: <strong>{_label(largest)}</strong> ({gv*100:+.1f}pp, benchmark {'leads' if gv > 0 else 'trails'}).")
            if gap <= 0:
                obs.append("Generated data matches or exceeds benchmark quality.")
            elif gap <= 0.1:
                obs.append("Gap is small (&lt;10pp) — generated quality is close to benchmark.")
            else:
                obs.append("Gap is significant (&gt;10pp) — prompt tuning or Phase 7 correction advised.")
            return obs

        def _obs_correlations() -> list[str]:
            avail = [m for m in FAILURE_MODE_NAMES if m in self.fdf.columns]
            if len(avail) < 2:
                return ["Insufficient data for correlation analysis."]
            corr = self.fdf[avail].corr()
            highest, pair = 0.0, ("", "")
            for i, m1 in enumerate(avail):
                for m2 in avail[i + 1:]:
                    v = corr.loc[m1, m2]
                    if abs(v) > abs(highest):
                        highest, pair = v, (m1, m2)
            obs = []
            if pair[0]:
                direction = "positive" if highest > 0 else "negative"
                obs.append(f"Highest correlation: <strong>{_label(pair[0])}</strong> &amp; <strong>{_label(pair[1])}</strong> ({highest:+.2f}, {direction}).")
                if highest > 0.5:
                    obs.append("Strong positive correlation — these modes likely share a root cause; fixing one may fix the other.")
                elif highest < -0.3:
                    obs.append("Negative correlation — improvements in one mode may coincide with regressions in the other.")
            obs.append("Modes near zero correlation are largely independent and must be addressed separately.")
            return obs

        all_obs = [_obs_heatmap(), _obs_by_category(), _obs_trends(), _obs_quality(), _obs_benchmark(), _obs_correlations()]

        def _metric_card(title: str, value: str, color: str) -> str:
            return (
                f'<div class="metric-card" style="border-top:4px solid {color};">'
                f'<div class="metric-value" style="color:{color};">{value}</div>'
                f'<div class="metric-label">{title}</div></div>'
            )

        f_color = "#e74c3c" if summary.overall_failure_rate > 0.15 else "#2ecc71"
        q_color = "#2ecc71" if summary.overall_quality_pass_rate >= 0.80 else "#e74c3c"
        gap_text = f"{summary.overall_benchmark_gap*100:+.1f}pp" if summary.overall_benchmark_gap is not None else "N/A"
        cards_html = "".join([
            _metric_card("Overall Failure Rate", f"{summary.overall_failure_rate*100:.1f}%", f_color),
            _metric_card("Quality Pass Rate", f"{summary.overall_quality_pass_rate*100:.1f}%", q_color),
            _metric_card("Problematic Samples", str(len(summary.most_problematic_items)), "#e67e22"),
            _metric_card("Benchmark Gap", gap_text, "#3498db"),
        ])

        sections_html = ""
        for (fname, title), obs_list in zip(charts, all_obs):
            data_uri = _embed(fname)
            img_html = (
                f'<img src="data:image/png;base64,{data_uri}" alt="{title}" class="chart-img">'
                if data_uri else '<p class="no-chart">Chart not yet available.</p>'
            )
            obs_html = "".join(f"<li>{o}</li>" for o in obs_list)
            sections_html += (
                f'<div class="chart-section"><h2>{title}</h2>{img_html}'
                f'<div class="observations"><h3>Observations</h3><ul>{obs_html}</ul></div></div>'
            )

        # ── Ph7 before/after section ─────────────────────────────────────────
        ph7_html = ""
        if ph7_report:
            def _arrow_fail(delta: float) -> str:
                return "↓" if delta > 0 else ("→" if delta == 0 else "↑")

            def _arrow_qual(delta: float) -> str:
                return "↑" if delta > 0 else ("→" if delta == 0 else "↓")

            def _color_fail(delta: float) -> str:
                return "#2ecc71" if delta > 0 else ("#e74c3c" if delta < 0 else "#999")

            def _color_qual(delta: float) -> str:
                return "#2ecc71" if delta > 0 else ("#e74c3c" if delta < 0 else "#999")

            target_color = "#2ecc71" if ph7_report.get("target_met") else "#e74c3c"
            target_text = "YES ✓" if ph7_report.get("target_met") else "NO ✗"
            iters = ph7_report.get("iterations_run", "?")
            diversity = ph7_report.get("diversity_score", 1.0)

            bfr = ph7_report.get("baseline_failure_rate", 0) * 100
            cfr = ph7_report.get("corrected_failure_rate", 0) * 100
            bqp = ph7_report.get("baseline_quality_pass_rate", 0) * 100
            cqp = ph7_report.get("corrected_quality_pass_rate", 0) * 100
            imp = ph7_report.get("improvement_pct", 0)

            overview_rows = "".join([
                f"<tr><td>Failure Rate</td><td>{bfr:.1f}%</td><td>{cfr:.1f}%</td>"
                f'<td style="color:{_color_fail(bfr-cfr)}">{_arrow_fail(bfr-cfr)} {abs(bfr-cfr):.1f}pp</td></tr>',
                f"<tr><td>Quality Pass</td><td>{bqp:.1f}%</td><td>{cqp:.1f}%</td>"
                f'<td style="color:{_color_qual(cqp-bqp)}">{_arrow_qual(cqp-bqp)} {abs(cqp-bqp):.1f}pp</td></tr>',
                f"<tr><td>Improvement</td><td>—</td><td>—</td>"
                f'<td style="color:{_color_fail(imp)}">{imp:+.1f}%</td></tr>',
            ])

            mode_rows = "".join(
                f"<tr><td>{_label(m)}</td>"
                f'<td style="color:{_color_fail(d)}">{_arrow_fail(d)} {d*100:+.1f}pp</td></tr>'
                for m, d in ph7_report.get("per_mode_delta", {}).items()
            )

            dim_rows = "".join(
                f"<tr><td>{_label(d)}</td>"
                f'<td style="color:{_color_qual(v)}">{_arrow_qual(v)} {v*100:+.1f}pp</td></tr>'
                for d, v in ph7_report.get("per_dim_quality_delta", {}).items()
            )

            iter_rows = ""
            if iteration_log:
                for entry in iteration_log:
                    m = entry.get("metrics", {})
                    fr = m.get("failure_rate", 0) * 100
                    qp = m.get("quality_pass_rate", 0) * 100
                    ip = m.get("improvement_pct", 0)
                    met = "✓" if m.get("targets_met") else "✗"
                    met_color = "#2ecc71" if m.get("targets_met") else "#e74c3c"
                    iter_rows += (
                        f"<tr><td>{entry.get('iteration','?')}</td>"
                        f"<td>{fr:.1f}%</td><td>{qp:.1f}%</td><td>{ip:+.1f}%</td>"
                        f'<td style="color:{met_color};font-weight:700">{met}</td></tr>'
                    )

            iter_table = (
                f"<h3>Iteration Log</h3>"
                f"<table><tr><th>Iter</th><th>Failure Rate</th><th>Quality Pass</th>"
                f"<th>Improvement</th><th>Targets Met</th></tr>{iter_rows}</table>"
            ) if iter_rows else ""

            ph7_html = f"""
<div class="chart-section ph7-section">
  <h2>Phase 7 — Prompt Correction Results</h2>
  <div class="ph7-meta">
    <span>Iterations run: <strong>{iters}</strong></span>
    &nbsp;&bull;&nbsp;
    <span>Targets met: <strong style="color:{target_color}">{target_text}</strong></span>
    &nbsp;&bull;&nbsp;
    <span>Diversity score: <strong>{diversity:.2f}</strong></span>
  </div>
  <h3>Before / After</h3>
  <table>
    <tr><th>Metric</th><th>Baseline</th><th>Corrected</th><th>Delta</th></tr>
    {overview_rows}
  </table>
  <div class="ph7-tables">
    <div>
      <h3>Per-Mode Failure Delta</h3>
      <table><tr><th>Mode</th><th>Delta</th></tr>{mode_rows}</table>
    </div>
    <div>
      <h3>Per-Dim Quality Delta</h3>
      <table><tr><th>Dimension</th><th>Delta</th></tr>{dim_rows}</table>
    </div>
  </div>
  {iter_table}
</div>"""

        title_label = f"Analysis Summary — {batch_label}" if batch_label else "Analysis Summary"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title_label}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#f5f5f5;color:#333}}
header{{background:#1a1a2e;color:#fff;padding:2rem}}
header h1{{font-size:1.6rem}}
header p{{margin-top:.4rem;font-size:.9rem;opacity:.7}}
.container{{max-width:1100px;margin:0 auto;padding:2rem 1rem}}
.metrics{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}}
.metric-card{{background:#fff;border-radius:8px;padding:1rem 1.5rem;flex:1;min-width:150px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.metric-value{{font-size:2rem;font-weight:700}}
.metric-label{{font-size:.75rem;color:#666;margin-top:.3rem;text-transform:uppercase;letter-spacing:.05em}}
.chart-section{{background:#fff;border-radius:8px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.chart-section h2{{font-size:1.15rem;margin-bottom:1rem;border-bottom:2px solid #eee;padding-bottom:.5rem}}
.chart-img{{width:100%;height:auto;border-radius:4px}}
.no-chart{{color:#999;font-style:italic;padding:2rem;text-align:center}}
.observations{{margin-top:1rem;background:#f9f9f9;border-radius:6px;padding:1rem 1.2rem}}
.observations h3{{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:#666;margin-bottom:.6rem}}
.observations ul{{padding-left:1.2rem}}
.observations li{{font-size:.9rem;line-height:1.6;margin-bottom:.3rem;color:#444}}
.ph7-section{{border-top:4px solid #8e44ad}}
.ph7-meta{{font-size:.9rem;color:#555;margin-bottom:1rem}}
.ph7-section h3{{font-size:.95rem;margin:1rem 0 .5rem;color:#444}}
.ph7-section table{{width:100%;border-collapse:collapse;font-size:.88rem;margin-bottom:.5rem}}
.ph7-section th,.ph7-section td{{padding:.4rem .7rem;border:1px solid #eee;text-align:left}}
.ph7-section th{{background:#f5f5f5;font-weight:600}}
.ph7-tables{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:.5rem}}
footer{{text-align:center;font-size:.75rem;color:#999;padding:2rem}}
</style>
</head>
<body>
<header><h1>{title_label}</h1><p>Generated {timestamp} &bull; {summary.total_samples} samples</p></header>
<div class="container">
<div class="metrics">{cards_html}</div>
{sections_html}
{ph7_html}
</div>
<footer>Synthetic Data DIY — Phase 6 Analysis</footer>
</body>
</html>"""

        out_path = output_dir / "analysis_summary.html"
        out_path.write_text(html)
        print(f"  Saved → {out_path}")
        return out_path


def _is_quality_data_clean(df: pd.DataFrame) -> bool:
    """Return False if more than 40% of rows have all-zero quality scores (rate-limit corruption)."""
    dim_cols = [c for c in QUALITY_DIM_NAMES if c in df.columns]
    if not dim_cols:
        return False
    all_zero = (df[dim_cols] == 0).all(axis=1).mean()
    return all_zero < 0.4


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
        if not _is_quality_data_clean(df):
            print(f"  [strategy comparison] skipping '{label}' — data appears corrupted (>40% all-zero rows)")
            continue
        strategy_rates[label] = [float(df[dim].mean()) if dim in df.columns else 0.0 for dim in QUALITY_DIM_NAMES]

    if not strategy_rates:
        print("  [strategy comparison] no clean data found, skipping chart")
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


def plot_strategy_failure_comparison(
    strategy_dirs: dict[str, Path],
    save_path: Path,
) -> None:
    """Grouped bar chart: overall failure rate + per-mode rates per strategy."""
    strategy_failure: dict[str, list[float]] = {}
    for label, d in strategy_dirs.items():
        csv = d / "failure_labeled_data.csv"
        if not csv.exists():
            print(f"  [failure comparison] skipping '{label}' — failure_labeled_data.csv not found")
            continue
        df = pd.read_csv(csv)
        strategy_failure[label] = [float(df[m].mean()) if m in df.columns else 0.0 for m in FAILURE_MODE_NAMES]

    if not strategy_failure:
        print("  [failure comparison] no data found, skipping chart")
        return

    labels = list(strategy_failure.keys())
    n_strategies = len(labels)
    n_modes = len(FAILURE_MODE_NAMES)
    x = range(n_modes)
    width = 0.8 / n_strategies
    colours = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(labels):
        offsets = [xi + (i - n_strategies / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, strategy_failure[label], width=width, label=label, color=colours[i % len(colours)])

    ax.set_title("Failure Mode Rates by Prompt Strategy", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(m) for m in FAILURE_MODE_NAMES], rotation=30, ha="right")
    ax.set_ylabel("Failure Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def run_multi_batch_comparison(
    base_output_dir: Path,
    labels: set[str] | None = None,
) -> dict:
    """Scan base_output_dir for batch subdirs and produce cross-strategy comparison charts.

    labels: when provided, only include batches whose directory name is in the set.
            Defaults to all non-underscore subdirs (original behaviour).

    Saves to base_output_dir/_comparison/.
    Returns a summary dict with per-batch stats.
    """
    batch_dirs = sorted([
        d for d in base_output_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (labels is None or d.name in labels)
    ])
    if not batch_dirs:
        print("No batch directories found.")
        return {}

    strategy_dirs: dict[str, Path] = {d.name: d for d in batch_dirs}
    print(f"Found {len(strategy_dirs)} batch(es): {', '.join(strategy_dirs)}")

    compare_dir = base_output_dir / "_comparison"
    compare_dir.mkdir(exist_ok=True)

    plot_strategy_comparison(strategy_dirs, compare_dir / "strategy_quality_comparison.png")
    plot_strategy_failure_comparison(strategy_dirs, compare_dir / "strategy_failure_comparison.png")

    summary: dict = {}
    for label, d in strategy_dirs.items():
        entry: dict = {}
        q_csv = d / "quality_eval_data.csv"
        f_csv = d / "failure_labeled_data.csv"
        if q_csv.exists():
            qdf = pd.read_csv(q_csv)
            clean = bool(_is_quality_data_clean(qdf))
            entry["quality_pass_rate"] = round(float(qdf["overall_quality_pass"].mean()), 4) if clean else None
            entry["quality_data_clean"] = clean
            entry["n_quality"] = len(qdf)
        if f_csv.exists():
            fdf = pd.read_csv(f_csv)
            entry["failure_rate"] = round(float(fdf["overall_failure"].mean()), 4)
            entry["n_failure"] = len(fdf)
        summary[label] = entry

    report_path = compare_dir / "comparison_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved → {report_path}")

    print("\nCross-batch summary:")
    print(f"  {'Batch':<30} {'Quality%':>9}  {'Failure%':>9}  {'N':>4}  Clean")
    for label, e in summary.items():
        q = f"{e['quality_pass_rate']*100:.1f}%" if e.get("quality_pass_rate") is not None else "  N/A  "
        f = f"{e['failure_rate']*100:.1f}%" if "failure_rate" in e else "  N/A"
        n = e.get("n_quality", e.get("n_failure", "?"))
        clean = "✓" if e.get("quality_data_clean") else "✗"
        print(f"  {label:<30} {q:>9}  {f:>9}  {n:>4}  {clean}")

    return summary


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

    ph7_report = None
    iteration_log = None
    if corrected_dir:
        ba_path = corrected_dir / "before_after_comparison.json"
        il_path = corrected_dir / "iteration_log.json"
        if ba_path.exists():
            ph7_report = json.loads(ba_path.read_text())
        if il_path.exists():
            iteration_log = json.loads(il_path.read_text())

    analyzer.generate_summary_page(
        output_dir, summary,
        batch_label=output_dir.name,
        ph7_report=ph7_report,
        iteration_log=iteration_log,
    )

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
