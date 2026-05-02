"""Heatmap visualization for validation analysis."""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logfire

from ..validators.schema_validator import ValidationResult
from .failure_modes import FailureModeAnalyzer, FailureCategory

if TYPE_CHECKING:
    from .failure_labeler import FailureLabeler


class HeatmapGenerator:
    """Generate heatmap visualizations for validation analysis."""

    # Define fields to track for resumes and jobs
    RESUME_FIELDS = [
        "contact.name",
        "contact.email",
        "contact.phone",
        "contact.location",
        "summary",
        "education",
        "experience",
        "skills",
        "certifications",
    ]

    JOB_FIELDS = [
        "title",
        "company.name",
        "company.industry",
        "company.size",
        "description",
        "requirements.required_skills",
        "requirements.experience_years",
        "responsibilities",
        "benefits",
    ]

    def __init__(self, output_dir: str = "data/validated"):
        """Initialize the heatmap generator.

        Args:
            output_dir: Directory to save generated visualizations.
        """
        logfire.configure()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

    def create_field_validation_heatmap(
        self,
        results: list[ValidationResult],
        data_type: str = "resume",
        filename: str = "field_validation_heatmap.png",
    ) -> Path:
        """Create a heatmap showing field-level validation success rates.

        Args:
            results: List of ValidationResult objects.
            data_type: Type of data ("resume" or "job").
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        fields = self.RESUME_FIELDS if data_type == "resume" else self.JOB_FIELDS

        # Count successes and failures per field
        field_stats = {field: {"success": 0, "failure": 0} for field in fields}

        for result in results:
            if result.is_valid:
                for field in fields:
                    field_stats[field]["success"] += 1
            else:
                failed_fields = {e.field for e in result.errors}
                for field in fields:
                    # Check if this field or any child field failed
                    if any(ff.startswith(field) for ff in failed_fields):
                        field_stats[field]["failure"] += 1
                    else:
                        field_stats[field]["success"] += 1

        # Create DataFrame
        df_data = []
        for field, stats in field_stats.items():
            total = stats["success"] + stats["failure"]
            success_rate = stats["success"] / total if total > 0 else 0
            df_data.append({
                "field": field,
                "success_rate": success_rate,
                "failures": stats["failure"],
            })

        df = pd.DataFrame(df_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Reshape for heatmap (fields as rows, single column for success rate)
        heatmap_data = df.pivot_table(
            index="field",
            values="success_rate",
            aggfunc="mean",
        )

        sns.heatmap(
            heatmap_data.values.reshape(-1, 1),
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            yticklabels=heatmap_data.index,
            xticklabels=["Success Rate"],
            ax=ax,
            cbar_kws={"label": "Validation Success Rate"},
        )

        ax.set_title(f"Field-Level Validation Success Rates ({data_type.capitalize()})")
        ax.set_ylabel("Field")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved field validation heatmap to {output_path}")
        return output_path

    def create_failure_mode_heatmap(
        self,
        analyzer: FailureModeAnalyzer,
        filename: str = "failure_mode_heatmap.png",
    ) -> Path:
        """Create a heatmap showing failure mode distribution.

        Args:
            analyzer: FailureModeAnalyzer with analyzed results.
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        if not analyzer.failure_modes:
            logfire.warning("No failure modes to visualize")
            return self._create_empty_heatmap("No Failure Modes", filename)

        # Create category-field matrix
        categories = [c.value for c in FailureCategory]
        df = analyzer.to_dataframe()

        # Get unique fields (top 10 by count)
        top_fields = df.nlargest(10, "count")["field"].unique()

        # Create pivot table
        pivot_data = []
        for field in top_fields:
            field_df = df[df["field"] == field]
            for category in categories:
                count = field_df[field_df["category"] == category]["count"].sum()
                pivot_data.append({
                    "field": field,
                    "category": category,
                    "count": count,
                })

        pivot_df = pd.DataFrame(pivot_data)
        pivot_table = pivot_df.pivot(index="field", columns="category", values="count")
        pivot_table = pivot_table.fillna(0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Error Count"},
        )

        ax.set_title("Failure Mode Distribution by Field and Category")
        ax.set_xlabel("Failure Category")
        ax.set_ylabel("Field")

        # Rotate x labels for readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved failure mode heatmap to {output_path}")
        return output_path

    def create_error_correlation_heatmap(
        self,
        results: list[ValidationResult],
        filename: str = "error_correlation_heatmap.png",
    ) -> Path:
        """Create a heatmap showing correlation between error types.

        Args:
            results: List of ValidationResult objects.
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        # Extract all unique fields with errors
        all_fields = set()
        for result in results:
            if not result.is_valid:
                for error in result.errors:
                    all_fields.add(error.field)

        if len(all_fields) < 2:
            logfire.warning("Not enough error fields for correlation analysis")
            return self._create_empty_heatmap("Insufficient Data for Correlation", filename)

        # Limit to top 15 fields
        field_counts = {}
        for result in results:
            if not result.is_valid:
                for error in result.errors:
                    field_counts[error.field] = field_counts.get(error.field, 0) + 1

        top_fields = sorted(field_counts.keys(), key=lambda x: field_counts[x], reverse=True)[:15]

        # Create co-occurrence matrix
        n_fields = len(top_fields)
        cooccurrence = np.zeros((n_fields, n_fields))

        for result in results:
            if not result.is_valid:
                error_fields = [e.field for e in result.errors if e.field in top_fields]
                for i, field1 in enumerate(top_fields):
                    for j, field2 in enumerate(top_fields):
                        if field1 in error_fields and field2 in error_fields:
                            cooccurrence[i, j] += 1

        # Normalize to correlation
        for i in range(n_fields):
            for j in range(n_fields):
                if i != j and cooccurrence[i, i] > 0 and cooccurrence[j, j] > 0:
                    cooccurrence[i, j] /= np.sqrt(cooccurrence[i, i] * cooccurrence[j, j])

        # Set diagonal to 1
        np.fill_diagonal(cooccurrence, 1)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            cooccurrence,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=top_fields,
            yticklabels=top_fields,
            ax=ax,
            cbar_kws={"label": "Correlation"},
        )

        ax.set_title("Error Field Correlation Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved error correlation heatmap to {output_path}")
        return output_path

    def create_summary_dashboard(
        self,
        results: list[ValidationResult],
        analyzer: FailureModeAnalyzer,
        filename: str = "validation_dashboard.png",
    ) -> Path:
        """Create a summary dashboard with multiple visualizations.

        Args:
            results: List of ValidationResult objects.
            analyzer: FailureModeAnalyzer with analyzed results.
            filename: Output filename.

        Returns:
            Path to the saved dashboard image.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Validation success/failure pie chart
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count

        axes[0, 0].pie(
            [valid_count, invalid_count],
            labels=["Valid", "Invalid"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 0].set_title("Validation Results")

        # 2. Errors by category bar chart
        stats = analyzer.get_statistics()
        if stats["by_category"]:
            categories = list(stats["by_category"].keys())
            counts = list(stats["by_category"].values())

            bars = axes[0, 1].barh(categories, counts, color=sns.color_palette("husl", len(categories)))
            axes[0, 1].set_xlabel("Error Count")
            axes[0, 1].set_title("Errors by Category")

            # Add count labels
            for bar, count in zip(bars, counts):
                axes[0, 1].text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    va="center",
                )
        else:
            axes[0, 1].text(0.5, 0.5, "No errors", ha="center", va="center")
            axes[0, 1].set_title("Errors by Category")

        # 3. Top error fields
        if stats["by_field"]:
            fields = list(stats["by_field"].keys())[:10]
            field_counts = [stats["by_field"][f] for f in fields]

            bars = axes[1, 0].barh(fields, field_counts, color=sns.color_palette("viridis", len(fields)))
            axes[1, 0].set_xlabel("Error Count")
            axes[1, 0].set_title("Top Error Fields")
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, "No errors", ha="center", va="center")
            axes[1, 0].set_title("Top Error Fields")

        # 4. Summary statistics text
        axes[1, 1].axis("off")
        summary_text = f"""
        Summary Statistics
        ==================

        Total Records: {len(results)}
        Valid: {valid_count} ({valid_count / len(results) * 100:.1f}%)
        Invalid: {invalid_count} ({invalid_count / len(results) * 100:.1f}%)

        Total Errors: {stats['total_errors']}
        Unique Error Modes: {stats['unique_modes']}
        """
        axes[1, 1].text(
            0.1, 0.9, summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle("Validation Analysis Dashboard", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved validation dashboard to {output_path}")
        return output_path

    def _create_empty_heatmap(self, message: str, filename: str) -> Path:
        """Create an empty heatmap with a message.

        Args:
            message: Message to display.
            filename: Output filename.

        Returns:
            Path to the saved image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_failure_mode_correlation_matrix(
        self,
        labeler: "FailureLabeler",
        filename: str = "failure_mode_correlation.png",
    ) -> Path:
        """Create a correlation matrix heatmap for failure modes.

        Shows how different failure types correlate with each other
        (e.g., do hallucinated skills often co-occur with awkward language?).

        Args:
            labeler: FailureLabeler with labeled pairs.
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        if not labeler.labels:
            logfire.warning("No labels for correlation analysis")
            return self._create_empty_heatmap("No Labels for Correlation", filename)

        # Build DataFrame of binary failure flags
        data = []
        for label in labeler.labels:
            data.append({
                "low_skills_overlap": 1 if label.skills_overlap_ratio < 0.5 else 0,
                "experience_mismatch": label.experience_mismatch,
                "seniority_mismatch": label.seniority_mismatch,
                "missing_core_skill": label.missing_core_skill,
                "hallucinated_skill": label.hallucinated_skill,
                "awkward_language": label.awkward_language_flag,
            })

        df = pd.DataFrame(data)

        # Calculate correlation matrix
        correlation_matrix = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        ax.set_title("Failure Mode Correlation Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved failure mode correlation matrix to {output_path}")
        return output_path

    def create_failure_rates_by_template_heatmap(
        self,
        labeler: "FailureLabeler",
        filename: str = "failure_by_template.png",
    ) -> Path:
        """Create a heatmap showing failure rates by prompt template.

        Helps answer: "Which prompt templates produce cleaner structures?"

        Args:
            labeler: FailureLabeler with labeled pairs.
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        if not labeler.labels:
            logfire.warning("No labels for template analysis")
            return self._create_empty_heatmap("No Labels for Template Analysis", filename)

        # Build DataFrame
        data = []
        for label in labeler.labels:
            template = label.prompt_template or "unknown"
            data.append({
                "template": template,
                "low_skills_overlap": 1 if label.skills_overlap_ratio < 0.5 else 0,
                "experience_mismatch": label.experience_mismatch,
                "seniority_mismatch": label.seniority_mismatch,
                "missing_core_skill": label.missing_core_skill,
                "hallucinated_skill": label.hallucinated_skill,
                "awkward_language": label.awkward_language_flag,
            })

        df = pd.DataFrame(data)

        # Group by template and calculate mean failure rates
        failure_cols = [
            "low_skills_overlap", "experience_mismatch", "seniority_mismatch",
            "missing_core_skill", "hallucinated_skill", "awkward_language"
        ]
        template_rates = df.groupby("template")[failure_cols].mean()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            template_rates,
            annot=True,
            fmt=".2%",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Failure Rate"},
        )

        ax.set_title("Failure Rates by Prompt Template")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Prompt Template")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved failure rates by template heatmap to {output_path}")
        return output_path

    def create_failure_rates_by_fit_level_heatmap(
        self,
        labeler: "FailureLabeler",
        filename: str = "failure_by_fit_level.png",
    ) -> Path:
        """Create a heatmap showing failure rates by fit level.

        Args:
            labeler: FailureLabeler with labeled pairs.
            filename: Output filename.

        Returns:
            Path to the saved heatmap image.
        """
        if not labeler.labels:
            logfire.warning("No labels for fit level analysis")
            return self._create_empty_heatmap("No Labels for Fit Level Analysis", filename)

        # Build DataFrame
        data = []
        for label in labeler.labels:
            fit = label.fit_level or "unknown"
            data.append({
                "fit_level": fit,
                "low_skills_overlap": 1 if label.skills_overlap_ratio < 0.5 else 0,
                "experience_mismatch": label.experience_mismatch,
                "seniority_mismatch": label.seniority_mismatch,
                "missing_core_skill": label.missing_core_skill,
                "hallucinated_skill": label.hallucinated_skill,
                "awkward_language": label.awkward_language_flag,
            })

        df = pd.DataFrame(data)

        # Define fit level order
        fit_order = ["excellent", "good", "partial", "poor", "mismatch", "unknown"]
        df["fit_level"] = pd.Categorical(df["fit_level"], categories=fit_order, ordered=True)

        # Group by fit level and calculate mean failure rates
        failure_cols = [
            "low_skills_overlap", "experience_mismatch", "seniority_mismatch",
            "missing_core_skill", "hallucinated_skill", "awkward_language"
        ]
        fit_rates = df.groupby("fit_level", observed=True)[failure_cols].mean()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            fit_rates,
            annot=True,
            fmt=".2%",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Failure Rate"},
        )

        ax.set_title("Failure Rates by Fit Level")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Fit Level")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved failure rates by fit level heatmap to {output_path}")
        return output_path

    def create_niche_vs_standard_comparison(
        self,
        labeler: "FailureLabeler",
        filename: str = "niche_vs_standard.png",
    ) -> Path:
        """Create a comparison chart of niche vs standard role failure rates.

        Helps answer: "Do niche roles have higher failure rates?"

        Args:
            labeler: FailureLabeler with labeled pairs.
            filename: Output filename.

        Returns:
            Path to the saved chart image.
        """
        if not labeler.labels:
            logfire.warning("No labels for niche analysis")
            return self._create_empty_heatmap("No Labels for Niche Analysis", filename)

        # Split labels into niche and standard
        niche_labels = [l for l in labeler.labels if l.is_niche_role]
        standard_labels = [l for l in labeler.labels if not l.is_niche_role]

        if not niche_labels or not standard_labels:
            return self._create_empty_heatmap("Insufficient Data for Niche Comparison", filename)

        # Calculate failure rates
        failure_types = [
            "low_skills_overlap", "experience_mismatch", "seniority_mismatch",
            "missing_core_skill", "hallucinated_skill", "awkward_language"
        ]

        def calc_rates(labels):
            n = len(labels)
            return {
                "low_skills_overlap": sum(1 for l in labels if l.skills_overlap_ratio < 0.5) / n,
                "experience_mismatch": sum(l.experience_mismatch for l in labels) / n,
                "seniority_mismatch": sum(l.seniority_mismatch for l in labels) / n,
                "missing_core_skill": sum(l.missing_core_skill for l in labels) / n,
                "hallucinated_skill": sum(l.hallucinated_skill for l in labels) / n,
                "awkward_language": sum(l.awkward_language_flag for l in labels) / n,
            }

        niche_rates = calc_rates(niche_labels)
        standard_rates = calc_rates(standard_labels)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(failure_types))
        width = 0.35

        bars1 = ax.bar(x - width/2, [niche_rates[ft] for ft in failure_types], width, label=f"Niche (n={len(niche_labels)})")
        bars2 = ax.bar(x + width/2, [standard_rates[ft] for ft in failure_types], width, label=f"Standard (n={len(standard_labels)})")

        ax.set_ylabel("Failure Rate")
        ax.set_title("Failure Rates: Niche vs Standard Roles")
        ax.set_xticks(x)
        ax.set_xticklabels([ft.replace("_", "\n") for ft in failure_types])
        ax.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved niche vs standard comparison to {output_path}")
        return output_path

    def create_hallucination_by_seniority_chart(
        self,
        labeler: "FailureLabeler",
        filename: str = "hallucination_by_seniority.png",
    ) -> Path:
        """Create a chart showing hallucination rates by seniority level.

        Helps answer: "Are hallucinated skills more common in senior roles?"

        Args:
            labeler: FailureLabeler with labeled pairs.
            filename: Output filename.

        Returns:
            Path to the saved chart image.
        """
        if not labeler.labels:
            logfire.warning("No labels for seniority analysis")
            return self._create_empty_heatmap("No Labels for Seniority Analysis", filename)

        # Group by fit level as proxy for seniority
        # (excellent/good fits typically match senior roles better)
        fit_levels = ["excellent", "good", "partial", "poor", "mismatch"]

        data = {fit: {"count": 0, "hallucinations": 0} for fit in fit_levels}

        for label in labeler.labels:
            fit = label.fit_level or "unknown"
            if fit in data:
                data[fit]["count"] += 1
                data[fit]["hallucinations"] += label.hallucinated_skill

        # Calculate rates
        fit_labels = []
        hallucination_rates = []
        counts = []

        for fit in fit_levels:
            if data[fit]["count"] > 0:
                fit_labels.append(fit.capitalize())
                hallucination_rates.append(data[fit]["hallucinations"] / data[fit]["count"])
                counts.append(data[fit]["count"])

        if not fit_labels:
            return self._create_empty_heatmap("No Data for Seniority Analysis", filename)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(fit_labels, hallucination_rates, color=sns.color_palette("viridis", len(fit_labels)))

        ax.set_ylabel("Hallucination Rate")
        ax.set_xlabel("Fit Level")
        ax.set_title("Hallucinated Skills Rate by Fit Level")
        ax.set_ylim(0, max(hallucination_rates) * 1.2 if hallucination_rates else 1)

        # Add value labels and counts
        for bar, rate, count in zip(bars, hallucination_rates, counts):
            height = bar.get_height()
            ax.annotate(f'{rate:.1%}\n(n={count})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logfire.info(f"Saved hallucination by seniority chart to {output_path}")
        return output_path
