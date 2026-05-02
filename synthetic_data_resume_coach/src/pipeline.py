"""Pipeline class for use by the FastAPI service and programmatic access.

For CLI usage, use src/main.py which provides _banner/_section progress output
and phase-by-phase control via --phase argument.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import logfire
from dotenv import load_dotenv

from .config import PipelineConfig
from .generators import JobDescriptionGenerator, ResumeGenerator
from .validators.schema_validator import SchemaValidator
from .analysis.failure_modes import FailureModeAnalyzer
from .analysis.failure_labeler import FailureLabeler
from .analysis.llm_judge import LLMJudge
from .analysis.heatmap import HeatmapGenerator
from .correction.llm_correction import LLMCorrector
from .schema import FitLevel
from .utils.storage import save_jsonl

try:
    from .evaluation.braintrust_eval import BraintrustEvaluator
    _BRAINTRUST_AVAILABLE = True
except ImportError:
    _BRAINTRUST_AVAILABLE = False


class Pipeline:
    """Main orchestration pipeline for synthetic data generation and validation."""

    FIT_LEVELS = [
        FitLevel.EXCELLENT,
        FitLevel.GOOD,
        FitLevel.PARTIAL,
        FitLevel.POOR,
        FitLevel.MISMATCH,
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline."""
        load_dotenv()
        self.config = config or PipelineConfig()

        self.resume_generator = ResumeGenerator(model=self.config.model)
        self.job_generator = JobDescriptionGenerator(model=self.config.model)
        self.validator = SchemaValidator()
        self.failure_analyzer = FailureModeAnalyzer()
        self.failure_labeler = FailureLabeler()
        self.heatmap_generator = HeatmapGenerator(
            output_dir=f"{self.config.output_dir}/validated"
        )
        self.corrector = LLMCorrector(
            model=self.config.model,
            max_retries=self.config.max_correction_retries,
        )

        self.llm_judge = None
        if self.config.enable_llm_judge:
            try:
                self.llm_judge = LLMJudge(model=self.config.model)
            except ValueError as e:
                logfire.warning(f"LLM Judge initialization failed: {e}")

        self.braintrust = None
        if self.config.enable_braintrust and _BRAINTRUST_AVAILABLE:
            try:
                self.braintrust = BraintrustEvaluator()
            except Exception as e:
                logfire.warning(f"Braintrust initialization failed: {e}")

        self.generated_dir = Path(self.config.output_dir) / "generated"
        self.validated_dir = Path(self.config.output_dir) / "validated"
        self.labeled_dir = Path(self.config.output_dir) / "labeled"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.validated_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)

        logfire.configure()
        logfire.info(
            "Pipeline initialized",
            num_jobs=self.config.num_jobs,
            resumes_per_job=self.config.resumes_per_job,
            model=self.config.model,
        )

    def run(
        self,
        num_jobs: Optional[int] = None,
        industries: Optional[list[str]] = None,
    ) -> dict:
        """Run the complete pipeline (jobs-first flow).

        Args:
            num_jobs: Number of jobs to generate (overrides config).
            industries: List of industries to generate for.

        Returns:
            Dictionary with pipeline results and statistics.
        """
        num_jobs = num_jobs or self.config.num_jobs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_times: dict[str, float] = {}

        results: dict = {
            "timestamp": timestamp,
            "num_jobs": num_jobs,
            "resumes_per_job": self.config.resumes_per_job,
            "generation": {},
            "validation": {},
            "correction": {},
            "analysis": {},
            "files": {},
            "stage_times_seconds": stage_times,
        }

        with logfire.span("pipeline_run", num_jobs=num_jobs):
            # ── Stage 1: Generate jobs ────────────────────────────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 1: Generating job descriptions...")
            print(f"\n{'─'*50}\nSTAGE 1 — Job Generation\n{'─'*50}")
            jobs = self.job_generator.generate_batch(
                count=num_jobs,
                industries=industries,
            )
            stage_times["generate_jobs"] = round(time.perf_counter() - t0, 2)
            results["generation"]["jobs_generated"] = len(jobs)

            jobs_file = self.job_generator.save_jobs(
                jobs,
                output_dir=str(self.generated_dir),
                filename=f"jobs_{timestamp}.jsonl",
            )
            results["files"]["jobs"] = str(jobs_file)

            # ── Stage 2: Generate resumes (5 fit levels per job) ─────────────
            t0 = time.perf_counter()
            logfire.info("Stage 2: Generating resumes for each job...")
            print(f"\n{'─'*50}\nSTAGE 2 — Resume Generation (jobs-first, {self.config.resumes_per_job} fit levels)\n{'─'*50}")
            all_pairs = []
            for i, job in enumerate(jobs):
                try:
                    pairs = self.job_generator.generate_with_multiple_resumes(
                        job=job,
                        resume_generator=self.resume_generator,
                        resumes_per_job=self.config.resumes_per_job,
                        fit_levels=self.FIT_LEVELS,
                    )
                    all_pairs.extend(pairs)
                    logfire.info(
                        f"Generated {len(pairs)} resumes for job {i + 1}/{len(jobs)}"
                    )
                except Exception as e:
                    logfire.error(
                        f"Failed generating resumes for job {i + 1}", error=str(e)
                    )

            stage_times["generate_resumes"] = round(time.perf_counter() - t0, 2)
            results["generation"]["pairs_generated"] = len(all_pairs)
            results["generation"]["resumes_generated"] = len(all_pairs)

            resumes = [p.resume for p in all_pairs]

            pairs_file = self.job_generator.save_pairs(
                all_pairs,
                output_dir=str(self.generated_dir),
                filename=f"pairs_{timestamp}.jsonl",
            )
            results["files"]["pairs"] = str(pairs_file)

            resumes_file = self.resume_generator.save_resumes(
                resumes,
                output_dir=str(self.generated_dir),
                filename=f"resumes_{timestamp}.jsonl",
            )
            results["files"]["resumes"] = str(resumes_file)

            # ── Stage 3: Validate resumes ─────────────────────────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 3: Validating resumes...")
            print(f"\n{'─'*50}\nSTAGE 3 — Resume Validation\n{'─'*50}")
            resume_data = [r.model_dump(mode="json") for r in resumes]
            resume_results, resume_summary = self.validator.validate_batch(
                resume_data, data_type="resume"
            )
            stage_times["validate_resumes"] = round(time.perf_counter() - t0, 2)
            results["validation"]["resumes"] = resume_summary

            # ── Stage 4: Validate jobs ────────────────────────────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 4: Validating job descriptions...")
            print(f"\n{'─'*50}\nSTAGE 4 — Job Validation\n{'─'*50}")
            job_data = [j.model_dump(mode="json") for j in jobs]
            job_results, job_summary = self.validator.validate_batch(
                job_data, data_type="job"
            )
            stage_times["validate_jobs"] = round(time.perf_counter() - t0, 2)
            results["validation"]["jobs"] = job_summary

            # ── Stage 5: Schema failure mode analysis ─────────────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 5: Analyzing schema validation failure modes...")
            print(f"\n{'─'*50}\nSTAGE 5 — Failure Mode Analysis\n{'─'*50}")
            all_results = resume_results + job_results
            self.failure_analyzer.analyze_results(all_results)
            failure_stats = self.failure_analyzer.get_statistics()
            results["analysis"]["schema_failure_modes"] = failure_stats

            failure_file = self.failure_analyzer.save_analysis(
                output_dir=str(self.validated_dir),
                filename=f"schema_failure_modes_{timestamp}.json",
            )
            results["files"]["schema_failure_analysis"] = str(failure_file)
            stage_times["schema_analysis"] = round(time.perf_counter() - t0, 2)

            # ── Stage 5b: Label resume-job pair failure modes ──────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 5b: Labeling resume-job pair failures...")
            pair_labels = self.failure_labeler.label_pairs(all_pairs)
            labeler_stats = self.failure_labeler.get_statistics()
            results["analysis"]["pair_failure_labels"] = labeler_stats

            labels_file = self.failure_labeler.save_labels(
                output_dir=str(self.labeled_dir),
                filename=f"failure_labels_{timestamp}.jsonl",
            )
            results["files"]["failure_labels"] = str(labels_file)

            invalid_labels = [label for label in pair_labels if not label.overall_pass]
            if invalid_labels:
                invalid_data = [label.to_dict() for label in invalid_labels]
                invalid_file = save_jsonl(
                    invalid_data,
                    self.labeled_dir / f"invalid_{timestamp}.jsonl",
                )
                results["files"]["invalid_records"] = str(invalid_file)

            if self.braintrust and getattr(self.braintrust, "enabled", False):
                self.braintrust.log_batch(pair_labels)

            stage_times["label_failures"] = round(time.perf_counter() - t0, 2)

            # ── Stage 5c: LLM Judge on sample (optional) ──────────────────────
            if self.llm_judge and all_pairs:
                t0 = time.perf_counter()
                logfire.info("Stage 5c: Running LLM Judge on sample pairs...")
                sample_pairs = all_pairs[: min(10, len(all_pairs))]
                judgments = []
                for pair in sample_pairs:
                    try:
                        judgment = self.llm_judge.judge_pair(
                            pair.resume, pair.job_description
                        )
                        judgments.append(judgment)
                        if self.braintrust and getattr(self.braintrust, "enabled", False):
                            self.braintrust.log_llm_judgment(judgment)
                    except Exception as e:
                        logfire.warning(f"LLM Judge failed for pair: {e}")

                stage_times["llm_judge"] = round(time.perf_counter() - t0, 2)
                results["analysis"]["llm_judgments"] = {
                    "judged_count": len(judgments),
                    "stats": self.llm_judge.get_statistics(),
                }

            # ── Stage 6: Correction loop (optional) ───────────────────────────
            if self.config.enable_correction:
                t0 = time.perf_counter()
                logfire.info("Stage 6: Running correction loop...")
                print(f"\n{'─'*50}\nSTAGE 6 — Correction Loop\n{'─'*50}")
                invalid_resumes = [r for r in resume_results if not r.is_valid]
                invalid_jobs = [r for r in job_results if not r.is_valid]
                correction_results = []

                if invalid_resumes:
                    logfire.info(f"Correcting {len(invalid_resumes)} invalid resumes...")
                    resume_corrections = self.corrector.correct_batch(
                        invalid_resumes, data_type="resume"
                    )
                    correction_results.extend(resume_corrections)

                if invalid_jobs:
                    logfire.info(f"Correcting {len(invalid_jobs)} invalid jobs...")
                    job_corrections = self.corrector.correct_batch(
                        invalid_jobs, data_type="job"
                    )
                    correction_results.extend(job_corrections)

                results["correction"] = self.corrector.get_stats()
                stage_times["correction"] = round(time.perf_counter() - t0, 2)

                if correction_results:
                    correction_file = self.corrector.save_results(
                        correction_results,
                        output_dir=str(self.validated_dir),
                        filename=f"corrections_{timestamp}.jsonl",
                    )
                    results["files"]["corrections"] = str(correction_file)

            # ── Stage 7: Visualizations (optional) ────────────────────────────
            if self.config.generate_heatmaps:
                t0 = time.perf_counter()
                logfire.info("Stage 7: Generating visualizations...")
                print(f"\n{'─'*50}\nSTAGE 7 — Visualizations\n{'─'*50}")

                resume_heatmap = self.heatmap_generator.create_field_validation_heatmap(
                    resume_results,
                    data_type="resume",
                    filename=f"resume_field_heatmap_{timestamp}.png",
                )
                results["files"]["resume_heatmap"] = str(resume_heatmap)

                job_heatmap = self.heatmap_generator.create_field_validation_heatmap(
                    job_results,
                    data_type="job",
                    filename=f"job_field_heatmap_{timestamp}.png",
                )
                results["files"]["job_heatmap"] = str(job_heatmap)

                failure_heatmap = self.heatmap_generator.create_failure_mode_heatmap(
                    self.failure_analyzer,
                    filename=f"schema_failure_heatmap_{timestamp}.png",
                )
                results["files"]["schema_failure_heatmap"] = str(failure_heatmap)

                dashboard = self.heatmap_generator.create_summary_dashboard(
                    all_results,
                    self.failure_analyzer,
                    filename=f"dashboard_{timestamp}.png",
                )
                results["files"]["dashboard"] = str(dashboard)

                if self.failure_labeler.labels:
                    corr_heatmap = self.heatmap_generator.create_failure_mode_correlation_matrix(
                        self.failure_labeler,
                        filename=f"failure_correlation_{timestamp}.png",
                    )
                    results["files"]["failure_correlation"] = str(corr_heatmap)

                    template_heatmap = self.heatmap_generator.create_failure_rates_by_template_heatmap(
                        self.failure_labeler,
                        filename=f"failure_by_template_{timestamp}.png",
                    )
                    results["files"]["failure_by_template"] = str(template_heatmap)

                    fit_heatmap = self.heatmap_generator.create_failure_rates_by_fit_level_heatmap(
                        self.failure_labeler,
                        filename=f"failure_by_fit_level_{timestamp}.png",
                    )
                    results["files"]["failure_by_fit_level"] = str(fit_heatmap)

                    niche_chart = self.heatmap_generator.create_niche_vs_standard_comparison(
                        self.failure_labeler,
                        filename=f"niche_vs_standard_{timestamp}.png",
                    )
                    results["files"]["niche_vs_standard"] = str(niche_chart)

                    hall_chart = self.heatmap_generator.create_hallucination_by_seniority_chart(
                        self.failure_labeler,
                        filename=f"hallucination_by_seniority_{timestamp}.png",
                    )
                    results["files"]["hallucination_by_seniority"] = str(hall_chart)

                stage_times["visualizations"] = round(time.perf_counter() - t0, 2)

            # ── Stage 8: Export validated data as JSONL ────────────────────────
            t0 = time.perf_counter()
            logfire.info("Stage 8: Exporting validated data...")

            valid_resumes = [
                r.data.model_dump(mode="json")
                for r in resume_results
                if r.is_valid and r.data
            ]
            valid_jobs = [
                r.data.model_dump(mode="json")
                for r in job_results
                if r.is_valid and r.data
            ]

            if valid_resumes:
                valid_resumes_file = self.validated_dir / f"valid_resumes_{timestamp}.jsonl"
                save_jsonl(valid_resumes, valid_resumes_file)
                results["files"]["valid_resumes"] = str(valid_resumes_file)

            if valid_jobs:
                valid_jobs_file = self.validated_dir / f"valid_jobs_{timestamp}.jsonl"
                save_jsonl(valid_jobs, valid_jobs_file)
                results["files"]["valid_jobs"] = str(valid_jobs_file)

            stage_times["export"] = round(time.perf_counter() - t0, 2)

            # ── Final stats ────────────────────────────────────────────────────
            results["final_stats"] = {
                "jobs_generated": len(jobs),
                "resumes_generated": len(resumes),
                "valid_resumes": len(valid_resumes),
                "valid_jobs": len(valid_jobs),
                "resume_success_rate": len(valid_resumes) / len(resumes) if resumes else 0,
                "job_success_rate": len(valid_jobs) / len(jobs) if jobs else 0,
            }

            if self.failure_labeler.labels:
                labeler_stats = self.failure_labeler.get_statistics()
                results["final_stats"]["pair_analysis"] = {
                    "total_pairs": labeler_stats["total_pairs"],
                    "overall_pass_rate": labeler_stats["overall_pass_rate"],
                    "average_skills_overlap": labeler_stats["average_skills_overlap"],
                    "failure_rates": labeler_stats["failure_rates"],
                }

            if (
                self.braintrust
                and getattr(self.braintrust, "enabled", False)
                and self.failure_labeler.labels
            ):
                logfire.info("Running Braintrust evaluation...")
                eval_metrics = self.braintrust.run_evaluation(
                    self.failure_labeler,
                    eval_name=f"pipeline_run_{timestamp}",
                )
                results["analysis"]["braintrust_evaluation"] = {
                    "total_pairs": eval_metrics.total_pairs,
                    "average_skills_overlap": eval_metrics.average_skills_overlap,
                    "failure_rate": eval_metrics.failure_rate,
                    "hallucination_rate": eval_metrics.hallucination_rate,
                }

            logfire.info("Pipeline complete", **results["final_stats"])

        # ── Save pipeline results summary ──────────────────────────────────────
        summary_file = self.validated_dir / f"pipeline_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        results["files"]["summary"] = str(summary_file)

        # ── Append to iteration log ────────────────────────────────────────────
        self._append_iteration_log(results, timestamp)

        return results

    def _append_iteration_log(self, results: dict, timestamp: str) -> None:
        """Append a structured entry to the iteration log."""
        log_file = Path(self.config.output_dir) / "iteration_log.jsonl"
        entry = {
            "timestamp": timestamp,
            "config": {
                "num_jobs": self.config.num_jobs,
                "resumes_per_job": self.config.resumes_per_job,
                "model": self.config.model,
                "max_correction_retries": self.config.max_correction_retries,
            },
            "metrics": results.get("final_stats", {}),
            "stage_times_seconds": results.get("stage_times_seconds", {}),
            "files": results.get("files", {}),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logfire.info("Iteration log updated", log_file=str(log_file))


def main():
    """Run the pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthetic Data Resume Coach - Data Generation Pipeline"
    )
    parser.add_argument(
        "--num-jobs", "-n",
        type=int,
        default=10,
        help="Number of job descriptions to generate (default: 10)",
    )
    parser.add_argument(
        "--resumes-per-job",
        type=int,
        default=5,
        help="Number of resumes to generate per job (default: 5)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama-3.3-70b-versatile",
        help="LLM model to use (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Disable correction loop",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Disable heatmap generation",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable LLM judge for quality assessment (slower)",
    )
    parser.add_argument(
        "--enable-braintrust",
        action="store_true",
        help="Enable Braintrust logging (requires BRAINTRUST_API_KEY)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        num_jobs=args.num_jobs,
        resumes_per_job=args.resumes_per_job,
        model=args.model,
        enable_correction=not args.no_correction,
        generate_heatmaps=not args.no_heatmaps,
        enable_llm_judge=args.enable_llm_judge,
        enable_braintrust=args.enable_braintrust,
        output_dir=args.output_dir,
    )

    pipeline = Pipeline(config)
    results = pipeline.run()

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    for key, path in results.get("files", {}).items():
        print(f"  - {key}: {path}")

    print("\nFinal Statistics:")
    for key, value in results.get("final_stats", {}).items():
        if isinstance(value, dict):
            print(f"  - {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.2%}")
                elif isinstance(v, dict):
                    print(f"      {k}: {v}")
                else:
                    print(f"      {k}: {v}")
        elif isinstance(value, float):
            print(f"  - {key}: {value:.2%}")
        else:
            print(f"  - {key}: {value}")

    print("\nStage Timings:")
    for stage, seconds in results.get("stage_times_seconds", {}).items():
        print(f"  - {stage}: {seconds}s")

    if "pair_failure_labels" in results.get("analysis", {}):
        print("\nPair Analysis Insights:")
        pair_stats = results["analysis"]["pair_failure_labels"]
        print(f"  - Total pairs analyzed: {pair_stats.get('total_pairs', 0)}")
        print(f"  - Overall pass rate: {pair_stats.get('overall_pass_rate', 0):.2%}")
        print(f"  - Average skills overlap: {pair_stats.get('average_skills_overlap', 0):.2%}")
        if "failure_rates" in pair_stats:
            print("  - Failure rates:")
            for failure_type, rate in pair_stats["failure_rates"].items():
                print(f"      {failure_type}: {rate:.2%}")


if __name__ == "__main__":
    main()
