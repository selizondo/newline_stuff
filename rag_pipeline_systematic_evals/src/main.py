"""
CLI entry point for P3 systematic RAG evaluation.

Usage:
    python -m src.main <pdf_path> [options]

Options:
    --pdf PATH              PDF to evaluate against (required)
    --n-pairs INT           QA pairs per chunk config (default: 25)
    --out-dir PATH          Experiment results directory (default: experiments)
    --qa-dir PATH           QA dataset cache directory (default: data/qa_datasets)
    --viz-dir PATH          Chart output directory (default: visualizations)
    --force                 Re-run even if result files already exist
    --no-charts             Skip chart generation
    --top-n INT             Number of top configs shown in summary (default: 5)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import EvaluationResult, build_experiment_grid
from src.evaluator import best_config
from src.grid_search import run_grid_search
from src.visualizer import generate_all_charts

console = Console()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Systematic RAG pipeline evaluation across a 4×2×3 config grid.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF document")
    parser.add_argument("--n-pairs",  type=int, default=25,
                        help="Synthetic QA pairs per chunk config (default: 25)")
    parser.add_argument("--out-dir",  type=Path, default=Path("experiments"),
                        help="Experiment results directory")
    parser.add_argument("--qa-dir",   type=Path, default=Path("data/qa_datasets"),
                        help="QA dataset cache directory")
    parser.add_argument("--viz-dir",  type=Path, default=Path("visualizations"),
                        help="Chart output directory")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run even if result files already exist")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--top-n",   type=int, default=5,
                        help="Number of top configs shown in summary table")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply cross-encoder reranking after retrieval (requires sentence-transformers)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _banner(pdf_path: Path, n_experiments: int) -> None:
    console.print(Panel(
        f"[bold cyan]RAG Pipeline — Systematic Evaluation[/]\n\n"
        f"  PDF          : [green]{pdf_path}[/]\n"
        f"  Experiments  : [yellow]{n_experiments}[/] configurations\n"
        f"  Grid         : 4 chunk × 2 embed × 3 retrieval",
        title="[bold]P3[/]",
        expand=False,
    ))


def _section(title: str) -> None:
    console.rule(f"[bold blue]{title}[/]")


def _print_results_table(results: list[EvaluationResult], top_n: int) -> None:
    sorted_results = sorted(results, key=lambda r: r.metrics.mrr, reverse=True)
    top = sorted_results[:top_n]

    table = Table(title=f"Top {top_n} Configurations by MRR", show_lines=True)
    table.add_column("Rank",       style="bold", justify="center", width=5)
    table.add_column("Experiment",             width=38)
    table.add_column("MRR",        justify="right")
    table.add_column("MAP",        justify="right")
    table.add_column("Recall@5",   justify="right")
    table.add_column("NDCG@5",     justify="right")
    table.add_column("Queries",    justify="right")
    table.add_column("Time (ms)",  justify="right")

    for rank, r in enumerate(top, 1):
        m = r.metrics
        table.add_row(
            str(rank),
            r.experiment_id,
            f"[bold green]{m.mrr:.4f}[/]" if rank == 1 else f"{m.mrr:.4f}",
            f"{m.map_score:.4f}",
            f"{m.recall_at_k.get('5', 0):.4f}",
            f"{m.ndcg_at_k.get('5', 0):.4f}",
            str(m.total_queries),
            f"{m.avg_retrieval_time_s * 1000:.1f}",
        )

    console.print(table)


def _print_best_summary(best: EvaluationResult) -> None:
    m = best.metrics
    console.print(Panel(
        f"[bold yellow]Best Configuration[/]: [cyan]{best.experiment_id}[/]\n\n"
        f"  MRR        : [bold]{m.mrr:.4f}[/]\n"
        f"  MAP        : {m.map_score:.4f}\n"
        f"  Recall@5   : {m.recall_at_k.get('5', 0):.4f}\n"
        f"  NDCG@5     : {m.ndcg_at_k.get('5', 0):.4f}\n"
        f"  Queries    : {m.total_queries}\n"
        f"  Avg latency: {m.avg_retrieval_time_s * 1000:.1f} ms",
        title="[bold green]Winner[/]",
        expand=False,
    ))


def _print_chart_paths(saved: dict[str, Path]) -> None:
    table = Table(title="Generated Charts", show_header=True)
    table.add_column("Chart",  style="cyan")
    table.add_column("Path",   style="dim")
    for name, path in saved.items():
        table.add_row(name, str(path))
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.pdf.exists():
        console.print(f"[red]Error:[/] PDF not found: {args.pdf}")
        return 1

    grid = build_experiment_grid(qa_pairs_per_config=args.n_pairs)
    _banner(args.pdf, len(grid))

    # --- Grid search ---
    _section("Running experiments")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Experiments", total=len(grid))

        results = run_grid_search(
            pdf_path=args.pdf,
            experiment_dir=args.out_dir,
            qa_dir=args.qa_dir,
            n_pairs=args.n_pairs,
            force=args.force,
            use_reranking=args.rerank,
        )
        progress.update(task, completed=len(results))

    console.print(f"[green]✓[/] Completed {len(results)} experiments.")

    # --- Results summary ---
    _section("Results")
    _print_results_table(results, top_n=args.top_n)

    best = best_config(results, primary="mrr")
    _print_best_summary(best)

    # --- Charts ---
    if not args.no_charts:
        _section("Generating charts")
        saved = generate_all_charts(results, output_dir=args.viz_dir)
        _print_chart_paths(saved)

    return 0


if __name__ == "__main__":
    sys.exit(main())
