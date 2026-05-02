"""
Run the evaluation grid against a corpus of PDFs.

Usage:
    python scripts/evaluate.py data/papers/ data/qrels.json \\
        --config config/experiments/baseline.yaml \\
        -o experiments/results/ \\
        --limit 50 --force

Outputs one JSON file per experiment cell to the results directory.
Prints a ranked summary table when complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import build_experiment_grid, build_grid_from_yaml
from src.evaluator import best_config, filter_qrels_by_docs, load_qrels
from src.experiment import run_grid
from src.models import ExperimentResult

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description="Run evaluation grid over a PDF corpus with qrels ground truth.",
    )
    p.add_argument("papers_dir", type=Path, help="Directory containing PDF files")
    p.add_argument("qrels",      type=Path, help="qrels.json file")
    p.add_argument("--config", "-c", type=Path, default=None,
                   help="YAML experiment config (default: built-in grid)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("experiments/results"),
                   help="Output directory for result JSONs (default: experiments/results)")
    p.add_argument("--index-dir", type=Path, default=Path("data/indices"),
                   help="Root directory for FAISS indices (default: data/indices)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max PDFs to ingest (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells that already have a result file")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top configs to show in summary (default: 3)")
    return p.parse_args(argv)


def _print_summary(results: list[ExperimentResult], top_k: int = 3) -> None:
    console.print()
    table = Table(title="Evaluation Results (ranked by MRR)", show_lines=False)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Experiment ID", style="cyan")
    table.add_column("MRR",      justify="right")
    table.add_column("MAP",      justify="right")
    table.add_column("R@5",      justify="right")
    table.add_column("NDCG@5",   justify="right")
    table.add_column("Latency",  justify="right", style="dim")

    ranked = sorted(results, key=lambda r: r.metrics.get("mrr", 0.0), reverse=True)
    for i, r in enumerate(ranked[:top_k]):
        m = r.metrics
        table.add_row(
            str(i + 1),
            r.experiment_id,
            f"{m.get('mrr', 0):.4f}",
            f"{m.get('map', 0):.4f}",
            f"{m.get('recall@5', 0):.4f}",
            f"{m.get('ndcg@5', 0):.4f}",
            f"{r.avg_latency_s*1000:.1f}ms",
        )

    console.print(table)

    winner = ranked[0]
    console.print(
        f"\n[bold green]Best config:[/] [cyan]{winner.experiment_id}[/] "
        f"(MRR={winner.metrics.get('mrr', 0):.4f})"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.papers_dir.exists():
        console.print(f"[red]Error:[/] papers directory not found: {args.papers_dir}")
        return 1

    if not args.qrels.exists():
        console.print(f"[red]Error:[/] qrels file not found: {args.qrels}")
        return 1

    pdf_paths = sorted(args.papers_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        console.print(f"[red]Error:[/] no PDF files found in {args.papers_dir}")
        return 1

    qrels_all = load_qrels(args.qrels)
    ingested_ids = {p.stem for p in pdf_paths}
    qrels = filter_qrels_by_docs(qrels_all, ingested_ids)

    configs = build_grid_from_yaml(args.config) if args.config else build_experiment_grid()

    filtered_note = (
        f" [dim](filtered from {len(qrels_all)})[/]" if len(qrels) < len(qrels_all) else ""
    )
    console.print(Panel(
        f"[bold cyan]RAG Pipeline — Evaluation Grid[/]\n\n"
        f"  PDFs        : [green]{len(pdf_paths)}[/]\n"
        f"  Queries     : [green]{len(qrels)}[/] in qrels{filtered_note}\n"
        f"  Grid cells  : [yellow]{len(configs)}[/]\n"
        f"  Results dir : [dim]{args.out_dir}[/]",
        title="[bold]P4[/]", expand=False,
    ))

    completed: list[ExperimentResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments", total=len(configs))

        def _cb(i: int, total: int, experiment_id: str) -> None:
            progress.update(task, completed=i, description=f"[cyan]{experiment_id}[/]")

        completed = run_grid(
            configs=configs,
            pdf_paths=pdf_paths,
            qrels=qrels,
            result_dir=args.out_dir,
            index_base_dir=args.index_dir,
            force=args.force,
            progress_cb=_cb,
        )
        progress.update(task, completed=len(configs))

    _print_summary(completed, top_k=args.top_k)
    console.print(
        f"\n[green]✓[/] {len(completed)} results written to [dim]{args.out_dir}[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
