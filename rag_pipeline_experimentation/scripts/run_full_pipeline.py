"""
Full P4 pipeline runner: load best config → generate answers → judge → visualize → log.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --n-queries 10 --result-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_common.chunkers import FixedSizeChunker
from src.embedders import SentenceTransformersEmbedder
from src.evaluator import load_result, load_qrels, filter_qrels_by_docs
from src.generator import generate_answer
from src.judge import judge_batch
from src.iteration_log import log_iteration
from src.models import ExperimentResult, JudgeScore
from src.pipeline import RAGPipeline
from src.visualizer import (
    plot_metrics_heatmap,
    plot_dimension_impact,
    plot_before_after,
    plot_radar,
    plot_latency,
    plot_fusion_sweep,
)

console = Console()


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full P4 pipeline end-to-end")
    p.add_argument("--result-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--index-dir",  type=Path, default=Path("data/indices"))
    p.add_argument("--qrels",      type=Path, default=Path("data/qrels_filtered.json"))
    p.add_argument("--n-queries",  type=int,  default=10,
                   help="Number of queries to run through generator+judge (default: 10)")
    p.add_argument("--viz-dir",    type=Path, default=Path("visualizations"))
    p.add_argument("--log",        type=Path, default=Path("experiments/iteration_log.jsonl"))
    p.add_argument("--model",      type=str,  default="",
                   help="Override LLM model (default: from LLM_MODEL env)")
    p.add_argument("--top-k",      type=int,  default=3,
                   help="Chunks per query for generation (default: 3)")
    p.add_argument("--max-tokens", type=int,  default=600,
                   help="Max tokens for each generated answer (default: 600)")
    p.add_argument("--delay",      type=float, default=15.0,
                   help="Seconds to sleep between generation calls (default: 15)")
    return p.parse_args(argv)


def _load_all_results(result_dir: Path) -> list[ExperimentResult]:
    results = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            results.append(load_result(path))
        except Exception as e:
            console.print(f"[yellow]  skip {path.name}: {e}[/]")
    return results


def _best_result(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda r: r.metrics.get("mrr", 0.0))


def _load_pipeline(result: ExperimentResult, index_base_dir: Path) -> RAGPipeline:
    cfg = result.config
    embed_model = cfg.get("embed", {}).get("model", "all-MiniLM-L6-v2")
    retrieval    = cfg.get("retrieval", {}).get("method", "dense")
    alpha        = cfg.get("retrieval", {}).get("alpha", 0.6)

    embedder = SentenceTransformersEmbedder(
        model_name=embed_model,
        cache_dir=Path("data/embed_cache"),
    )
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(512, 64),
        embedder=embedder,
        retrieval_method=retrieval,
        alpha=alpha,
    )
    index_dir = index_base_dir / result.experiment_id
    pipeline.load(index_dir)
    return pipeline


def _fusion_sweep_data(results: list[ExperimentResult]) -> dict[float, float]:
    """Extract alpha → NDCG@5 from available hybrid + dense results."""
    sweep: dict[float, float] = {}
    for r in results:
        method = r.config.get("retrieval", {}).get("method", "")
        score  = r.metrics.get("ndcg@5", 0.0)
        if method == "dense":
            sweep[1.0] = max(sweep.get(1.0, 0.0), score)
        elif method == "hybrid":
            alpha = r.config.get("retrieval", {}).get("alpha", 0.6)
            sweep[float(alpha)] = max(sweep.get(float(alpha), 0.0), score)
        elif method == "bm25":
            sweep[0.0] = max(sweep.get(0.0, 0.0), score)
    return sweep


def main(argv=None) -> int:
    args = _parse_args(argv)

    # ── 1. Load experiment results ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1 / 5 — Loading experiment results[/]")
    results = _load_all_results(args.result_dir)
    if not results:
        console.print(f"[red]No results found in {args.result_dir}[/]")
        return 1
    console.print(f"  Loaded {len(results)} results.")

    best = _best_result(results)
    console.print(
        f"  Best config: [cyan]{best.experiment_id}[/]  "
        f"MRR={best.metrics.get('mrr', 0):.4f}  NDCG@5={best.metrics.get('ndcg@5', 0):.4f}"
    )

    # ── 2. Generate + judge ───────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2 / 5 — Loading index for best config[/]")
    pipeline = _load_pipeline(best, args.index_dir)
    n_chunks = len(pipeline.chunks)
    n_docs   = len(pipeline.documents)
    console.print(f"  Loaded {n_chunks:,} chunks from {n_docs} documents.")

    console.print("\n[bold cyan]Step 3 / 5 — Generating answers + judging[/]")
    if not args.qrels.exists():
        console.print(f"[red]qrels not found: {args.qrels}[/]")
        return 1

    qrels_all = load_qrels(args.qrels)
    indexed_doc_ids = {doc.id for doc in pipeline.documents}
    qrels = filter_qrels_by_docs(qrels_all, indexed_doc_ids)
    console.print(f"  Filtered qrels: {len(qrels)} queries with relevant docs in index "
                  f"(from {len(qrels_all)} total)")

    queries = list(qrels.items())[: args.n_queries]
    console.print(f"  Running {len(queries)} queries through generator + judge…")

    from llm_utils.config import get_settings
    llm_model = args.model or get_settings().generation_model

    import re as _re
    import time as _time
    import llm_utils.client as _llm_client
    from openai import RateLimitError

    # Disable llm_utils' 240s inter-cycle sleep so our outer retry handles pacing.
    _llm_client.INTER_CYCLE_SLEEP = 0.0

    def _generate_with_backoff(query, retrieval_results, model, max_tokens):
        for attempt in range(6):
            try:
                return generate_answer(query, retrieval_results, model=model,
                                       max_tokens=max_tokens, max_retries=0)
            except RateLimitError as e:
                wait = 65
                m = _re.search(r"try again in ([\d.]+)s", str(e))
                if m:
                    wait = int(float(m.group(1))) + 5
                console.print(f"  [yellow]rate limit — waiting {wait}s… (attempt {attempt+1}/6)[/]")
                _time.sleep(wait)
        raise RuntimeError("rate limit retries exhausted")

    qa_pairs = []
    for i, (qid, entry) in enumerate(queries, 1):
        query = entry["query"]
        console.print(f"  [{i}/{len(queries)}] {query[:60]}")
        with console.status("  retrieving…"):
            retrieval_results = pipeline.query(query, top_k=args.top_k)
        with console.status("  generating…"):
            qa = _generate_with_backoff(query, retrieval_results, llm_model, args.max_tokens)
        qa_pairs.append(qa)
        if i < len(queries):
            _time.sleep(args.delay)

    console.print("  Judging answers…")
    scores: list[JudgeScore] = judge_batch(qa_pairs, model=llm_model)

    avg_scores = {
        "relevance":       round(sum(s.relevance        for s in scores) / len(scores), 4),
        "accuracy":        round(sum(s.accuracy         for s in scores) / len(scores), 4),
        "completeness":    round(sum(s.completeness     for s in scores) / len(scores), 4),
        "citation_quality":round(sum(s.citation_quality for s in scores) / len(scores), 4),
    }
    avg_overall = round(sum(avg_scores.values()) / 4, 4)
    console.print(f"  Avg judge score: [green]{avg_overall:.2f}[/] / 5.0  {avg_scores}")

    # ── 3. Visualizations ────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 4 / 5 — Generating visualizations[/]")
    viz_dir = args.viz_dir

    worst = min(results, key=lambda r: r.metrics.get("mrr", 0.0))
    before_metrics = {k: v for k, v in worst.metrics.items() if k in ("mrr", "ndcg@5", "recall@5", "precision@5")}
    after_metrics  = {k: v for k, v in best.metrics.items()  if k in ("mrr", "ndcg@5", "recall@5", "precision@5")}

    fusion_data = _fusion_sweep_data(results)

    charts = [
        ("metrics_heatmap.png",       lambda: plot_metrics_heatmap(results, out_dir=viz_dir)),
        ("dimension_impact.png",      lambda: plot_dimension_impact(results, out_dir=viz_dir)),
        ("before_after.png",          lambda: plot_before_after(before_metrics, after_metrics,
                                                                label_before=worst.experiment_id[:20],
                                                                label_after=best.experiment_id[:20],
                                                                out_dir=viz_dir)),
        ("radar_generation.png",      lambda: plot_radar(scores, config_label=best.experiment_id[:30], out_dir=viz_dir)),
        ("latency_distribution.png",  lambda: plot_latency(results, out_dir=viz_dir)),
        ("fusion_sweep.png",          lambda: plot_fusion_sweep(fusion_data, out_dir=viz_dir)),
    ]

    for name, fn in charts:
        with console.status(f"  {name}…"):
            path = fn()
        console.print(f"  [green]✓[/] {path}")

    # ── 4. Iteration log ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 5 / 5 — Logging iteration[/]")
    log_iteration(
        change=f"Full pipeline run — best config: {best.experiment_id}",
        reason=f"20-paper baseline grid complete. MRR={best.metrics.get('mrr', 0):.4f}",
        after_metrics={
            **{k: best.metrics[k] for k in ("mrr", "ndcg@5", "recall@5") if k in best.metrics},
            "avg_judge_score": avg_overall,
        },
        config=best.config,
        log_path=args.log,
    )
    console.print(f"  [green]✓[/] Logged to {args.log}")

    # ── Summary ───────────────────────────────────────────────────────────────
    table = Table(title="Pipeline Run Summary", show_lines=False)
    table.add_column("Step", style="cyan")
    table.add_column("Result")
    table.add_row("Experiments loaded",  str(len(results)))
    table.add_row("Best config",          best.experiment_id)
    table.add_row("Best MRR",            f"{best.metrics.get('mrr', 0):.4f}")
    table.add_row("Queries generated",   str(len(qa_pairs)))
    table.add_row("Avg judge score",     f"{avg_overall:.2f} / 5.0")
    table.add_row("Charts saved",        str(len(charts)))
    table.add_row("Iteration log",       str(args.log))
    console.print()
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
