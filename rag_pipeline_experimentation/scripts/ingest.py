"""
Ingest PDFs into a FAISS index.

Usage:
    python scripts/ingest.py data/papers/ -o data/indices/baseline_index \\
        --chunk-strategy recursive --chunk-size 512 --overlap 100 \\
        --embed-model all-MiniLM-L6-v2 --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Allow running as `python scripts/ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModelName, RetrievalConfig
from src.chunkers_ext import RecursiveChunker, SlidingWindowChunker
from src.embedders import SentenceTransformersEmbedder
from src.pipeline import RAGPipeline
from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker

console = Console()

_CHUNKER_MAP = {
    "fixed":          lambda cfg: FixedSizeChunker(cfg.chunk_size, cfg.overlap),
    "recursive":      lambda cfg: RecursiveChunker(cfg.chunk_size, cfg.overlap),
    "sliding_window": lambda cfg: SlidingWindowChunker(cfg.window_size, cfg.step),
    "sentence":       lambda cfg: SentenceBasedChunker(cfg.sentences_per_chunk, cfg.overlap_sentences),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest",
        description="Parse PDFs and build a FAISS retrieval index.",
    )
    p.add_argument("papers_dir", type=Path, help="Directory containing PDF files")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("data/indices/default"),
                   help="Output directory for FAISS index (default: data/indices/default)")
    p.add_argument("--chunk-strategy", default="recursive",
                   choices=list(_CHUNKER_MAP), help="Chunking strategy")
    p.add_argument("--chunk-size",  type=int, default=512)
    p.add_argument("--overlap",     type=int, default=100)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--step",        type=int, default=5)
    p.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                   choices=[m.value for m in EmbedModelName])
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of PDFs to ingest (default: all)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.papers_dir.exists():
        console.print(f"[red]Error:[/] papers directory not found: {args.papers_dir}")
        return 1

    pdf_paths = sorted(args.papers_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        console.print(f"[red]Error:[/] no PDF files found in {args.papers_dir}")
        return 1

    console.print(Panel(
        f"[bold cyan]RAG Pipeline — Ingestion[/]\n\n"
        f"  PDFs           : [green]{len(pdf_paths)}[/]\n"
        f"  Chunk strategy : [yellow]{args.chunk_strategy}[/] "
        f"(size={args.chunk_size}, overlap={args.overlap})\n"
        f"  Embed model    : [yellow]{args.embed_model}[/]\n"
        f"  Output         : [dim]{args.out_dir}[/]",
        title="[bold]P4[/]", expand=False,
    ))

    chunk_cfg = ChunkConfig(
        strategy=ChunkStrategy(args.chunk_strategy),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        window_size=args.window_size,
        step=args.step,
    )
    chunker = _CHUNKER_MAP[args.chunk_strategy](chunk_cfg)
    embedder = SentenceTransformersEmbedder(
        model_name=args.embed_model,
        batch_size=args.batch_size,
    )

    pipeline = RAGPipeline(chunker=chunker, embedder=embedder)
    chunk_label = chunk_cfg.label()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting PDFs", total=len(pdf_paths))

        chunks = pipeline.ingest(
            pdf_paths=pdf_paths,
            index_dir=args.out_dir,
            chunk_label=chunk_label,
        )
        progress.update(task, completed=len(pdf_paths))

    console.print(
        f"[green]✓[/] Indexed [bold]{len(chunks)}[/] chunks from "
        f"[bold]{len(pipeline.documents)}[/] documents → [dim]{args.out_dir}[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
