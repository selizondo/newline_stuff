#!/usr/bin/env python3
"""
Interactive QA REPL for the P4 RAG pipeline.

Usage:
    python scripts/serve.py -i data/indices/my_index
    python scripts/serve.py -i data/indices/my_index --retrieval hybrid --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from rag_common.chunkers import FixedSizeChunker
from src.embedders import SentenceTransformersEmbedder
from src.generator import generate_answer
from src.pipeline import RAGPipeline

console = Console()


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive QA REPL for P4 RAG pipeline")
    p.add_argument("-i", "--index-dir", type=Path, required=True,
                   help="Path to a saved FAISS index directory")
    p.add_argument("--model", "-m", type=str, default="",
                   help="LLM model for answer generation (default: from LLM_MODEL env)")
    p.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                   help="Embedding model (must match the index; default: all-MiniLM-L6-v2)")
    p.add_argument("--retrieval", type=str, default="dense",
                   choices=["dense", "bm25", "hybrid"],
                   help="Retrieval method (default: dense)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of chunks to retrieve (default: 5)")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Hybrid dense weight (default: 0.6; ignored unless --retrieval hybrid)")
    return p.parse_args(argv)


def _resolve_model(arg_model: str) -> str:
    if arg_model:
        return arg_model
    from llm_utils.config import get_settings
    return get_settings().generation_model


def main(argv=None) -> int:
    args = _parse_args(argv)
    model = _resolve_model(args.model)

    if not args.index_dir.exists():
        console.print(f"[red]Error:[/] index directory not found: {args.index_dir}")
        return 1

    console.print(Panel(
        f"[bold cyan]PaperSearch — Interactive QA[/]\n\n"
        f"  Index      : [dim]{args.index_dir}[/]\n"
        f"  Embedding  : [dim]{args.embedding_model}[/]\n"
        f"  Retrieval  : [dim]{args.retrieval}[/]"
        + (f"  (α={args.alpha})" if args.retrieval == "hybrid" else "")
        + f"  top-k={args.top_k}\n"
        f"  LLM model  : [dim]{model}[/]\n\n"
        "Type [bold]quit[/] or [bold]exit[/] to stop. [bold]clear[/] to reset screen.",
        title="[bold]P4 RAG[/]", expand=False,
    ))

    embedder = SentenceTransformersEmbedder(
        model_name=args.embedding_model,
        cache_dir=Path("data/embed_cache"),
    )
    # The chunker is irrelevant after index load; pass a dummy to satisfy the constructor.
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(512, 64),
        embedder=embedder,
        retrieval_method=args.retrieval,
        alpha=args.alpha,
    )

    console.print("Loading index...", end=" ")
    pipeline.load(args.index_dir)
    n_chunks = len(pipeline.chunks)
    n_docs   = len(pipeline.documents)
    console.print(f"[green]done[/] — {n_chunks:,} chunks from {n_docs} documents.")

    while True:
        try:
            query = console.input("\n[bold yellow]Question:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            console.print("Bye!")
            break
        if query.lower() == "clear":
            console.clear()
            continue

        with console.status("Retrieving…"):
            results, retrieval_s = pipeline.query_timed(query, top_k=args.top_k)

        with console.status("Generating answer…"):
            qa = generate_answer(query, results, model=model)

        console.print(Panel(
            Text(qa.answer),
            title=(
                f"[bold green]Answer[/]  "
                f"[dim]({retrieval_s*1000:.0f} ms retrieval, "
                f"{qa.generation_time_s*1000:.0f} ms generation)[/]"
            ),
        ))

        if qa.citations:
            console.print("[dim]Citations:[/]")
            for c in qa.citations:
                page = f", p.{c.page_number}" if c.page_number else ""
                snippet = c.text_snippet[:80].replace("\n", " ")
                console.print(f"  • [cyan]{c.source}{page}[/] — {snippet}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())
