"""
Download the Open RAG Benchmark dataset (Vectara/open_ragbench) and arXiv PDFs.

Steps:
  1. Pull metadata files (queries.json, qrels.json, pdf_urls.json) from HF Hub.
  2. Download up to --limit arXiv PDFs into data/papers/.
  3. Write data/qrels_filtered.json — qrels narrowed to downloaded papers only.

Usage:
    # Download metadata + 50 PDFs
    python scripts/download_dataset.py --limit 50

    # Metadata only (no PDFs)
    python scripts/download_dataset.py --no-pdfs

    # Force re-download even if files exist
    python scripts/download_dataset.py --limit 50 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

console = Console()

# ---------------------------------------------------------------------------
# HuggingFace dataset file locations
# ---------------------------------------------------------------------------

_HF_BASE = (
    "https://huggingface.co/datasets/vectara/open_ragbench"
    "/resolve/main/pdf/arxiv"
)
_DATASET_FILES = {
    "pdf_urls.json": f"{_HF_BASE}/pdf_urls.json",
    "queries.json":  f"{_HF_BASE}/queries.json",
    "qrels.json":    f"{_HF_BASE}/qrels.json",
    "answers.json":  f"{_HF_BASE}/answers.json",
}

_DATA_DIR   = Path("data")
_PAPERS_DIR = _DATA_DIR / "papers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download `url` to `dest`. Returns True if downloaded, False if skipped."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _beir_to_evaluator_qrels(
    queries: dict,
    qrels_raw: dict,
    paper_ids: set[str],
) -> dict[str, dict]:
    """
    Convert Open RAG Benchmark qrels + queries into the evaluator's format.

    qrels.json format:
        {query_id: {"doc_id": str, "section_id": int}}

    queries.json format:
        {query_id: {"query": str, ...}}

    Evaluator format:
        {query_id: {"query": str, "relevant_doc_ids": [str]}}

    Filters to queries whose relevant doc is in `paper_ids`.
    """
    result: dict[str, dict] = {}

    for query_id, rel in qrels_raw.items():
        doc_id = rel.get("doc_id", "")
        if not doc_id or doc_id not in paper_ids:
            continue

        entry = queries.get(query_id, {})
        if isinstance(entry, dict):
            query_text = entry.get("query", entry.get("text", ""))
        else:
            query_text = str(entry)

        if not query_text:
            continue

        result[query_id] = {
            "query": query_text,
            "relevant_doc_ids": [doc_id],
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="download_dataset",
        description="Download Open RAG Benchmark metadata and arXiv PDFs.",
    )
    p.add_argument("--limit", type=int, default=50,
                   help="Number of PDFs to download (default: 50)")
    p.add_argument("--no-pdfs", action="store_true",
                   help="Download metadata only; skip PDF download")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between PDF downloads — be polite to arXiv (default: 1.0)")
    p.add_argument("--force", action="store_true",
                   help="Re-download files even if they already exist")
    p.add_argument("--data-dir", type=Path, default=_DATA_DIR,
                   help="Root data directory (default: data/)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    data_dir   = args.data_dir
    papers_dir = data_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]Open RAG Benchmark — Download[/]\n\n"
        f"  Metadata dest : [dim]{data_dir}[/]\n"
        f"  PDFs dest     : [dim]{papers_dir}[/]\n"
        f"  PDF limit     : [yellow]{args.limit}[/]\n"
        f"  Delay         : [dim]{args.delay}s[/]",
        title="[bold]P4 Dataset[/]", expand=False,
    ))

    # ------------------------------------------------------------------
    # Step 1: metadata files
    # ------------------------------------------------------------------

    console.print("\n[bold]Step 1/3[/] — Downloading metadata files")
    for filename, url in _DATASET_FILES.items():
        dest = data_dir / filename
        try:
            downloaded = _download_file(url, dest, force=args.force)
            status = "[green]downloaded[/]" if downloaded else "[dim]already exists[/]"
            console.print(f"  {filename}: {status}")
        except Exception as exc:
            console.print(f"  [red]FAILED[/] {filename}: {exc}")
            return 1

    # ------------------------------------------------------------------
    # Step 2: PDF download
    # ------------------------------------------------------------------

    if not args.no_pdfs:
        pdf_urls_path = data_dir / "pdf_urls.json"
        if not pdf_urls_path.exists():
            console.print("[red]Error:[/] pdf_urls.json not found — metadata download failed?")
            return 1

        pdf_urls: dict[str, str] = _load_json(pdf_urls_path)
        items = list(pdf_urls.items())[: args.limit]

        console.print(f"\n[bold]Step 2/3[/] — Downloading {len(items)} PDFs")

        failed: list[str] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("PDFs", total=len(items))
            for paper_id, url in items:
                dest = papers_dir / f"{paper_id}.pdf"
                progress.update(task, description=f"[cyan]{paper_id}[/]")
                try:
                    _download_file(url, dest, force=args.force)
                except Exception as exc:
                    failed.append(paper_id)
                    console.print(f"\n  [yellow]WARN[/] {paper_id}: {exc}")
                finally:
                    progress.advance(task)
                    time.sleep(args.delay)

        downloaded_pdfs = sorted(papers_dir.glob("*.pdf"))
        console.print(
            f"  [green]✓[/] {len(downloaded_pdfs)} PDFs in {papers_dir}"
            + (f"  ([yellow]{len(failed)} failed[/])" if failed else "")
        )
    else:
        downloaded_pdfs = sorted(papers_dir.glob("*.pdf"))
        console.print("\n[dim]Step 2/3 — PDF download skipped (--no-pdfs)[/]")

    # ------------------------------------------------------------------
    # Step 3: build filtered qrels
    # ------------------------------------------------------------------

    console.print("\n[bold]Step 3/3[/] — Building filtered qrels")

    queries_path = data_dir / "queries.json"
    qrels_path   = data_dir / "qrels.json"

    if not queries_path.exists() or not qrels_path.exists():
        console.print("[yellow]WARN:[/] queries.json or qrels.json missing — skipping qrels build")
        return 0

    paper_ids = {p.stem for p in downloaded_pdfs}
    queries   = _load_json(queries_path)
    qrels_raw = _load_json(qrels_path)

    filtered = _beir_to_evaluator_qrels(queries, qrels_raw, paper_ids)

    out_path = data_dir / "qrels_filtered.json"
    out_path.write_text(json.dumps(filtered, indent=2))

    console.print(
        f"  [green]✓[/] {len(filtered)} queries map to {len(paper_ids)} papers "
        f"→ [dim]{out_path}[/]"
    )

    if not filtered:
        console.print(
            "\n  [yellow]Warning:[/] 0 queries survived filtering.\n"
            "  The qrels doc IDs may not match PDF stems. Check the qrels format."
        )

    console.print(
        f"\n[bold green]Done.[/] Run the POC with:\n"
        f"  python scripts/evaluate.py data/papers/ data/qrels_filtered.json "
        f"--config config/experiments/baseline.yaml --limit 5\n"
        f"\n  Full 50-paper run:\n"
        f"  python scripts/evaluate.py data/papers/ data/qrels_filtered.json "
        f"--config config/experiments/baseline.yaml"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
