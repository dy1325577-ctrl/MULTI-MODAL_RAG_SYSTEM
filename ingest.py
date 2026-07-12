"""Run the full indexing pipeline on a PDF: parse -> chunk -> summarize -> index.

Usage:
    python ingest.py data/source.pdf
    python ingest.py data/source.pdf --force   # ignore caches, rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.index import build_index
from src.ingest import parse_pdf, summarize


def run(pdf_path, force: bool = False) -> None:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    print(f"[1/3] Parsing + chunking {pdf_path.name} (isolated process; first run is slow)...")
    doc = parse_pdf.parse(pdf_path, force=force)
    print(f"      chunks: {len(doc.chunks)} | tables: {len(doc.tables)} | images: {len(doc.images)}")

    print("[2/3] Summarizing tables & images with the VLM (slow, cached)...")
    summarize.summarize(doc.tables, doc.images, force=force)

    print("[3/3] Building indexes + docstore...")
    build_index.build(doc.chunks, doc.tables, doc.images)

    print(f"Done. Indexed {len(doc.chunks)} chunks, "
          f"{len(doc.tables)} tables, {len(doc.images)} images.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index a PDF into the multimodal RAG store.")
    ap.add_argument("pdf", nargs="?", default="data/ATTENTION.pdf",
                    help="Path to the PDF to index (default: data/ATTENTION.pdf)")
    ap.add_argument("--force", action="store_true", help="Ignore caches and rebuild")
    args = ap.parse_args()
    run(args.pdf, force=args.force)
