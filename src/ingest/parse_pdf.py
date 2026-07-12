"""Phase 3 — parse a PDF into text-chunk / table / image records.

The heavy lifting (unstructured `hi_res` + chunking) runs in a SEPARATE process
(`parse_worker`) that writes a plain-JSON cache. This wrapper just triggers that
worker on a cache miss and loads the JSON — so `unstructured`/`onnxruntime` are
never imported into the main pipeline, which otherwise segfaults on Windows when
they coexist with torch + bge. See parse_worker.py for the full explanation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class ParsedDoc:
    source: str = ""
    chunks: list = field(default_factory=list)   # plain dicts, ready to embed
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


def _cache_path(pdf_path: Path) -> Path:
    return config.PARSED_DIR / f"{pdf_path.stem}.json"


def parse(pdf_path, force: bool = False) -> ParsedDoc:
    pdf_path = Path(pdf_path)
    cache = _cache_path(pdf_path)

    if force or not cache.exists():
        # run parsing in an isolated process (must not share with the torch stack)
        subprocess.run(
            [sys.executable, "-m", "src.ingest.parse_worker", str(pdf_path)],
            check=True,
            cwd=str(config.ROOT),
        )

    data = json.loads(cache.read_text(encoding="utf-8"))
    return ParsedDoc(
        source=data.get("source", pdf_path.name),
        chunks=data.get("chunks", []),
        tables=data.get("tables", []),
        images=data.get("images", []),
    )
