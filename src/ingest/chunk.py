"""Phase 4 — chunk text elements into atomic passages.

Tables and images are *never* chunked; each stays a single unit (handled in
``build_index``). Only the narrative text stream is chunked here, using
``unstructured``'s title-aware chunker so passages break on section boundaries.
"""
from __future__ import annotations

import uuid

import config


def chunk_text_elements(text_elements) -> list[dict]:
    """Turn a list of unstructured text Elements into passage dicts."""
    if not text_elements:
        return []

    from unstructured.chunking.title import chunk_by_title

    chunks = chunk_by_title(
        text_elements,
        max_characters=config.CHUNK_MAX_CHARS,
        combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER,
        overlap=config.CHUNK_OVERLAP,
    )

    out: list[dict] = []
    for ch in chunks:
        text = (ch.text or "").strip()
        if not text:
            continue
        page = int(getattr(ch.metadata, "page_number", None) or 0)
        out.append({
            "doc_id": uuid.uuid4().hex,
            "type": "text",
            "page": page,
            "text": text,
            "source": getattr(ch.metadata, "filename", "") or "",
        })
    return out
