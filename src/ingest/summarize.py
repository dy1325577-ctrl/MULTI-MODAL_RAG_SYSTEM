"""Phase 5 — VLM summaries of tables and images (the trick that makes the text
index searchable for visual content).

This is the slowest indexing step, so results are cached to
``data/summaries/summaries.json`` keyed by ``doc_id`` and saved after *every*
item — a crash or OOM mid-run never loses completed work.
"""
from __future__ import annotations

import json

from PIL import Image

import config
from src.models import vlm

IMAGE_PROMPT = (
    "Describe this image in detail for a search index. Cover what it depicts, "
    "any text, labels, axis names, or numbers that are visible, and what "
    "question it could help answer. Be specific but concise."
)
TABLE_PROMPT = (
    "Summarize this table for a search index. State what it measures, what its "
    "columns are, and the key values or trends. Here is the table as HTML:\n\n{html}"
)


def _cache_file():
    return config.SUMMARIES_DIR / "summaries.json"


def _load_cache() -> dict:
    path = _cache_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    _cache_file().write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarize(tables: list[dict], images: list[dict], force: bool = False) -> None:
    """Attach a ``summary`` field to each table/image dict (cached by doc_id)."""
    cache = {} if force else _load_cache()

    for t in tables:
        cached = cache.get(t["doc_id"])
        if cached is not None:
            t["summary"] = cached
            continue
        prompt = TABLE_PROMPT.format(html=t["html"][:4000])
        summary = vlm.chat(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=config.SUMMARY_MAX_NEW_TOKENS,
        )
        t["summary"] = summary
        cache[t["doc_id"]] = summary
        _save_cache(cache)

    for im in images:
        cached = cache.get(im["doc_id"])
        if cached is not None:
            im["summary"] = cached
            continue
        image = Image.open(im["image_path"]).convert("RGB")
        summary = vlm.ask_about_image(
            image, IMAGE_PROMPT, max_new_tokens=config.SUMMARY_MAX_NEW_TOKENS
        )
        im["summary"] = summary
        cache[im["doc_id"]] = summary
        _save_cache(cache)
