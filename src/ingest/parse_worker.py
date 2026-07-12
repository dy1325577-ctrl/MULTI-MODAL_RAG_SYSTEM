"""Parsing + chunking worker — runs in its OWN process.

This exists solely to keep `unstructured`/`onnxruntime` out of the process that
later runs torch + bge + CLIP. On Windows those two native stacks share a
process badly and segfault during bge's batched CPU inference. So parsing runs
here, writes a plain-JSON cache (no unstructured objects), and exits — leaving
the main pipeline free of the conflicting libraries.

Run as:  python -m src.ingest.parse_worker <pdf_path>
"""
from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path

import config


def run(pdf_path) -> Path:
    pdf_path = Path(pdf_path)

    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )

    text_elements, tables, images = [], [], []
    for el in elements:
        category = getattr(el, "category", None) or el.__class__.__name__
        meta = el.metadata
        page = int(getattr(meta, "page_number", None) or 0)

        if category == "Table":
            tables.append({
                "doc_id": uuid.uuid4().hex, "type": "table", "page": page,
                "html": getattr(meta, "text_as_html", None) or (el.text or ""),
                "text": el.text or "", "source": pdf_path.name,
            })
        elif category == "Image":
            b64 = getattr(meta, "image_base64", None)
            if not b64:
                continue
            did = uuid.uuid4().hex
            img_path = config.IMAGES_DIR / f"{did}.png"
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(b64))
            images.append({
                "doc_id": did, "type": "image", "page": page,
                "image_path": str(img_path), "source": pdf_path.name,
            })
        else:
            text_elements.append(el)

    # chunk text here, while unstructured Element objects are available
    chunks = []
    if text_elements:
        for ch in chunk_by_title(
            text_elements,
            max_characters=config.CHUNK_MAX_CHARS,
            combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER,
            overlap=config.CHUNK_OVERLAP,
        ):
            text = (ch.text or "").strip()
            if not text:
                continue
            chunks.append({
                "doc_id": uuid.uuid4().hex, "type": "text",
                "page": int(getattr(ch.metadata, "page_number", None) or 0),
                "text": text, "source": getattr(ch.metadata, "filename", "") or "",
            })

    out = {"source": pdf_path.name, "chunks": chunks, "tables": tables, "images": images}
    cache = config.PARSED_DIR / f"{pdf_path.stem}.json"
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[worker] parsed: {len(chunks)} chunks, {len(tables)} tables, {len(images)} images", flush=True)
    return cache


if __name__ == "__main__":
    run(sys.argv[1])
