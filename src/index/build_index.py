"""Phase 6 — build the two Chroma indexes and the unified docstore.

Index A (``text_index``, bge)  : text chunks + table summaries + image summaries
Index B (``image_index``, CLIP): raw PDF image vectors

The docstore (``data/docstore.json``) maps every ``doc_id`` to its *raw* payload
so retrieval can hydrate real tables/images regardless of which index hit.
Metadata lives on the Chroma records and in the docstore — never embedded.
"""
from __future__ import annotations

import json

from PIL import Image

import config
from src.models import embedders


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _fresh_collection(client, name):
    """Drop and recreate a collection so re-indexing never duplicates records."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def build(chunks: list[dict], tables: list[dict], images: list[dict]) -> None:
    client = _client()
    text_col = _fresh_collection(client, config.TEXT_COLLECTION)
    image_col = _fresh_collection(client, config.IMAGE_COLLECTION)

    docstore: dict[str, dict] = {}

    # ---- Index A: everything searchable-as-text -------------------------
    ids, docs, metas = [], [], []

    for c in chunks:
        ids.append(c["doc_id"])
        docs.append(c["text"])
        metas.append({"type": "text", "page": c["page"], "source": c.get("source", "")})
        docstore[c["doc_id"]] = {"type": "text", "page": c["page"], "text": c["text"]}

    for t in tables:
        ids.append(t["doc_id"])
        docs.append(t["summary"])
        metas.append({"type": "table", "page": t["page"], "source": t.get("source", "")})
        docstore[t["doc_id"]] = {
            "type": "table", "page": t["page"],
            "summary": t["summary"], "html": t["html"],
        }

    for im in images:
        ids.append(im["doc_id"])
        docs.append(im["summary"])
        metas.append({"type": "image", "page": im["page"], "source": im.get("source", "")})
        docstore[im["doc_id"]] = {
            "type": "image", "page": im["page"],
            "summary": im["summary"], "image_path": im["image_path"],
        }

    if ids:
        text_col.add(
            ids=ids,
            embeddings=embedders.embed_documents(docs),
            documents=docs,
            metadatas=metas,
        )

    # ---- Index B: raw image pixels via CLIP -----------------------------
    if images:
        img_ids, img_embs, img_docs, img_metas = [], [], [], []
        for im in images:
            image = Image.open(im["image_path"]).convert("RGB")
            img_ids.append(im["doc_id"])
            img_embs.append(embedders.clip_embed_image(image))
            img_docs.append(im["summary"])
            img_metas.append({
                "type": "image", "page": im["page"], "image_path": im["image_path"],
            })
        image_col.add(
            ids=img_ids, embeddings=img_embs, documents=img_docs, metadatas=img_metas
        )

    config.DOCSTORE_PATH.write_text(
        json.dumps(docstore, ensure_ascii=False, indent=2), encoding="utf-8"
    )
