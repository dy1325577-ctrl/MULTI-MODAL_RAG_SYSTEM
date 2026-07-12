"""Embedding models — both pinned to CPU so the GPU stays free for the VLM.

- ``bge-small`` (sentence-transformers) embeds text: chunks, summaries, queries.
- ``CLIP`` embeds images and short text into a *shared* space, which is what
  makes "find the PDF image that looks like this one" work.

All vectors are L2-normalized so Chroma's cosine distances are meaningful.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

import config


@lru_cache(maxsize=1)
def _bge():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.TEXT_EMBED_MODEL_ID, device=config.EMBED_DEVICE)


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(config.CLIP_MODEL_ID, token=config.HF_TOKEN)
    model.to(config.EMBED_DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL_ID, token=config.HF_TOKEN)
    return model, proc


# ------------------------------------------------------------------ bge ----
def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed passages/summaries for indexing (no query instruction)."""
    vecs = _bge().encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query (with the bge retrieval instruction prepended)."""
    q = config.BGE_QUERY_INSTRUCTION + text
    vec = _bge().encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.tolist()


# ----------------------------------------------------------------- CLIP ----
def _l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


def clip_embed_image(image) -> list[float]:
    import torch

    model, proc = _clip()
    inputs = proc(images=image, return_tensors="pt").to(config.EMBED_DEVICE)
    with torch.no_grad():
        # explicit projection path — robust across transformers versions, where
        # get_image_features may return a wrapper object instead of a bare tensor
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = model.visual_projection(vision_out.pooler_output)
    return _l2(feats.cpu().numpy())[0].tolist()


def clip_embed_text(text: str) -> list[float]:
    import torch

    model, proc = _clip()
    inputs = proc(
        text=[text], return_tensors="pt", padding=True, truncation=True
    ).to(config.EMBED_DEVICE)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        feats = model.text_projection(text_out.pooler_output)
    return _l2(feats.cpu().numpy())[0].tolist()
