"""Phase 8 — assemble the multimodal prompt and generate the grounded answer.

Image budget matters on 6 GB VRAM: the user's query image goes first, then at
most a couple of retrieved images, capped at ``MAX_IMAGES_IN_PROMPT`` total.
Text chunks and table HTML are passed as text context with page citations.
"""
from __future__ import annotations

from PIL import Image

import config
from src.models import vlm

SYSTEM = (
    "You are a helpful assistant answering questions about a specific document. "
    "Answer using ONLY the provided context (text passages, tables, and images). "
    "Cite page numbers in parentheses. If the context does not contain the answer, "
    "say so plainly rather than guessing."
)


def _build_messages(query_text, query_image, retrieved):
    content = []
    image_budget = config.MAX_IMAGES_IN_PROMPT

    # 1) the user's own query image first
    if query_image is not None and image_budget > 0:
        content.append({"type": "image", "image": query_image})
        content.append({"type": "text",
                        "text": "[Above: the image the user attached to their question.]"})
        image_budget -= 1

    # 2) text + table context, collecting retrieved images for later
    context_blocks = []
    retrieved_images = []
    for r in retrieved:
        if r["type"] == "text":
            context_blocks.append(f"(page {r['page']}) {r['text']}")
        elif r["type"] == "table":
            context_blocks.append(f"(page {r['page']}) TABLE:\n{r['html']}")
        elif r["type"] == "image":
            context_blocks.append(f"(page {r['page']}) IMAGE — {r.get('summary', '')}")
            retrieved_images.append(r)

    if context_blocks:
        content.append({"type": "text",
                        "text": "CONTEXT:\n\n" + "\n\n---\n\n".join(context_blocks)})

    # 3) attach the most relevant retrieved images, within budget
    for r in retrieved_images:
        if image_budget <= 0:
            break
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            continue
        content.append({"type": "image", "image": img})
        content.append({"type": "text",
                        "text": f"[Above: retrieved image from page {r['page']}.]"})
        image_budget -= 1

    # 4) finally, the question
    content.append({"type": "text", "text": f"QUESTION: {query_text}"})

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def generate_answer(query_text, query_image=None, retrieved=None) -> str:
    messages = _build_messages(query_text, query_image, retrieved or [])
    return vlm.chat(messages, max_new_tokens=config.ANSWER_MAX_NEW_TOKENS)


def stream_answer(query_text, query_image=None, retrieved=None):
    messages = _build_messages(query_text, query_image, retrieved or [])
    yield from vlm.stream(messages, max_new_tokens=config.ANSWER_MAX_NEW_TOKENS)
