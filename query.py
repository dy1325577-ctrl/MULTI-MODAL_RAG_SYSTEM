"""Ask the indexed document a question from the CLI (text and/or image).

Usage:
    python query.py "What does the revenue chart show?"
    python query.py "Explain this diagram" --image similar.png
"""
from __future__ import annotations

import argparse
import sys

# Windows consoles default to cp1252, which can't print some Unicode found in PDF
# text (e.g. the "fi" ligature). Force UTF-8 so context and answers print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image

from src.generation import answer
from src.retrieval.retriever import Retriever


def run(question: str, image_path: str | None = None, k: int | None = None) -> None:
    image = Image.open(image_path).convert("RGB") if image_path else None

    retriever = Retriever()
    retrieved = retriever.retrieve(text_query=question, image_query=image, k=k)

    print("\nRetrieved context:")
    for r in retrieved:
        preview = (r.get("text") or r.get("summary") or "").replace("\n", " ")
        print(f"  [{r['type'].upper():5s} p{r['page']}] {preview[:100]}")

    print("\nAnswer:\n")
    for piece in answer.stream_answer(question, image, retrieved):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Query the multimodal RAG store.")
    ap.add_argument("question", help="Your question")
    ap.add_argument("--image", default=None, help="Optional path to a query image")
    ap.add_argument("-k", type=int, default=None, help="Number of passages to retrieve")
    args = ap.parse_args()
    run(args.question, args.image, args.k)
