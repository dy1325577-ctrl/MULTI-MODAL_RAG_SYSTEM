"""Milestone M1 smoke test.

Run from the project root:
    python scripts/check_env.py

Verifies CUDA is visible, loads the VLM in 4-bit, has it describe a generated
test image, and prints peak GPU memory (expect roughly 4-5 GB).
"""
from __future__ import annotations

import os
import sys

# allow running as `python scripts/check_env.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        sys.exit("No CUDA GPU visible — fix this before going further.")

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 128), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 30, 210, 100], outline="black", width=3)
    draw.text((70, 55), "HELLO", fill="black")

    from src.models import vlm

    print("\nLoading VLM in 4-bit (first run downloads ~4 GB)...")
    reply = vlm.ask_about_image(img, "What text and shape do you see?", max_new_tokens=64)
    print("VLM says:", reply)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\nPeak GPU memory: {peak:.2f} GB")
    print("If that's under ~5.5 GB you have headroom. M1 passed.")


if __name__ == "__main__":
    main()
