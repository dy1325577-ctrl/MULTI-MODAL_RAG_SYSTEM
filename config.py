"""Central configuration for the multimodal RAG system.

Every path, model id, and tunable knob lives here so the rest of the code
never hard-codes them. Importing this module also creates the data folders.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"          # extracted PDF images
PARSED_DIR = DATA_DIR / "parsed"          # cached unstructured output
SUMMARIES_DIR = DATA_DIR / "summaries"    # cached VLM summaries (JSON)
CHROMA_DIR = DATA_DIR / "chroma"          # persistent vector DB
DOCSTORE_PATH = DATA_DIR / "docstore.json"

for _d in (DATA_DIR, IMAGES_DIR, PARSED_DIR, SUMMARIES_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- models ----
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

HF_TOKEN = os.getenv("HF_TOKEN") or None   # optional; avoids Hub rate limits

# ----------------------------------------------------- VLM / generation ----
# max_pixels caps how many image tokens the processor emits -> protects VRAM.
VLM_MIN_PIXELS = 256 * 28 * 28
VLM_MAX_PIXELS = 768 * 28 * 28
SUMMARY_MAX_NEW_TOKENS = 320
ANSWER_MAX_NEW_TOKENS = 512    # shorter = faster to finish on a 6 GB GPU (raise for longer answers)
CAPTION_MAX_NEW_TOKENS = 96
MAX_IMAGES_IN_PROMPT = 3    # includes the user's query image

# ------------------------------------------------------------- chunking ----
CHUNK_MAX_CHARS = 1500
CHUNK_COMBINE_UNDER = 300
CHUNK_OVERLAP = 150

# ------------------------------------------------------------ retrieval ----
TOP_N_BGE_TEXT = 8      # path 1: query text -> bge      -> text index
TOP_N_CLIP_TEXT = 4     # path 2: query text -> CLIP txt  -> image index
TOP_N_CLIP_IMAGE = 6    # path 3: query image -> CLIP img -> image index
TOP_N_CAPTION = 4       # path 4: query image -> caption -> bge -> text index
RRF_K = 60              # reciprocal-rank-fusion constant
CLIP_SIM_FLOOR = 0.20   # drop image matches below this cosine similarity
FINAL_K = 6             # results kept after fusion

# --------------------------------------------------- Chroma collections ----
TEXT_COLLECTION = "text_index"
IMAGE_COLLECTION = "image_index"

# ------------------------------------------------------------ embedding ----
# bge-*-en-v1.5 retrieves better when the *query* carries this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBED_DEVICE = "cpu"    # embedders stay on CPU so the VLM owns the GPU
