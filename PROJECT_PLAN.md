# Multimodal RAG System — Complete Project Plan (Hugging Face local stack)

A local, GPU-friendly **Multimodal Retrieval-Augmented Generation** system that:

- Ingests a PDF containing **text, images, and tables** (parsed with `unstructured`)
- Lets the user query with **text + image (+ tables)** together
- Retrieves relevant chunks even when the query image is only **similar** (not identical) to a PDF image
- Generates answers with a **local vision LLM loaded from Hugging Face via `transformers`** (no Ollama, no external inference server)
- Serves everything through a **chat dashboard**

---

## 0. Hardware Constraints & Model Choices (decide this first)

**Machine:** RTX 3050 laptop (6 GB VRAM), 16 GB RAM, Windows 11.
**Serving:** every model is downloaded once from the Hugging Face Hub (authenticated with your **HF token**) into the local HF cache, then loaded in-process with `transformers` / `sentence-transformers`.

6 GB VRAM is the hard limit. A 3B vision LLM in FP16 is ~7.5 GB — **it does not fit**. So the VLM must be loaded in **4-bit (bitsandbytes NF4)**, and the embedders stay on CPU:

| Component | Model (HF repo id) | Size (loaded) | Runs on | Why |
|---|---|---|---|---|
| Vision LLM (generation + image/table summaries) | `Qwen/Qwen2.5-VL-3B-Instruct` in **4-bit NF4** via `bitsandbytes` | ~3 GB weights + ~1–1.5 GB activations/KV | GPU | Best small VLM; handles images + tables; 4-bit fits the 6 GB card |
| Text embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) | ~440 MB | **CPU** | Strong English text retrieval, still fast on CPU, keeps VRAM for the VLM |
| Image/cross-modal embeddings | `openai/clip-vit-base-patch32` (or `google/siglip-base-patch16-224`) | ~600 MB | **CPU** | Shared image↔text space → enables *similar-image* retrieval |
| Vector DB | **Chroma** (two collections) | RAM/disk | CPU | Zero-setup, persistent, metadata filtering |
| PDF parser | `unstructured` (`hi_res` strategy) | — | CPU | Extracts text, images (base64), tables (HTML) with layout awareness |
| Dashboard | **Streamlit** | — | CPU | Fastest path to a chat UI with image upload |

**VRAM budget:** Qwen2.5-VL-3B NF4 (~3 GB) + vision tower activations + KV cache (~1–1.5 GB) + CUDA overhead (~0.6 GB) ≈ **4.5–5 GB** → fits, with little headroom. Rules that keep it fitting:
- Embedders pinned to CPU (`device="cpu"`), always.
- ≤ 2–3 images per generation call, resized via the processor's `max_pixels` cap (≈ 768×768 worth of pixels).
- `max_new_tokens` ≤ 1024; cap prompt context (~6–8 k tokens).
- Close other GPU users (browser hardware acceleration) during runs.

**Fallbacks / upgrades:**
- If even 4-bit is tight or slow → `Qwen/Qwen2-VL-2B-Instruct` (4-bit ≈ 2 GB) or `HuggingFaceTB/SmolVLM2-2.2B-Instruct`.
- If CLIP retrieval quality disappoints → `jinaai/jina-clip-v2` (~1.7 GB, on CPU): one unified space for long text *and* images; can replace both embedders.
- 7B VLMs are **out of reach** on 6 GB even at 4-bit once activations are counted — don't waste time trying.

---

## 1. The Core Architecture Problem (and its solution)

**Your concern is correct:** a single embedding space for "images + text + tables + metadata" doesn't work well, because:

1. CLIP-style text encoders truncate at ~77 tokens → terrible for embedding long text chunks.
2. Text-specialist embedders (BGE) can't embed images at all.
3. Tables lose all structure if you just embed their raw text.

**Solution: a dual-index, summary-linked design** (the "MultiVector + CLIP sidecar" pattern):

```
                          ┌────────────────────────────┐
        PDF ──unstructured──►  Elements                 │
                          │  • text blocks              │
                          │  • tables (HTML + text)     │
                          │  • images (base64 + files)  │
                          └───────┬────────────────────┘
                                  │
             ┌────────────────────┼──────────────────────┐
             ▼                    ▼                       ▼
        text chunks        table → VLM summary      image → VLM summary
             │                    │                       │
             │   (raw table HTML + raw image path kept in a DOCSTORE,
             │    linked by doc_id — summaries are what get embedded)
             ▼                    ▼                       ▼
   ┌──────────────────────────────────────────┐   ┌─────────────────────┐
   │  INDEX A — TEXT SPACE (bge-small)        │   │ INDEX B — VISUAL     │
   │  chunks + table summaries + img summaries│   │ SPACE (CLIP)         │
   │  Chroma collection: "text_index"         │   │ raw PDF images       │
   └──────────────────────────────────────────┘   │ Chroma: "image_index"│
                                                  └─────────────────────┘
```

**Query time (user sends image + text):**

```
 user text ──bge──────────────► search INDEX A ─┐
 user text ──CLIP text tower──► search INDEX B ─┤
 user image ──CLIP image tower► search INDEX B ─┤  ◄── this is what makes
 user image ──VLM caption──bge► search INDEX A ─┤      "similar-looking image"
                                                │      queries work
                                                ▼
                              Reciprocal Rank Fusion (RRF)
                                                │
                              resolve doc_ids → DOCSTORE
                        (fetch RAW table HTML + RAW image files)
                                                ▼
                     Prompt = user text + user image + retrieved
                              text chunks + retrieved images + tables
                                                ▼
                        Qwen2.5-VL-3B (transformers, 4-bit) → answer
```

**Why this satisfies your "similar image" requirement:** the query image is embedded with CLIP's *image tower* and matched against CLIP embeddings of the PDF's images. CLIP similarity is semantic, not pixel-exact — a similar-looking chart/diagram/photo lands near the original in embedding space. The VLM caption path adds a second chance via text.

**Why metadata stops being a problem:** metadata (page number, element type, source file, doc_id) is stored as Chroma metadata fields — it is **never embedded**. Raw payloads (table HTML, image files) live in a simple docstore (folder + JSON map) keyed by `doc_id`. Embeddings only ever carry *retrievable meaning*; metadata carries *provenance and linkage*.

---

## 2. Phase-by-Phase Plan

### Phase 0 — System prerequisites (Windows)

1. Install **Miniconda** (if not present).
2. Verify GPU: `nvidia-smi` → confirm driver + CUDA runtime visible.
3. Create a **Hugging Face account + access token** (Settings → Access Tokens → new "read" token). Qwen/BGE/CLIP aren't gated, but the token avoids rate limits and unlocks gated models later.
4. Install `unstructured`'s system deps:
   - **Poppler** (PDF rendering): download poppler-windows release, add `bin/` to PATH.
   - **Tesseract OCR** (scanned/embedded-image text): installer from UB-Mannheim build, add to PATH.
5. Sanity checks:
   ```
   nvidia-smi
   pdftoppm -h
   tesseract --version
   ```

### Phase 1 — Conda environment + model downloads

```bash
conda create -n mmrag python=3.11 -y
conda activate mmrag

# System binaries unstructured needs (no manual PATH editing on Windows)
conda install -c conda-forge poppler tesseract -y

# PyTorch with CUDA (required for the 4-bit VLM on GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# HF model loading + 4-bit quantization
pip install transformers accelerate bitsandbytes huggingface_hub
pip install qwen-vl-utils          # Qwen2.5-VL image preprocessing helper

# Parsing
pip install "unstructured[pdf]" pillow

# Embeddings + retrieval
pip install sentence-transformers chromadb

# Dashboard + utilities
pip install streamlit pydantic python-dotenv tqdm
```

Authenticate and pre-download all models once (they land in the HF cache, `C:\Users\<you>\.cache\huggingface`):

```bash
huggingface-cli login          # paste your HF token (or set HF_TOKEN in .env)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download BAAI/bge-base-en-v1.5
huggingface-cli download openai/clip-vit-base-patch32
```

Freeze it: `pip freeze > requirements.txt`. Put `HF_TOKEN=hf_...` in `.env` (and `.env` in `.gitignore`).

> ⚠️ **Windows notes:** `bitsandbytes` ships Windows wheels since v0.43 — plain `pip install` works, no compilation. `flash-attn` is NOT installable on Windows; load the model with `attn_implementation="sdpa"` (PyTorch's built-in attention) instead.

### Phase 2 — Model loading layer (the piece an inference server would normally do)

`src/models/vlm.py` — one module owns the VLM as a **singleton**: loaded once, reused for summarization, captioning, and generation. Loading it twice = instant CUDA OOM on 6 GB.

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="cuda:0",
    attn_implementation="sdpa",          # flash-attn unavailable on Windows
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28,            # caps image tokens → protects VRAM
)

def chat(messages, max_new_tokens=512) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]
```

`src/models/embedders.py` — both embedders, **pinned to CPU**:

```python
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

text_embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # stays on CPU
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# expose: embed_text(str)->vec, embed_image(PIL.Image)->vec, clip_embed_text(str)->vec
# L2-normalize every vector so Chroma cosine distances are meaningful.
```

**Smoke test before building anything else:** load the VLM, have it describe one test image, and watch `nvidia-smi` — usage should sit around 4–5 GB. This is milestone M1.

### Phase 3 — Ingestion (PDF → elements)

`src/ingest/parse_pdf.py`

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="data/source.pdf",
    strategy="hi_res",                     # layout-aware; needed for tables/images
    infer_table_structure=True,            # tables come back with .metadata.text_as_html
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=True,   # images as base64 in metadata
)
```

Post-process into three buckets:
- **Text elements** → keep as-is for chunking.
- **Table elements** → save `text_as_html` (structure preserved) + plain text.
- **Image elements** → decode base64, save to `data/images/{doc_id}.png`.

Every element gets a generated `doc_id` (uuid) and metadata: `{source, page_number, element_type}`.

> ⚠️ First `hi_res` run downloads a YOLOX layout model (~1 GB) and is slow (minutes for a large PDF). One-time indexing cost — cache parsed output to `data/parsed/elements.pkl` so you never re-parse unnecessarily.

### Phase 4 — Chunking

- Use `unstructured.chunking.title.chunk_by_title` on text elements (`max_characters≈1500`, `combine_text_under_n_chars≈300`, `overlap≈150`).
- **Never chunk tables or images** — each is one atomic unit.
- Keep each chunk's page number(s) in metadata so images and text from the same page can be cross-linked (`page_number` is your join key for "context around this image").

### Phase 5 — Indexing-time summarization (the trick that makes Index A work)

`src/ingest/summarize.py` — for each **table** and **image**, one `vlm.chat(...)` call:

- **Image prompt:** *"Describe this image in detail for a search index: what it depicts, any text/labels/axis names, numbers, and what question it could answer."*
- **Table prompt:** pass the HTML — *"Summarize this table: what it measures, its columns, and key values/trends."*

Store: `{doc_id, summary, type, page}`. **Cache summaries to JSON on disk** — this is the slowest indexing step (~3–8 s per image on a 4-bit 3B model); never redo it. Process images **one at a time** — batching multiple images through the VLM blows the VRAM budget.

> This step is *why* a 3B VLM is enough overall: retrieval quality comes mostly from good summaries + fusion, not from raw LLM size.

### Phase 6 — Embedding + vector store

`src/index/build_index.py`

**Index A — `text_index` (Chroma), embedder = bge-small (CPU):**
| What gets embedded | What's in metadata |
|---|---|
| text chunk content | `doc_id, type="text", page, source` |
| table **summary** | `doc_id, type="table", page` → raw HTML in docstore |
| image **summary** | `doc_id, type="image", page` → image path in docstore |

**Index B — `image_index` (Chroma), embedder = CLIP image tower (CPU):**
| What gets embedded | Metadata |
|---|---|
| raw PDF image pixels | `doc_id, page, image_path` |

Pass **precomputed embeddings** to Chroma (`collection.add(embeddings=...)`) — don't register embedding functions with Chroma; your own model wrappers stay in control.

**Docstore:** `data/docstore.json` mapping `doc_id → {raw_html | image_path, page, type}`. (Upgrade to SQLite later if the PDF set grows.)

Chroma runs in persistent mode: `chromadb.PersistentClient(path="data/chroma")`, collections created with `metadata={"hnsw:space": "cosine"}`.

### Phase 7 — Retrieval + fusion

`src/retrieval/retriever.py`

```
retrieve(text_query, image_query=None, k=6):
  candidates = []
  1. bge(text_query)            → text_index.query(top 8)
  2. clip_text(text_query)      → image_index.query(top 4)
  3. if image_query:
       clip_image(image_query)  → image_index.query(top 6)      # similar-image path
       caption = vlm.chat(image_query, "describe briefly")
       bge(caption)             → text_index.query(top 4)
  4. RRF-fuse all ranked lists: score(d) = Σ 1/(60 + rank_i(d))
  5. dedupe by doc_id, take top k
  6. hydrate from docstore (raw tables/images) + pull same-page
     neighbor text for any retrieved image (page_number join)
```

Tuning knobs to expose in `config.py`: per-path `top_n`, RRF constant, a CLIP cosine-similarity floor (~0.2) to drop garbage image matches, and optional metadata filters (`type`, `page`).

### Phase 8 — Generation

`src/generation/answer.py` — build one multimodal `vlm.chat(...)` call:

- **System:** "Answer using ONLY the provided context (text, tables, images). Cite page numbers. Say so if the context is insufficient."
- **User message (Qwen chat format):** a content list mixing `{"type": "image", "image": path_or_pil}` and `{"type": "text", "text": ...}` items — the question + retrieved text chunks + retrieved table HTML + **user's query image first, then top 1–2 retrieved images** (≤3 images total; each image costs VRAM and context).
- `max_new_tokens=1024`. For streaming in the UI, use `transformers.TextIteratorStreamer` with `model.generate` running on a background thread.

### Phase 9 — Dashboard (Streamlit)

`app.py`

- Load the VLM and embedders with `@st.cache_resource` so they load **once per server process**, not on every Streamlit rerun — on 6 GB this is mandatory; a second load = CUDA OOM.
- `st.chat_message` / `st.chat_input` loop with session-state history.
- `st.file_uploader` (sidebar or above input) for the query **image**; show a thumbnail in the user's chat bubble.
- Answer bubble streams tokens (`st.write_stream` + `TextIteratorStreamer`); below it, an expander **"📎 Retrieved context"** showing: text chunks with page numbers, retrieved images (`st.image`), tables rendered from HTML (`st.markdown(html, unsafe_allow_html=True)`).
- Sidebar: PDF upload → triggers the ingest→summarize→index pipeline with a progress bar (cache by file hash so re-uploads are instant); `k` slider; "clear chat".
- Run: `streamlit run app.py`.

### Phase 10 — Evaluation & tuning

Build a tiny gold set (10–15 queries) before tuning anything:
- 5 pure-text questions
- 5 image-based ("here is a similar chart — what does the PDF say about this?") using *screenshots/redraws*, not exact crops, to test the similarity requirement
- 3–5 table questions

Measure: **retrieval hit@k** (is the right chunk in top-k?) separately from **answer quality** (manual 1–5). Debug retrieval before touching generation. Typical fixes in order: better summary prompts → RRF weights → swap CLIP for SigLIP/jina-clip-v2 → better generation prompt.

---

## 3. Project Structure

```
MULTI_MODAL/
├── PROJECT_PLAN.md
├── requirements.txt
├── .env                       # HF_TOKEN=hf_...  (gitignored)
├── config.py                  # model ids, paths, top_n, RRF constant, max_pixels
├── app.py                     # Streamlit dashboard
├── data/
│   ├── source.pdf
│   ├── images/                # extracted PDF images
│   ├── parsed/                # cached unstructured output
│   ├── summaries/             # cached VLM summaries (JSON)
│   ├── docstore.json
│   └── chroma/                # persistent vector DB
└── src/
    ├── models/
    │   ├── vlm.py             # Phase 2 — Qwen2.5-VL 4-bit singleton (load once!)
    │   └── embedders.py       # Phase 2 — bge + CLIP wrappers (CPU)
    ├── ingest/
    │   ├── parse_pdf.py       # Phase 3
    │   ├── chunk.py           # Phase 4
    │   └── summarize.py       # Phase 5
    ├── index/
    │   └── build_index.py     # Phase 6
    ├── retrieval/
    │   └── retriever.py       # Phase 7 (multi-path + RRF)
    └── generation/
        └── answer.py          # Phase 8 (VLM call + streaming)
```

---

## 4. Milestones (build in this order)

- [ ] **M0** — Env built; HF login done; models pre-downloaded; poppler/tesseract verified (`Phase 0–1`)
- [ ] **M1** — VLM loads in 4-bit and describes a test image; `nvidia-smi` shows ≈4–5 GB (`Phase 2`)
- [ ] **M2** — PDF parses; images land in `data/images/`, tables have HTML (`Phase 3–4`)
- [ ] **M3** — All images/tables have cached VLM summaries (`Phase 5`)
- [ ] **M4** — Both Chroma indexes built; a raw similarity query returns sane results (`Phase 6`)
- [ ] **M5** — CLI script: text-only question → fused retrieval → grounded answer (`Phase 7–8`)
- [ ] **M6** — Image+text query works end-to-end, including a *similar* (non-identical) image
- [ ] **M7** — Streamlit chat dashboard with retrieved-context display (`Phase 9`)
- [ ] **M8** — Gold-set eval done; knobs tuned (`Phase 10`)

---

## 5. Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **CUDA OOM** — the #1 risk of the raw-transformers route; nothing manages VRAM for you | 4-bit NF4 load; `max_pixels` cap on the processor; ≤3 images per call; `max_new_tokens≤1024`; embedders on CPU; `torch.cuda.empty_cache()` between ingest batches; singleton model + `@st.cache_resource` |
| `bitsandbytes` / CUDA mismatch on Windows | Use pip wheels (bnb ≥0.43) with the cu121 torch build; verify with `python -m bitsandbytes` |
| `flash-attn` won't install on Windows | Don't try — `attn_implementation="sdpa"` is built into PyTorch and works fine |
| `unstructured hi_res` painful on Windows (poppler/tesseract PATH issues) | Install both binaries first, verify from the *conda* shell; fallback `strategy="fast"` loses images/tables — only for debugging text flow |
| First run must download ~10 GB of weights | Pre-download everything with `huggingface-cli download` in Phase 1 so runtime never blocks on downloads |
| CLIP misses on diagrams/charts (trained on photos) | VLM-caption→text path covers it; upgrade to SigLIP or `jina-clip-v2` if hit-rate is low |
| 3B model hallucinates beyond context | Strict grounding system prompt + retrieved-context expander in the UI so you can verify sources |
| Indexing a big PDF is slow (summaries) | Aggressive caching at every stage (parse → summaries → index); only ever pay the cost once per PDF |
| Long chats blow the context budget | Keep only last ~4 turns in the prompt; retrieval context is per-turn, not accumulated |
