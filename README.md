# Multimodal RAG (local, Hugging Face stack)

Ask questions about a PDF — including tables and images — using **text + an optional image** as your query. Runs entirely on your own machine (RTX 3050, 6 GB). See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the design and [ARCHITECTURE.html](ARCHITECTURE.html) for the diagram.

## How it works (one paragraph)

A PDF is parsed with `unstructured` into text, tables, and images. Text is chunked; tables and images are summarized by a local vision model. Everything searchable goes into **two Chroma indexes** — a text space (`bge-base`) and a visual space (`CLIP`). At query time your text and image fan out across **four retrieval paths**, results are fused with reciprocal rank fusion, the raw tables/images are pulled back, and **Qwen2.5-VL-3B (4-bit)** writes a grounded answer. The CLIP image path is what lets a *similar* (not identical) query image still find the right PDF image.

## Setup

**1. System deps + environment (conda):**
```bash
conda create -n mmrag python=3.11 -y
conda activate mmrag
conda install -c conda-forge poppler tesseract -y   # unstructured needs these
```

**2. PyTorch (CUDA build FIRST), then the rest:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Hugging Face token:**
```bash
copy .env.example .env      # then paste your token into .env
huggingface-cli login       # optional; or rely on HF_TOKEN in .env
```

**4. Pre-download the models (optional but avoids a slow first run):**
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download BAAI/bge-base-en-v1.5
huggingface-cli download openai/clip-vit-base-patch32
```

## Verify the GPU + model load (do this first)

```bash
python scripts/check_env.py
```
Confirms CUDA is visible, loads the VLM in 4-bit, describes a test image, and prints peak GPU memory (should be ~4–5 GB).

## Use it

**Index a PDF** (parse → summarize → build indexes; cached, so it's slow only once):
```bash
python ingest.py data/source.pdf
```

**Ask from the CLI:**
```bash
python query.py "What does the revenue chart show?"
python query.py "Explain this diagram" --image path/to/similar_image.png
```

**Chat dashboard:**
```bash
streamlit run app.py
```
Upload a PDF in the sidebar to index it, then chat — attach an image to any question.

## Project layout

```
config.py            all paths, model ids, tunable knobs
ingest.py            CLI: run the full indexing pipeline on a PDF
query.py             CLI: retrieve + answer a single question
app.py               Streamlit chat dashboard
scripts/check_env.py GPU + VLM smoke test (milestone M1)
src/
  models/vlm.py        Qwen2.5-VL 4-bit singleton (the only GPU model)
  models/embedders.py  bge + CLIP wrappers (CPU)
  ingest/parse_pdf.py  PDF -> text/table/image elements
  ingest/chunk.py      text -> passages
  ingest/summarize.py  VLM summaries of tables & images (cached)
  index/build_index.py two Chroma indexes + unified docstore
  retrieval/retriever.py  4-path retrieval + RRF fusion + hydration
  generation/answer.py    multimodal prompt assembly + generation
```

## Notes / gotchas

- **Embedders run on CPU by design** — that is what keeps the VLM inside 6 GB VRAM. Don't move them to GPU.
- **`flash-attn` is not used** (unavailable on Windows); the model loads with `attn_implementation="sdpa"`.
- Indexing caches every stage under `data/`. Re-run with `--force` to rebuild from scratch.
- First `hi_res` parse downloads a ~1 GB layout model and is slow — normal, one-time.
