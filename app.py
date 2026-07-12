"""Streamlit chat dashboard for the multimodal RAG system.

Run from the project root:
    streamlit run app.py

The VLM + embedders load once per server process via ``@st.cache_resource`` —
on 6 GB VRAM this is mandatory; a second load would OOM.
"""
from __future__ import annotations

import streamlit as st
from PIL import Image

import config
from src.generation import answer
from src.index import build_index
from src.ingest import parse_pdf, summarize
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="Multimodal RAG", page_icon="🗺️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    """Load the retriever (and, lazily, the models) once per server process."""
    return Retriever()


def ingest_pdf(uploaded) -> tuple[int, int, int]:
    pdf_path = config.DATA_DIR / uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())
    doc = parse_pdf.parse(pdf_path)
    summarize.summarize(doc.tables, doc.images)
    build_index.build(doc.chunks, doc.tables, doc.images)
    return len(doc.chunks), len(doc.tables), len(doc.images)


# ----------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Document")
    pdf = st.file_uploader("Upload a PDF to index", type=["pdf"])
    if pdf is not None and st.button("Index this PDF", use_container_width=True):
        with st.status("Indexing… first run parses the PDF and loads the model.",
                       expanded=True) as status:
            n_c, n_t, n_i = ingest_pdf(pdf)
            status.update(
                label=f"Indexed {n_c} chunks, {n_t} tables, {n_i} images.",
                state="complete",
            )
        get_retriever.clear()   # force the retriever to pick up the new index

    st.divider()
    k = st.slider("Passages to retrieve (k)", 2, 12, config.FINAL_K)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------- header ----
st.title("Multimodal RAG")
st.caption("Ask about your PDF with text and an optional image.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("image") is not None:
            st.image(m["image"], width=220)
        st.markdown(m["content"])

# ------------------------------------------------------------- inputs ----
query_image_file = st.file_uploader(
    "Attach an image to your next question (optional)",
    type=["png", "jpg", "jpeg"], key="imgq",
)
prompt = st.chat_input("Ask a question…")

if prompt:
    query_image = (
        Image.open(query_image_file).convert("RGB") if query_image_file else None
    )

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "image": query_image}
    )
    with st.chat_message("user"):
        if query_image is not None:
            st.image(query_image, width=220)
        st.markdown(prompt)

    try:
        with st.spinner("Loading models (the first time can take a minute)…"):
            retriever = get_retriever()
    except Exception:
        st.error("No index found yet — upload and index a PDF from the sidebar first.")
        st.stop()

    with st.spinner("Searching the document…"):
        retrieved = retriever.retrieve(text_query=prompt, image_query=query_image, k=k)

    with st.chat_message("assistant"):
        with st.spinner("Generating the answer… (the first question also loads "
                        "the vision model, ~1–2 min)"):
            answer_text = st.write_stream(
                answer.stream_answer(prompt, query_image, retrieved)
            )

        with st.expander("📎 Retrieved context"):
            for r in retrieved:
                st.markdown(f"**{r['type'].upper()} — page {r['page']}**")
                if r["type"] == "text":
                    st.write(r["text"])
                elif r["type"] == "table":
                    st.markdown(r["html"], unsafe_allow_html=True)
                elif r["type"] == "image":
                    st.image(r["image_path"], width=320)
                    if r.get("summary"):
                        st.caption(r["summary"])
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "image": None}
    )
