# app.py — Streamlit UI (entry point)
# Run this file: streamlit run app.py

import streamlit as st
import hashlib

from config import EMBED_MODEL, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MEMORY_TURNS
from pdf_processor import index_pdf
from rag_pipeline import search_and_answer

# ── Page Setup ────────────────────────────────────────────────
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG PDF Assistant")
st.caption("Upload a PDF → Ask questions → Powered by Ollama (local)")

# ── Session State ─────────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages         = []
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
if "indexed_pdf_hash" not in st.session_state: st.session_state.indexed_pdf_hash = None
if "collection_name"  not in st.session_state: st.session_state.collection_name  = None

# ── Sidebar — PDF Upload ──────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        pdf_hash  = hashlib.md5(pdf_bytes).hexdigest()
        coll_name = f"rag_{pdf_hash[:12]}"

        if st.session_state.indexed_pdf_hash != pdf_hash:
            with st.spinner("Reading and indexing PDF…"):
                n = index_pdf(pdf_bytes, coll_name)
            st.session_state.indexed_pdf_hash = pdf_hash
            st.session_state.collection_name  = coll_name
            st.session_state.messages         = []
            st.session_state.chat_history     = []
            st.success(f"✅ Indexed **{n}** chunks from **{uploaded_file.name}**")
        else:
            st.info(f"✅ **{uploaded_file.name}** already indexed!")

    st.divider()
    st.markdown("**⚙️ Configuration**")
    st.markdown(f"- Embedding: `{EMBED_MODEL.split('/')[-1]}`")
    st.markdown(f"- LLM: `{OLLAMA_MODEL}`")
    st.markdown(f"- Chunk size: `{CHUNK_SIZE}` | Overlap: `{CHUNK_OVERLAP}`")
    st.markdown(f"- Top-K retrieval: `{TOP_K}`")
    st.markdown(f"- Memory turns: `{MEMORY_TURNS}`")
    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()

# ── Guard — No PDF yet ────────────────────────────────────────
if not st.session_state.collection_name:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# ── Prompt History Display ────────────────────────────────────
st.subheader("💬 Prompt History")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Source Chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i} — Page {src['page']}** · relevance score: `{src['score']}`")
                    st.caption(f"> {src['text']}…")
                    st.divider()

# ── Chat Input ────────────────────────────────────────────────
question = st.chat_input("Ask anything about your PDF…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer…"):
            answer, sources = search_and_answer(
                question,
                st.session_state.collection_name,
                st.session_state.chat_history,
            )
        st.markdown(answer)
        with st.expander("📚 View Source Chunks"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i} — Page {src['page']}** · relevance score: `{src['score']}`")
                st.caption(f"> {src['text']}…")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
