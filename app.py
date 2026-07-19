"""
app.py
------
Minimal Streamlit UI for Phase 1: upload PDFs/DOCX, ask questions,
see grounded answers with source citations.
"""

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import ingest_file, answer_query  # noqa: E402

st.set_page_config(page_title="Enterprise Knowledge Search", layout="wide")
st.title("🔍 Enterprise Knowledge Search")
st.caption("Upload documents, then ask questions in plain English. "
           "Hybrid retrieval (vector + BM25) with cross-encoder reranking (PDF + DOCX).")

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# --- Upload section ---
st.subheader("1. Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.ingested_files:
            continue  # don't re-ingest the same file this session

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner(f"Processing {uploaded_file.name}..."):
            result = ingest_file(tmp_path, display_name=uploaded_file.name)

        st.session_state.ingested_files.append(uploaded_file.name)

        if result["status"] == "duplicate":
            st.info(f"'{uploaded_file.name}' was already ingested previously - skipped re-processing.")
        else:
            st.success(f"Ingested {uploaded_file.name} ({result['chunks_added']} chunks)")

        os.unlink(tmp_path)

if st.session_state.ingested_files:
    st.caption(f"Documents in this session: {', '.join(st.session_state.ingested_files)}")

# --- Query section ---
st.subheader("2. Ask a question")
query = st.text_input("e.g. 'Summarize the onboarding process' or 'Which document mentions Terraform?'")

if st.button("Search") and query:
    with st.spinner("Retrieving and generating answer..."):
        result = answer_query(query)

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### Sources")
        for s in result["sources"]:
            with st.expander(f"{s['source_file']} — page {s['page_number']}"):
                st.write(s["snippet"] + "...")
