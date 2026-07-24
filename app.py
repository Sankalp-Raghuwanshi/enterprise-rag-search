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

# Streamlit Cloud has no local .env file - secrets are set via its own
# secrets manager (st.secrets) instead. Everything downstream (llm.py,
# vision.py) reads GROQ_API_KEY via os.getenv(), so bridging st.secrets
# into os.environ here means those modules need zero changes to work
# in both environments (local .env for dev, Streamlit secrets for cloud).
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass  # falls through - local .env (via python-dotenv) handles it instead

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import ingest_file, answer_query  # noqa: E402
from agent import run_agent  # noqa: E402

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
            image_note = f", {result['images_added']} images described" if result.get("images_added") else ""
            st.success(f"Ingested {uploaded_file.name} ({result['chunks_added']} chunks{image_note})")

        os.unlink(tmp_path)

if st.session_state.ingested_files:
    st.caption(f"Documents in this session: {', '.join(st.session_state.ingested_files)}")

# --- Query section ---
st.subheader("2. Ask a question")
query = st.text_input("e.g. 'Summarize the onboarding process' or 'Which document mentions Terraform?'")
use_agent = st.checkbox(
    "Use agent mode (decides if retrieval is needed, decomposes complex questions)",
    value=False,
)

if st.button("Search") and query:
    if use_agent:
        with st.spinner("Agent is routing, decomposing, and retrieving..."):
            result = run_agent(query)

        with st.expander("🧠 Agent trace (what it decided to do)"):
            st.write(f"**Used retrieval:** {result['trace']['used_retrieval']}")
            st.write(f"**Sub-questions:** {result['trace']['sub_questions']}")
    else:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_query(query)

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### Sources")
        for s in result["sources"]:
            with st.expander(f"{s['source_file']} — page {s['page_number']}"):
                st.write(s["snippet"] + "...")
