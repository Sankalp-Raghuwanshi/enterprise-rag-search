# Enterprise Knowledge Search (RAG)

A miniature enterprise search platform: upload documents, ask natural
language questions, get grounded answers with citations.

## Phase 1 (current)

- Upload PDF / DOCX
- Extract → chunk → embed (local `all-MiniLM-L6-v2`) → store in Chroma
- Vector-only retrieval → LLM answer generation with citations
- Streamlit UI

**Not yet included** (see project roadmap): BM25/hybrid retrieval,
reranking, PPTX/CSV/Markdown support, evaluation harness, FastAPI +
cloud deployment. These come in later phases.

## Setup

```bash
cd enterprise-rag-search
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GROK_API_KEY
```

## Run

```bash
streamlit run app.py
```

## Manual testing (no UI)

```bash
cd src
python pipeline.py ingest ../data/uploads/some_file.pdf
python pipeline.py ask "What is this document about?"
```

You can also test each stage individually:

```bash
python src/extract.py path/to/file.pdf      # check text extraction
python src/chunker.py path/to/file.pdf      # check chunking
python src/embed_store.py path/to/file.pdf "your test query"  # check retrieval
```

## Project structure

```
enterprise-rag-search/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── extract.py          # PDF/DOCX text extraction
│   ├── chunker.py          # text splitting with overlap
│   ├── embed_store.py      # embeddings + Chroma vector store
│   ├── llm.py              # LLM API wrapper (Grok)
│   └── pipeline.py         # orchestration: ingest_file(), answer_query()
├── data/
│   ├── uploads/             # (gitignored) local file staging
│   └── vectorstore/         # (gitignored) persistent Chroma DB
└── tests/                   # evaluation harness (Phase 5)
```

## Notes on design decisions

- **Chunking**: recursive character splitting (paragraph → sentence →
  hard cut), 800 chars/chunk, 150 char overlap. No external chunking
  library dependency - keeps the logic visible and explainable.
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers - small,
  fast on CPU, no API cost or rate limits for a project at this scale.
- **Vector store**: Chroma, persistent local storage - zero infra setup
  needed for Phase 1, swappable later if needed.
- **Citations**: every chunk carries `source_file` + `page_number`
  metadata from ingestion through to the final answer.
