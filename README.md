# Enterprise Knowledge Search (RAG)

A miniature enterprise search platform: upload documents, ask natural
language questions in plain English, get grounded answers with citations.
Built as a hands-on exploration of retrieval-augmented generation (RAG),
hybrid search, and reranking — the same core techniques behind modern
"thinking search" products.

## What it does

- Upload PDF / DOCX documents
- Automatically extracts text, chunks it, embeds it, and indexes it for
  both semantic (vector) and keyword (BM25) search
- Ask questions like *"Summarize the onboarding process"* or *"Which
  document mentions Terraform?"*
- Retrieves relevant passages using **hybrid retrieval** (vector search +
  BM25, fused via Reciprocal Rank Fusion), then **reranks** them with a
  cross-encoder for precision
- Generates a grounded answer with inline citations `[Chunk N]`, mapped
  back to the real source file and page
- Deduplicates re-uploaded files by content hash

## Measured results

Built an evaluation harness (`tests/run_evaluation.py`) that measures
retrieval accuracy (hit rate @ top-5) across a 12-question test set:

| Method       | Accuracy |
|--------------|----------|
| Vector-only  | 91.7%    |
| BM25-only    | 100.0%   |
| **Hybrid**   | **100.0%** |

Vector search alone missed a plain factual question ("What is this
person's major and where do they study?") — hybrid retrieval recovered
it by combining semantic and keyword signals. Full methodology and
caveats are documented in `LEARNINGS.md` (Section 14) — this is a small,
illustrative evaluation on one document, not a large-scale benchmark.

## Architecture

```
                    ┌─────────────┐
   Upload doc  ───► │  extract.py │  (PDF/DOCX → text, page-level metadata)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ chunker.py  │  (recursive splitting, 800 char / 150 overlap)
                    └──────┬──────┘
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐      ┌──────────────────┐
     │  embed_store.py  │      │ keyword_index.py │
     │  (vector search) │      │  (BM25 search)   │
     └────────┬─────────┘      └────────┬─────────┘
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ hybrid_retrieval.py │  (Reciprocal Rank Fusion)
                └──────────┬──────────┘
                           ▼
                   ┌───────────────┐
                   │  reranker.py  │  (cross-encoder precision pass)
                   └───────┬───────┘
                           ▼
                     ┌───────────┐
                     │  llm.py   │  (grounded answer + citations, via Groq)
                     └─────┬─────┘
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐      ┌──────────────────┐
     │     app.py       │      │     api.py       │
     │ (Streamlit UI)   │      │  (FastAPI service)│
     └─────────────────┘      └──────────────────┘
```

`pipeline.py` orchestrates all of the above (`ingest_file()` /
`answer_query()`) and is used by both the Streamlit UI and the FastAPI
service — one pipeline, two interfaces.

## Setup

```bash
git clone <this-repo-url>
cd enterprise-rag-search
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY (from console.groq.com)
```

## Run — Streamlit UI

```bash
streamlit run app.py
```

## Run — FastAPI service

```bash
cd src
uvicorn api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Run — evaluation harness

```bash
cd tests
python run_evaluation.py ../data/uploads/<your_file>.pdf eval_dataset.json 5
```

Edit `eval_dataset.json` with your own test questions and verified page
numbers (see `LEARNINGS.md` Section 14 for how the questions were built).

## Manual testing (component by component)

```bash
python src/extract.py path/to/file.pdf              # extraction only
python src/chunker.py path/to/file.pdf               # chunking only
python src/embed_store.py path/to/file.pdf "query"   # vector search only
python src/keyword_index.py path/to/file.pdf "query" # BM25 only
python src/hybrid_retrieval.py path/to/file.pdf "query"  # vector vs BM25 vs hybrid, side by side
python src/reranker.py                                # reranker on fake data
python src/pipeline.py ingest path/to/file.pdf
python src/pipeline.py ask "your question"
```

## Deployment (Terraform / AWS)

`terraform/` contains infrastructure-as-code to provision an EC2 instance
running the FastAPI service (`terraform/main.tf`, `terraform/user_data.sh.tpl`).

**Design decision — EC2, not Lambda**: this project's dependencies
(PyTorch, sentence-transformers, embedding + reranker model weights) are
well beyond Lambda's deployment package limits, and the models benefit
from staying loaded in memory between requests rather than a cold start
per invocation. A small EC2 instance is the realistic, honest choice.

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars with your AWS key pair name and Groq API key
terraform init
terraform plan
terraform apply
# ... when done:
terraform destroy   # important - avoids ongoing AWS charges
```

**Status**: this configuration is written and reviewed but not
continuously deployed, to avoid ongoing AWS costs during development.

## Project structure

```
enterprise-rag-search/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
├── LEARNINGS.md              # concept-by-concept explanation of the whole project
├── README.md
├── src/
│   ├── extract.py            # PDF/DOCX text extraction with citation metadata
│   ├── chunker.py             # recursive text splitting with overlap
│   ├── embed_store.py         # embeddings (sentence-transformers) + Chroma vector store
│   ├── keyword_index.py       # BM25 keyword search
│   ├── hybrid_retrieval.py    # Reciprocal Rank Fusion of vector + BM25
│   ├── reranker.py            # cross-encoder reranking
│   ├── llm.py                 # Groq API wrapper, grounded prompting
│   ├── pipeline.py            # orchestration: ingest_file(), answer_query()
│   └── api.py                 # FastAPI service wrapping the pipeline
├── terraform/
│   ├── main.tf                 # EC2 + security group provisioning
│   ├── user_data.sh.tpl        # boot script: installs deps, runs API as systemd service
│   └── terraform.tfvars.example
├── tests/
│   ├── eval_dataset.json       # 12 test questions with verified expected sources
│   └── run_evaluation.py       # retrieval accuracy evaluation harness
└── data/                        # (gitignored) local file staging + persisted indexes
```

## Key design decisions

- **Chunking**: hand-written recursive splitter (paragraph → sentence →
  hard cut), 800 chars/chunk, 150 char overlap — no external chunking
  library, so the logic is fully explainable.
- **Embeddings**: `all-MiniLM-L6-v2` — small, fast on CPU, free, runs
  locally with no API cost or rate limits.
- **Hybrid retrieval**: Reciprocal Rank Fusion, not raw score averaging —
  vector distance and BM25 score are on incompatible scales, so fusion
  uses rank position instead of raw scores.
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` on a shortlist of
  ~15 hybrid candidates — "retrieve wide, rerank narrow," the standard
  pattern in real search systems.
- **Citations**: sources shown to the user are filtered to only the
  chunks the LLM actually cited (parsed from `[Chunk N]` markers), not
  everything retrieved.
- **Deduplication**: content-hash based, so re-uploading the same file
  (even under a different temp filename, as Streamlit does) is detected
  and skipped.
- **Deployment**: EC2 over Lambda, for the size/cold-start reasons above.

See `LEARNINGS.md` for a full concept-by-concept explanation of every
design decision and the reasoning behind it.
