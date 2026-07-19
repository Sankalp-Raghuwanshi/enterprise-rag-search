"""
api.py
------
FastAPI wrapper around the RAG pipeline - this is what gets deployed to
AWS (see terraform/), separate from the Streamlit frontend. Splitting
frontend (Streamlit) from a real backend API is what makes this
deployable as an actual service, not just a local demo script.

Run locally:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs
(FastAPI auto-generates this - it's one of the reasons FastAPI is a
common choice for ML services).
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pipeline import ingest_file, answer_query

app = FastAPI(
    title="Enterprise Knowledge Search API",
    description="RAG-powered document search with hybrid retrieval and reranking.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health_check():
    """Basic liveness check - useful for load balancers / deployment checks."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index a PDF or DOCX file."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path, display_name=file.filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Ask a question over all ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_query(request.question, top_k=request.top_k)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
