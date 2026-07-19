"""
pipeline.py
-----------
Ties extract -> chunk -> retrieve (hybrid: vector + BM25) -> generate
into two simple functions:

    ingest_file(file_path, display_name=None)
    answer_query(query_text)

Phase 2 updates:
 - retrieval is now hybrid (vector + BM25 fused via RRF) instead of
   vector-only. Reranking comes in Phase 3.
 - ingestion is deduplicated by content hash (see _manifest functions
   below) so re-uploading the same file - even under a different temp
   path/name, as Streamlit does - doesn't create duplicate chunks.

Keep this file thin - it's the orchestration layer, not where logic lives.
"""

import hashlib
import json
from pathlib import Path

from extract import extract_document
from chunker import chunk_pages
from hybrid_retrieval import HybridRetriever
from llm import generate_answer

# Single shared retriever instance for the app's lifetime.
_store = HybridRetriever()

MANIFEST_PATH = Path("../data/ingested_files.json")


def _load_manifest() -> dict:
    """Manifest maps content_hash -> {display_name, chunk_count}."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _hash_file(file_path: str) -> str:
    """SHA-256 hash of the file's actual bytes - identifies content
    regardless of filename, so the same file uploaded twice (even under
    a different temp path) is recognized as a duplicate."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_file(file_path: str, display_name: str = None) -> dict:
    """
    Extract, chunk, embed, and index one file - skipping it if this exact
    content has already been ingested.

    display_name: the filename to show in citations (defaults to the
    file_path's own name). Pass this explicitly when file_path is a temp
    path that doesn't reflect the user's original filename (e.g. Streamlit
    uploads).

    Returns: {"status": "ingested" | "duplicate", "chunks_added": int,
              "display_name": str}
    """
    display_name = display_name or Path(file_path).name
    file_hash = _hash_file(file_path)
    manifest = _load_manifest()

    if file_hash in manifest:
        return {
            "status": "duplicate",
            "chunks_added": 0,
            "display_name": manifest[file_hash]["display_name"],
        }

    pages = extract_document(file_path, display_name=display_name)
    chunks = chunk_pages(pages)
    _store.add_chunks(chunks)

    manifest[file_hash] = {"display_name": display_name, "chunk_count": len(chunks)}
    _save_manifest(manifest)

    return {"status": "ingested", "chunks_added": len(chunks), "display_name": display_name}


def answer_query(query_text: str, top_k: int = 5) -> dict:
    """
    Retrieve relevant chunks for a query and generate a grounded answer.
    Returns a dict with the answer text and the source chunks used,
    so the UI can render citations.
    """
    retrieved = _store.query(query_text, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No documents have been ingested yet, or none were relevant to this query.",
            "sources": [],
        }

    answer = generate_answer(query_text, retrieved)

    sources = [
        {
            "source_file": r["metadata"]["source_file"],
            "page_number": r["metadata"]["page_number"],
            "snippet": r["text"][:200],
        }
        for r in retrieved
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py ingest <file_path>")
        print("  python pipeline.py ask '<question>'")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        result = ingest_file(sys.argv[2])
        if result["status"] == "duplicate":
            print(f"Skipped - '{result['display_name']}' was already ingested.")
        else:
            print(f"Ingested {result['chunks_added']} chunks from {result['display_name']}")

    elif command == "ask":
        result = answer_query(sys.argv[2])
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source_file']} p.{s['page_number']}")

    else:
        print(f"Unknown command: {command}")
