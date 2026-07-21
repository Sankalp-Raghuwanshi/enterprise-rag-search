"""
pipeline.py
-----------
Ties extract -> chunk -> retrieve (hybrid) -> rerank -> generate into
two simple functions:

    ingest_file(file_path, display_name=None)
    answer_query(query_text)

Phase 2: hybrid retrieval (vector + BM25 fused via RRF), content-hash
dedup on ingestion.
Phase 3: cross-encoder reranking - retrieve a wide candidate pool via
hybrid search, then rerank down to the final top_k for precision.
Phase 4: sources shown to the user are filtered to only the chunks the
LLM actually cited (parsed from "[Chunk N]" markers in the answer).

Keep this file thin - it's the orchestration layer, not where logic lives.
"""

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import List

from extract import extract_document
from chunker import chunk_pages, Chunk
from image_extractor import extract_images
from vision import describe_image
from hybrid_retrieval import HybridRetriever
from reranker import Reranker
from llm import generate_answer

# Single shared instances for the app's lifetime.
_store = HybridRetriever()
_reranker = Reranker()

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


def _process_images(file_path: str, display_name: str) -> List[Chunk]:
    """
    Extract embedded images (PDF only - DOCX images not yet supported),
    describe each with a vision LLM, and wrap each description as a
    Chunk - so images become searchable through the exact same hybrid
    retrieval + rerank + citation pipeline as text, with no special
    casing needed anywhere downstream.

    Failures here (e.g. a misconfigured vision model, a corrupt image)
    are caught and skipped individually - one bad image should not
    fail the whole ingestion of an otherwise-fine document.
    """
    if Path(file_path).suffix.lower() != ".pdf":
        return []  # image extraction currently PDF-only, see image_extractor.py

    image_chunks = []
    try:
        images = extract_images(file_path, display_name=display_name)
    except Exception as e:
        print(f"Image extraction failed for {display_name}: {e}")
        return []

    for img in images:
        try:
            description = describe_image(img.image_bytes, image_ext=img.image_ext)
        except Exception as e:
            print(f"Vision description failed for {display_name} "
                  f"p.{img.page_number} image {img.image_index}: {e}")
            continue

        image_chunks.append(Chunk(
            id=str(uuid.uuid4()),
            text=f"[Image on page {img.page_number}]: {description}",
            source_file=img.source_file,
            page_number=img.page_number,
            chunk_index=f"image-{img.image_index}",
        ))

    return image_chunks


def ingest_file(file_path: str, display_name: str = None, process_images: bool = True) -> dict:
    """
    Extract, chunk, embed, and index one file - skipping it if this exact
    content has already been ingested.

    display_name: the filename to show in citations (defaults to the
    file_path's own name). Pass this explicitly when file_path is a temp
    path that doesn't reflect the user's original filename (e.g. Streamlit
    uploads).

    process_images: if True (default) and the file is a PDF, also extract
    embedded images and index vision-LLM-generated descriptions of them,
    so images become searchable alongside text.

    Returns: {"status": "ingested" | "duplicate", "chunks_added": int,
              "images_added": int, "display_name": str}
    """
    display_name = display_name or Path(file_path).name
    file_hash = _hash_file(file_path)
    manifest = _load_manifest()

    if file_hash in manifest:
        return {
            "status": "duplicate",
            "chunks_added": 0,
            "images_added": 0,
            "display_name": manifest[file_hash]["display_name"],
        }

    pages = extract_document(file_path, display_name=display_name)
    chunks = chunk_pages(pages)

    image_chunks = _process_images(file_path, display_name) if process_images else []

    all_chunks = chunks + image_chunks
    _store.add_chunks(all_chunks)

    manifest[file_hash] = {"display_name": display_name, "chunk_count": len(all_chunks)}
    _save_manifest(manifest)

    return {
        "status": "ingested",
        "chunks_added": len(chunks),
        "images_added": len(image_chunks),
        "display_name": display_name,
    }


def retrieve_context(query_text: str, top_k: int = 5, retrieval_pool: int = 15) -> list:
    """
    Retrieve and rerank chunks for a query, WITHOUT generating an LLM answer.
    This is the piece the agent (agent.py) calls once per sub-question,
    before combining everything into a single final generation step -
    separating "find relevant context" from "write an answer" is what
    lets the agent run retrieval multiple times cheaply while only
    calling the (slower, costlier) generation step once at the end.
    """
    candidates = _store.query(query_text, top_k=retrieval_pool)
    if not candidates:
        return []
    return _reranker.rerank(query_text, candidates, top_k=top_k)


def answer_query(query_text: str, top_k: int = 5, retrieval_pool: int = 15) -> dict:
    """
    Retrieve relevant chunks for a query, rerank them, and generate a
    grounded answer. Returns a dict with the answer text and the source
    chunks the LLM actually cited, so the UI can render citations.

    retrieval_pool: how many candidates hybrid retrieval pulls BEFORE
    reranking narrows down to top_k. Wider than top_k on purpose - the
    cross-encoder needs a reasonable shortlist to actually improve on,
    not just the same top_k hybrid retrieval already picked.
    """
    reranked = retrieve_context(query_text, top_k=top_k, retrieval_pool=retrieval_pool)

    if not reranked:
        return {
            "answer": "No documents have been ingested yet, or none were relevant to this query.",
            "sources": [],
        }

    answer = generate_answer(query_text, reranked)

    all_sources = [
        {
            "source_file": r["metadata"]["source_file"],
            "page_number": r["metadata"]["page_number"],
            "snippet": r["text"][:200],
        }
        for r in reranked
    ]

    # Phase 4: only surface sources the LLM actually cited (e.g. "[Chunk 2]"),
    # so the Sources panel reflects what was used, not everything retrieved.
    cited_indices = set(int(n) for n in re.findall(r"\[Chunk (\d+)\]", answer))
    if cited_indices:
        sources = [s for i, s in enumerate(all_sources, start=1) if i in cited_indices]
    else:
        # Fallback: if the model didn't use the citation format, show everything
        # retrieved rather than an empty (and misleading) sources list.
        sources = all_sources

    # Dedupe by (source_file, page_number) for display - multiple chunks can
    # legitimately share a page, which would otherwise repeat in the list.
    seen_pages = set()
    deduped_sources = []
    for s in sources:
        key = (s["source_file"], s["page_number"])
        if key not in seen_pages:
            seen_pages.add(key)
            deduped_sources.append(s)
    sources = deduped_sources

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
            print(f"Ingested {result['chunks_added']} chunks "
                  f"({result.get('images_added', 0)} images described) from {result['display_name']}")

    elif command == "ask":
        result = answer_query(sys.argv[2])
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source_file']} p.{s['page_number']}")

    else:
        print(f"Unknown command: {command}")
