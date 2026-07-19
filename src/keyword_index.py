"""
keyword_index.py
-----------------
BM25 keyword search - the "exact term matching" counterpart to vector
search in embed_store.py.

Why this exists: vector search finds semantic similarity ("meaning"),
but genuinely fails on queries where the *exact word* is what matters
(e.g. "which document mentions Terraform" - a rare proper noun). BM25
ranks documents by term frequency, so exact-word queries are its
strength, not its weakness.

Persistence note: rank_bm25 has no built-in save/load, so we persist
the raw chunk data ourselves (as JSONL) and rebuild the in-memory BM25
index from it on startup. For a project at this scale (hundreds to a
few thousand chunks) rebuilding on load is fast and simple - no need
for a more complex persistent search engine.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from chunker import Chunk

PERSIST_PATH = "../data/keyword_index/chunks.jsonl"


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer. No external NLP dependency needed -
    BM25 just needs consistent tokens to count, not linguistic sophistication."""
    return re.findall(r"\b\w+\b", text.lower())


class KeywordIndex:
    def __init__(self, persist_path: str = PERSIST_PATH):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._records: List[dict] = []  # each: {id, text, metadata}
        self._bm25: Optional[BM25Okapi] = None

        self._load_existing()

    def _load_existing(self) -> None:
        """Rebuild the in-memory index from disk, if a previous session left one."""
        if self.persist_path.exists():
            with open(self.persist_path, "r") as f:
                self._records = [json.loads(line) for line in f if line.strip()]
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        if not self._records:
            self._bm25 = None
            return
        tokenized_corpus = [_tokenize(r["text"]) for r in self._records]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Append new chunks to the persisted corpus and rebuild the index.
        Rebuilding on every add is simple and fine at this project's scale;
        a production system would use an incremental index instead."""
        if not chunks:
            return

        new_records = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                },
            }
            for c in chunks
        ]

        with open(self.persist_path, "a") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")

        self._records.extend(new_records)
        self._rebuild_bm25()

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Return top_k chunks ranked by BM25 score (higher = more relevant)."""
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue  # don't return completely irrelevant matches
            record = self._records[idx]
            results.append({
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
                "score": float(scores[idx]),
            })
        return results

    def count(self) -> int:
        return len(self._records)


if __name__ == "__main__":
    # Quick manual test: ingest a file, then run a BM25 query against it.
    import sys
    from extract import extract_document
    from chunker import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python keyword_index.py <file_path> [query]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "Terraform"

    pages = extract_document(file_path)
    chunks = chunk_pages(pages)

    index = KeywordIndex()
    index.add_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks. Total in index: {index.count()}")

    results = index.query(query, top_k=5)
    print(f"\nTop BM25 results for: '{query}'")
    for r in results:
        print(f"\n[{r['metadata']['source_file']} p.{r['metadata']['page_number']}] "
              f"(score={r['score']:.4f})")
        print(r["text"][:200])
