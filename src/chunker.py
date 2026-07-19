"""
chunker.py
----------
Splits extracted pages into overlapping chunks suitable for embedding.

Design choice: simple recursive character splitting (no external dependency).
We split on paragraph breaks first, then sentences, then hard character
limits as a last resort - this keeps chunks semantically coherent instead
of cutting mid-sentence.
"""

import uuid
from dataclasses import dataclass, field
from typing import List

from extract import ExtractedPage

# Tune these two - they are the main levers for retrieval quality.
CHUNK_SIZE = 800       # target characters per chunk (~150-200 tokens)
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int  # position of this chunk within its source page/block


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursively split on the largest available separator that fits."""
    separators = ["\n\n", "\n", ". ", " "]

    def split(text: str, seps: List[str]) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        if not seps:
            # last resort: hard cut
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        parts = text.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)

        # Any piece still too long gets split further down the separator list
        final = []
        for c in chunks:
            if len(c) > chunk_size:
                final.extend(split(c, seps[1:]))
            else:
                final.append(c)
        return final

    raw_chunks = split(text, separators)

    # Add overlap between consecutive chunks
    overlapped = []
    for i, c in enumerate(raw_chunks):
        if i == 0 or overlap == 0:
            overlapped.append(c)
        else:
            prev_tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + c)

    return overlapped


def chunk_pages(pages: List[ExtractedPage], chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Turn a list of ExtractedPages into a flat list of Chunks with metadata preserved."""
    all_chunks = []
    for page in pages:
        pieces = _split_text(page.text, chunk_size, overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=piece,
                source_file=page.source_file,
                page_number=page.page_number,
                chunk_index=idx,
            ))
    return all_chunks


if __name__ == "__main__":
    from extract import extract_document
    import sys

    if len(sys.argv) != 2:
        print("Usage: python chunker.py <file_path>")
        sys.exit(1)

    pages = extract_document(sys.argv[1])
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n[{c.source_file} p.{c.page_number} chunk {c.chunk_index}] ({len(c.text)} chars)")
        print(c.text[:200])
