"""
embed_store.py
--------------
Embeds chunks with a local sentence-transformers model and stores them
in a persistent Chroma collection. Also handles basic vector retrieval.

Model choice: all-MiniLM-L6-v2
 - small (~80MB), fast on CPU, good enough quality for a project like this
 - runs locally, no API cost, no rate limits
"""

from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import Chunk

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "../data/vectorstore"
COLLECTION_NAME = "enterprise_docs"


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and store a batch of chunks. Safe to call incrementally."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Vector similarity search. Returns list of dicts with text, metadata, distance."""
        query_embedding = self.model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    # Quick manual test: ingest a file, then run a test query against it.
    import sys
    from extract import extract_document
    from chunker import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python embed_store.py <file_path> [query]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What is this document about?"

    pages = extract_document(file_path)
    chunks = chunk_pages(pages)

    store = VectorStore()
    store.add_chunks(chunks)
    print(f"Stored {len(chunks)} chunks. Collection now has {store.count()} total.")

    results = store.query(query, top_k=5)
    print(f"\nTop results for: '{query}'")
    for r in results:
        print(f"\n[{r['metadata']['source_file']} p.{r['metadata']['page_number']}] "
              f"(distance={r['distance']:.4f})")
        print(r["text"][:200])
