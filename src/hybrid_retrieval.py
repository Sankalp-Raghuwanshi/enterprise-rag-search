"""
hybrid_retrieval.py
--------------------
Combines vector search (embed_store.py) and keyword search (keyword_index.py)
into one ranked result list, using Reciprocal Rank Fusion (RRF).

Why RRF specifically: vector "distance" and BM25 "score" are on completely
different numeric scales (cosine distance is 0-2ish, BM25 scores are
unbounded and corpus-dependent) - you can't just add them together
meaningfully. RRF sidesteps this by using each result's *rank position*
(1st, 2nd, 3rd...) instead of its raw score. A chunk ranked #1 by either
method contributes the same fusion score, regardless of what its
underlying similarity/BM25 number was.

Formula: fused_score(doc) = sum over each ranking list of  1 / (k + rank)
where k=60 is a standard constant from the original RRF paper - it
dampens the impact of very top ranks so one list doesn't totally
dominate the other.
"""

from typing import List

from embed_store import VectorStore
from keyword_index import KeywordIndex
from chunker import Chunk

RRF_K = 60  # standard constant from the RRF paper - not typically tuned


def reciprocal_rank_fusion(result_lists: List[List[dict]], k: int = RRF_K) -> List[dict]:
    """
    Merge multiple ranked result lists (each a list of dicts with an 'id' key)
    into one fused ranking. Returns results sorted by fused score, descending,
    with duplicate ids merged (keeping the first-seen text/metadata).
    """
    fused_scores = {}
    seen_records = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in seen_records:
                seen_records[item_id] = item

    ranked_ids = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)

    fused_results = []
    for item_id in ranked_ids:
        record = dict(seen_records[item_id])
        record["fused_score"] = fused_scores[item_id]
        fused_results.append(record)

    return fused_results


class HybridRetriever:
    """
    Drop-in replacement for using VectorStore alone - same add_chunks()/
    query() shape, but retrieval now blends semantic and keyword matching.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.keyword_index = KeywordIndex()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.vector_store.add_chunks(chunks)
        self.keyword_index.add_chunks(chunks)

    def query(self, query_text: str, top_k: int = 5, candidates_per_method: int = 20) -> List[dict]:
        """
        Retrieve top_k results using hybrid search.

        candidates_per_method: how many results to pull from EACH method
        before fusing - wider than top_k on purpose, so a chunk that's
        (say) rank 15 in vector search but rank 1 in BM25 still has a
        chance to surface after fusion, instead of being cut off early.
        """
        vector_results = self.vector_store.query(query_text, top_k=candidates_per_method)
        keyword_results = self.keyword_index.query(query_text, top_k=candidates_per_method)

        fused = reciprocal_rank_fusion([vector_results, keyword_results])
        return fused[:top_k]

    def count(self) -> int:
        return self.vector_store.count()


if __name__ == "__main__":
    # Compare vector-only vs hybrid on the same query - this is the
    # exact before/after evidence worth keeping for your write-up.
    import sys
    from extract import extract_document
    from chunker import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python hybrid_retrieval.py <file_path> [query]")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "Terraform"

    pages = extract_document(file_path)
    chunks = chunk_pages(pages)

    retriever = HybridRetriever()
    retriever.add_chunks(chunks)

    print(f"=== Vector-only results for: '{query}' ===")
    for r in retriever.vector_store.query(query, top_k=5):
        print(f"[{r['metadata']['source_file']} p.{r['metadata']['page_number']}] "
              f"distance={r['distance']:.4f}")
        print(r["text"][:150], "\n")

    print(f"\n=== BM25-only results for: '{query}' ===")
    for r in retriever.keyword_index.query(query, top_k=5):
        print(f"[{r['metadata']['source_file']} p.{r['metadata']['page_number']}] "
              f"score={r['score']:.4f}")
        print(r["text"][:150], "\n")

    print(f"\n=== Hybrid (fused) results for: '{query}' ===")
    for r in retriever.query(query, top_k=5):
        print(f"[{r['metadata']['source_file']} p.{r['metadata']['page_number']}] "
              f"fused_score={r['fused_score']:.4f}")
        print(r["text"][:150], "\n")
