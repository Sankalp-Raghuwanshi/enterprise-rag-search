"""
reranker.py
-----------
Cross-encoder reranking: the final precision pass after hybrid retrieval.

Why this exists: vector search and BM25 both score a query against EACH
chunk independently and fast (that's how they scale to many documents).
A cross-encoder instead looks at the query and a candidate chunk TOGETHER
in one pass, so it can judge relevance far more precisely - but it's much
slower, so it's only run on a small shortlist (e.g. top 15-20 candidates),
never the whole corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 - small, fast on CPU, trained
specifically for passage reranking (MS MARCO is a standard IR benchmark).
"""

from typing import List

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL_NAME)

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """
        candidates: list of dicts with a 'text' key (output of hybrid retrieval).
        Returns the top_k candidates re-sorted by cross-encoder relevance score,
        with a 'rerank_score' field added to each.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for candidate, score in scored[:top_k]:
            record = dict(candidate)
            record["rerank_score"] = float(score)
            results.append(record)
        return results


if __name__ == "__main__":
    # Quick manual test: rerank a fake candidate list against a query.
    fake_candidates = [
        {"id": "1", "text": "Terraform is used to provision AWS infrastructure.",
         "metadata": {"source_file": "a.pdf", "page_number": 1}},
        {"id": "2", "text": "The weather today is sunny with a light breeze.",
         "metadata": {"source_file": "b.pdf", "page_number": 1}},
        {"id": "3", "text": "Infrastructure as code tools like Terraform simplify cloud deployment.",
         "metadata": {"source_file": "c.pdf", "page_number": 1}},
    ]

    reranker = Reranker()
    results = reranker.rerank("How is AWS infrastructure provisioned?", fake_candidates)

    print("Reranked results:")
    for r in results:
        print(f"[score={r['rerank_score']:.4f}] {r['text']}")
