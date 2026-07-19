"""
run_evaluation.py
------------------
Retrieval evaluation harness. This is the piece that turns "I built RAG"
into "I measured my RAG system and it works" - the single highest-value
addition for interview credibility.

What it measures: for each test question, did the expected source page
show up in the top-k retrieved results? We report this "hit rate" for
vector-only, BM25-only, and hybrid retrieval separately, so you can see
concretely whether hybrid actually helps on YOUR documents - not just
take it on faith.

Usage:
    python run_evaluation.py <file_path> <eval_dataset.json> [top_k]

Example:
    python run_evaluation.py ../data/uploads/resume.pdf eval_dataset.json 5
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extract import extract_document
from chunker import chunk_pages
from hybrid_retrieval import HybridRetriever


def load_eval_dataset(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def hit_at_k(results: list, expected_source_file: str, expected_page: int) -> bool:
    """Did the expected (source_file, page) appear anywhere in the results?"""
    for r in results:
        meta = r["metadata"]
        if meta["source_file"] == expected_source_file and meta["page_number"] == expected_page:
            return True
    return False


def run_evaluation(file_path: str, dataset_path: str, top_k: int = 5):
    print(f"Ingesting {file_path} for evaluation...")
    pages = extract_document(file_path, display_name=Path(file_path).name)
    chunks = chunk_pages(pages)

    retriever = HybridRetriever()
    retriever.add_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks.\n")

    dataset = load_eval_dataset(dataset_path)

    results_table = []
    for item in dataset:
        question = item["question"]
        expected_file = item["expected_source_file"]
        expected_page = item["expected_page"]

        vector_results = retriever.vector_store.query(question, top_k=top_k)
        keyword_results = retriever.keyword_index.query(question, top_k=top_k)
        hybrid_results = retriever.query(question, top_k=top_k)

        row = {
            "question": question,
            "vector_hit": hit_at_k(vector_results, expected_file, expected_page),
            "bm25_hit": hit_at_k(keyword_results, expected_file, expected_page),
            "hybrid_hit": hit_at_k(hybrid_results, expected_file, expected_page),
        }
        results_table.append(row)

    # --- Print per-question results ---
    print(f"{'Question':<50} {'Vector':<8} {'BM25':<8} {'Hybrid':<8}")
    print("-" * 76)
    for row in results_table:
        q_display = (row["question"][:47] + "...") if len(row["question"]) > 47 else row["question"]
        print(f"{q_display:<50} "
              f"{'HIT' if row['vector_hit'] else 'miss':<8} "
              f"{'HIT' if row['bm25_hit'] else 'miss':<8} "
              f"{'HIT' if row['hybrid_hit'] else 'miss':<8}")

    # --- Print summary accuracy ---
    n = len(results_table)
    vector_acc = sum(r["vector_hit"] for r in results_table) / n * 100
    bm25_acc = sum(r["bm25_hit"] for r in results_table) / n * 100
    hybrid_acc = sum(r["hybrid_hit"] for r in results_table) / n * 100

    print("\n" + "=" * 76)
    print(f"Retrieval accuracy @ top-{top_k} (n={n} questions)")
    print(f"  Vector-only : {vector_acc:.1f}%")
    print(f"  BM25-only   : {bm25_acc:.1f}%")
    print(f"  Hybrid      : {hybrid_acc:.1f}%")
    print("=" * 76)

    return {"vector_accuracy": vector_acc, "bm25_accuracy": bm25_acc, "hybrid_accuracy": hybrid_acc}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_evaluation.py <file_path> <eval_dataset.json> [top_k]")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_path = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    run_evaluation(file_path, dataset_path, top_k)
