"""
agent.py
--------
An agentic layer on top of the RAG pipeline. Three responsibilities,
each a separate LLM call so each step is simple, inspectable, and
debuggable on its own:

  1. ROUTE: does this question need document retrieval at all, or is it
     conversational / answerable without looking anything up?
  2. DECOMPOSE: if the question is complex ("compare X and Y", "why did
     Z happen"), break it into simpler sub-questions that each retrieve
     well on their own - the same pattern used in the NL-to-SQL project's
     query decomposition, just applied to unstructured document search
     instead of SQL.
  3. SYNTHESIZE: retrieve context for every sub-question, merge and
     deduplicate the results, and generate ONE final grounded answer
     across all of it - so the user gets a single coherent response,
     not a list of disconnected sub-answers.

Design choice: no agent framework (LangChain, etc.) - this is a small,
fully hand-written control loop. At this project's scope (a few LLM
calls in a fixed sequence, no tool-calling loop or long-horizon planning)
a framework would add abstraction without adding real capability, and
would make the actual decision logic harder to point to and explain.
"""

import json
import re
from typing import List

from llm import call_llm, generate_answer
from pipeline import retrieve_context

MAX_SUBQUESTIONS = 4        # cap decomposition - prevents runaway LLM calls on adversarial input
MAX_MERGED_CHUNKS = 8        # cap final context size sent to the synthesis step


def needs_retrieval(question: str) -> bool:
    """
    ROUTE step: classify whether this question requires searching the
    ingested documents, or can be answered without retrieval (greetings,
    meta-questions about the assistant itself, etc.).

    Kept as a strict yes/no classification (not a free-form LLM answer)
    so it's cheap, fast, and doesn't itself risk hallucinating content.
    """
    system_prompt = (
        "You classify whether a user question requires searching a document "
        "knowledge base to answer, or can be answered without it (e.g. greetings, "
        "small talk, or questions about how the assistant itself works).\n\n"
        "Respond with ONLY one word: YES or NO."
    )
    response = call_llm(system_prompt, question, temperature=0.0)
    return response.strip().upper().startswith("Y")


def decompose_question(question: str) -> List[str]:
    """
    DECOMPOSE step: break a complex question into simpler sub-questions
    that each retrieve well independently. Simple questions come back
    as a single-item list unchanged - decomposition only kicks in when
    the LLM judges it genuinely helps.
    """
    system_prompt = (
        "You break down a user's question into 1-{max_n} simpler sub-questions "
        "that could each be answered independently by searching a document "
        "knowledge base, then combined into a full answer.\n\n"
        "If the question is already simple and doesn't need breaking down, "
        "return it unchanged as a single sub-question.\n\n"
        "Respond with ONLY a JSON array of strings, nothing else. "
        'Example: ["sub-question 1", "sub-question 2"]'
    ).format(max_n=MAX_SUBQUESTIONS)

    response = call_llm(system_prompt, question, temperature=0.0)

    try:
        # Models sometimes wrap JSON in markdown fences despite instructions -
        # strip those defensively before parsing.
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
        sub_questions = json.loads(cleaned)
        if isinstance(sub_questions, list) and sub_questions:
            return sub_questions[:MAX_SUBQUESTIONS]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: if parsing fails for any reason, just treat the original
    # question as a single sub-question rather than failing the whole request.
    return [question]


def _dedupe_chunks(chunks: List[dict]) -> List[dict]:
    """Merge chunk lists from multiple sub-questions, dropping duplicates by id."""
    seen_ids = set()
    deduped = []
    for chunk in chunks:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            deduped.append(chunk)
    return deduped


def run_agent(question: str, top_k_per_subquestion: int = 5) -> dict:
    """
    Full agent loop: route -> (optionally decompose) -> retrieve per
    sub-question -> merge -> synthesize one final answer.

    Returns: {"answer": str, "sources": list, "trace": dict} - "trace"
    exposes the agent's intermediate decisions (used retrieval? which
    sub-questions?) so the UI can show its work, which is valuable both
    for user trust and for debugging.
    """
    trace = {"used_retrieval": False, "sub_questions": [question]}

    if not needs_retrieval(question):
        answer = call_llm(
            "You are a helpful assistant. Answer directly and concisely.",
            question,
            temperature=0.3,
        )
        return {"answer": answer, "sources": [], "trace": trace}

    trace["used_retrieval"] = True
    sub_questions = decompose_question(question)
    trace["sub_questions"] = sub_questions

    all_chunks = []
    for sub_q in sub_questions:
        all_chunks.extend(retrieve_context(sub_q, top_k=top_k_per_subquestion))

    merged_chunks = _dedupe_chunks(all_chunks)[:MAX_MERGED_CHUNKS]

    if not merged_chunks:
        return {
            "answer": "No documents have been ingested yet, or none were relevant to this query.",
            "sources": [],
            "trace": trace,
        }

    answer = generate_answer(question, merged_chunks)

    all_sources = [
        {
            "source_file": c["metadata"]["source_file"],
            "page_number": c["metadata"]["page_number"],
            "snippet": c["text"][:200],
        }
        for c in merged_chunks
    ]

    cited_indices = set(int(n) for n in re.findall(r"\[Chunk (\d+)\]", answer))
    if cited_indices:
        sources = [s for i, s in enumerate(all_sources, start=1) if i in cited_indices]
    else:
        sources = all_sources

    # Dedupe by (source_file, page_number) for display - multiple chunks can
    # legitimately share a page (a page split into several chunks), which
    # would otherwise show the same page repeated in the Sources list.
    seen_pages = set()
    deduped_sources = []
    for s in sources:
        key = (s["source_file"], s["page_number"])
        if key not in seen_pages:
            seen_pages.add(key)
            deduped_sources.append(s)
    sources = deduped_sources

    return {"answer": answer, "sources": sources, "trace": trace}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py '<question>'")
        sys.exit(1)

    result = run_agent(sys.argv[1])

    print(f"\nUsed retrieval: {result['trace']['used_retrieval']}")
    print(f"Sub-questions: {result['trace']['sub_questions']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source_file']} p.{s['page_number']}")
