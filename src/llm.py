"""
llm.py
------
Thin wrapper around whichever LLM API you're using (Groq, by default,
since that's what you already have a key for - swap the base_url/model
if you switch providers later).

NOTE: "Groq" (api.groq.com, fast-inference hardware company, keys start
with gsk_) is a DIFFERENT company from "Grok" (xAI's model, api.x.ai).
Easy to mix up because they sound identical - this file uses Groq.

Keeping this isolated in one file means swapping providers later
is a one-file change, not a refactor.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # check console.groq.com/docs/models for current options


def generate_answer(query: str, context_chunks: list, model: str = GROQ_MODEL) -> str:
    """
    Given a user query and a list of retrieved chunk dicts (with 'text' and
    'metadata'), build a grounded prompt and call the LLM. Returns raw answer
    text (citation formatting is handled by the caller in pipeline.py).
    """
    context_blocks = []
    for i, chunk in enumerate(context_chunks):
        src = chunk["metadata"]["source_file"]
        page = chunk["metadata"]["page_number"]
        context_blocks.append(f"[Chunk {i+1} | Source: {src}, p.{page}]\n{chunk['text']}")

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user's question "
        "using ONLY the provided context chunks. If the answer isn't in the "
        "context, say so clearly - do not make anything up.\n\n"
        "When you use information from a chunk, cite it inline using the format "
        "[Chunk N] matching the chunk numbers given below."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    fake_chunks = [
        {"text": "Terraform is used to provision AWS infrastructure for the search API.",
         "metadata": {"source_file": "deployment.pdf", "page_number": 3}}
    ]
    answer = generate_answer("Which document mentions Terraform?", fake_chunks)
    print(answer)
