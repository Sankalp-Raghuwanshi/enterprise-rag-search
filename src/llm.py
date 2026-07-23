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
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated June 2026; verify at console.groq.com/docs/models


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.1, model: str = GROQ_MODEL) -> str:
    """
    General-purpose LLM call - the raw building block underneath both
    generate_answer() (grounded Q&A) and the agent's planning steps
    (deciding whether to retrieve, decomposing questions). Keeping this
    as one shared function means both use cases go through the same
    API-calling logic, error handling, and provider config.
    """
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
            "temperature": temperature,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


MAX_CHUNK_CHARS_IN_PROMPT = 1200  # caps each chunk's contribution to the prompt - see generate_answer()


def generate_answer(query: str, context_chunks: list, model: str = GROQ_MODEL) -> str:
    """
    Given a user query and a list of retrieved chunk dicts (with 'text' and
    'metadata'), build a grounded prompt and call the LLM. Returns raw answer
    text (citation formatting is handled by the caller in pipeline.py).

    Each chunk's text is capped at MAX_CHUNK_CHARS_IN_PROMPT before being
    added to the prompt. This matters more than it looks: text chunks from
    chunker.py are already bounded (~800 chars), but vision-generated image
    descriptions (vision.py) can run much longer and unpredictably - a
    request combining several long image descriptions can exceed the API's
    payload size limit (a real 413 error hit during testing). Truncating
    per-chunk keeps the total prompt size bounded regardless of what kind
    of chunk it is or how verbose the vision model gets.
    """
    context_blocks = []
    for i, chunk in enumerate(context_chunks):
        src = chunk["metadata"]["source_file"]
        page = chunk["metadata"]["page_number"]
        text = chunk["text"]
        if len(text) > MAX_CHUNK_CHARS_IN_PROMPT:
            text = text[:MAX_CHUNK_CHARS_IN_PROMPT] + "... [truncated]"
        context_blocks.append(f"[Chunk {i+1} | Source: {src}, p.{page}]\n{text}")

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user's question "
        "using ONLY the provided context chunks. If the answer isn't in the "
        "context, say so clearly - do not make anything up.\n\n"
        "When you use information from a chunk, cite it inline using the format "
        "[Chunk N] matching the chunk numbers given below."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    return call_llm(system_prompt, user_prompt, temperature=0.1, model=model)


if __name__ == "__main__":
    # Quick manual test with fake context
    fake_chunks = [
        {"text": "Terraform is used to provision AWS infrastructure for the search API.",
         "metadata": {"source_file": "deployment.pdf", "page_number": 3}}
    ]
    answer = generate_answer("Which document mentions Terraform?", fake_chunks)
    print(answer)
