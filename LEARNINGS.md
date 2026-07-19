# Learnings: Phase 1 — What You Actually Built and Why

This is the "explain it like I'm learning it" version. Read this alongside
the actual code in `src/` — each section below tells you *why* that file
exists and what it's doing conceptually, not just what commands to run.

---

## 1. The big picture: what problem are we solving?

You can't just paste a whole PDF into an LLM and ask it questions — LLMs
have a limited context window, and even when documents fit, stuffing in
irrelevant text wastes tokens and confuses the model.

**RAG (Retrieval-Augmented Generation)** solves this with two stages:

1. **Retrieval**: find the small handful of relevant text snippets from
   your documents that actually relate to the question.
2. **Generation**: hand those snippets (not the whole document) to an
   LLM and ask it to answer *using only that context*.

Everything you built is either preparing data for retrieval (extract →
chunk → embed → store) or doing retrieval + generation at query time
(retrieve → prompt → answer).

---

## 2. `extract.py` — turning files into plain text

**Problem it solves**: PDFs and Word docs aren't plain text internally —
they're structured binary formats with fonts, layout, XML, etc. You need
a library that knows how to parse that format and pull out just the text.

- `pypdf.PdfReader` reads a PDF page by page and extracts the text layer.
- `python-docx`'s `Document` object reads a Word file's paragraphs.

**Why we kept `page_number` / `source_file` on every extracted piece**:
this is the seed of citations. If you throw away *where* text came from
at this stage, you can never get it back later. Notice how `page_number`
flows through `extract.py` → `chunker.py` → `embed_store.py` → the final
answer's "Sources" list. That's one continuous thread — trace it if you
want to really understand the codebase.

**Limitation you now know about**: this only works on PDFs with a real
text layer. A *scanned* PDF (a photo of a page saved as PDF) has no text
layer at all — `pypdf` would extract nothing or garbage. Fixing that
needs OCR (a different tool entirely), which is out of scope for Phase 1.

---

## 3. `chunker.py` — why we can't just embed whole documents

**Problem it solves**: embedding models (next section) work best on
small, focused pieces of text — a paragraph-sized chunk, not a 10-page
document. Also, retrieval only returns *whole chunks* — if a whole
document is one chunk, you'd retrieve the whole thing every time, which
defeats the purpose of retrieval.

**Why "recursive" splitting**: naively cutting text every 800 characters
would slice sentences in half. Our splitter tries paragraph breaks
first (`\n\n`), then line breaks, then sentence boundaries (`. `), and
only does a hard character cut as a last resort. This keeps each chunk
a coherent, self-contained thought as much as possible.

**Why overlap (150 characters)**: imagine an important sentence sits
exactly on the boundary between chunk 4 and chunk 5. Without overlap,
that idea gets split and neither chunk fully contains it. Overlap means
the tail of one chunk repeats at the start of the next, so boundary
information doesn't get lost.

**The tuning knobs that matter**: `CHUNK_SIZE` and `CHUNK_OVERLAP` at
the top of the file. Smaller chunks = more precise retrieval but less
context per chunk. Bigger chunks = more context but less precise
matching. This is a real tradeoff you'll be asked about in interviews —
there's no universally "correct" number, only ones you tested and chose.

---

## 4. `embed_store.py` — the two most important concepts in this whole project

### 4a. What is an embedding?

An embedding model turns a piece of text into a list of numbers (a
**vector** — for `all-MiniLM-L6-v2`, 384 numbers). The key property:
**texts with similar meaning produce vectors that are numerically close
together**, even if they don't share any of the same words.

Example: "How do I reset my password?" and "I forgot my login
credentials" would embed to nearby vectors, even though they share
almost no words. This is what makes *semantic* search possible — it's
matching on meaning, not on exact keywords.

`self.model.encode(texts)` is the line where this actually happens —
every chunk's text goes in, a list of 384 numbers comes out.

### 4b. What is a vector store, and what does "search" mean here?

Once every chunk is a vector, "search" becomes a math problem: embed
the user's question into a vector too, then find which stored chunk
vectors are *closest* to it (we use cosine similarity — the angle
between two vectors, not their raw distance).

**Chroma** is the database that stores all these vectors efficiently
and can answer "give me the top-k closest vectors to this query vector"
quickly, even across thousands of chunks. `collection.query(...)` is
where that search actually happens.

### 4c. Why this specific embedding model?

`all-MiniLM-L6-v2` is small (~80MB), runs on CPU with no GPU needed, and
is free (runs locally — no API calls, no cost, no rate limits). Bigger
embedding models exist and can be more accurate, but this is a completely
reasonable choice for a project at this scale, and you can justify the
choice in an interview (speed/cost vs. marginal accuracy gain).

---

## 5. `llm.py` — turning retrieved chunks into an actual answer

This is the "Generation" half of RAG. Two things matter here:

**The system prompt is doing real work**: notice it explicitly says
*"using ONLY the provided context... if the answer isn't in the
context, say so clearly."* Without this instruction, LLMs will happily
answer from their general training knowledge instead of your documents
— which defeats the entire purpose of RAG (this is called
**hallucination outside the provided context**, and grounding
instructions like this are the main defense against it).

**Low temperature (0.1)**: temperature controls how "creative" vs.
"deterministic" the model's output is. For factual Q&A over documents,
you want low creativity — you want the model to stick closely to what's
actually in the context, not invent phrasing that drifts from the source.

**Why chunk numbering (`[Chunk 1]`, `[Chunk 2]`...) matters**: this is
the mechanism the LLM uses to tell you *which* piece of context it used
for each claim. The `pipeline.py` code maps those chunk numbers back to
real source files/pages for display — this is your citation system.

---

## 6. `pipeline.py` — why a thin orchestration layer exists at all

Every other file is a specialist: extraction knows about PDFs, chunking
knows about text-splitting, embedding knows about vectors, LLM knows
about prompting. `pipeline.py` doesn't know *how* to do any of these
things — it just calls them in the right order:

```
ingest_file()  →  extract → chunk → embed & store
answer_query() →  retrieve → generate → attach sources
```

This separation matters because it means you can change *how* chunking
works (say, in Phase 2) without touching extraction, embedding, or the
UI at all. That's the whole point of breaking code into modules instead
of one giant script — it's a real software design principle
(**separation of concerns**), not just tidiness.

---

## 7. `app.py` — Streamlit basics

Streamlit re-runs your *entire script top to bottom* every time you
interact with the page (upload a file, click a button, type in a box).
That's why `st.session_state` exists — it's the only way to remember
things (like "which files have already been ingested this session")
across those re-runs. Without it, every click would forget everything
and re-ingest files repeatedly.

---

## 8. Environment/tooling lessons (the frustrating parts you fought through)

These aren't ML concepts, but they're real skills you now have evidence
of, and they're worth understanding *why* they happened:

- **`venv` (virtual environment)**: an isolated, self-contained Python
  installation just for this project, so its package versions don't
  conflict with other projects or your system Python. This is why
  `pip` "wasn't found" until you activated the venv — outside of it,
  your shell was looking at a different Python that didn't have these
  packages installed at all.
- **Why `python` vs `python3` mattered**: macOS ships a minimal system
  Python accessible only via `python3`. Once you're *inside* an
  activated venv, `python` works too, because the venv defines its own
  `python` pointing at itself.
- **Why quoting file paths with spaces mattered**: your shell treats
  unquoted spaces as argument separators — `ingest resume file.pdf`
  looks like two separate arguments (`resume` and `file.pdf`) without
  quotes. Wrapping the whole path in quotes tells the shell "this is one
  single argument."
- **Groq vs. Grok**: two unrelated companies with near-identical names
  — Groq (`groq.com`, fast inference hardware, keys start `gsk_`) and
  xAI's Grok model (`x.ai`). Worth remembering since it's a genuinely
  common mix-up, not a mistake unique to you.
- **`.env` files**: a plain text file holding secrets (API keys) that
  your code loads at runtime via `load_dotenv()`, so the key never gets
  hardcoded into source code or accidentally committed to GitHub.

---

## 9. What Phase 1 *can't* do yet (and why Phase 2+ exists)

Try asking the app something like *"which document mentions Terraform"*
— a query where the *exact word* matters more than the general meaning.
Vector search is built for semantic similarity, not exact keyword
matching, so it can genuinely miss queries like this. That's not a bug
— it's a real, well-known limitation of pure vector search, and it's
exactly the gap **BM25 (keyword search) + hybrid retrieval** in Phase 2
is designed to close. You now have a real, first-hand example of the
problem you're about to go solve, instead of just reading about it.

---

## How to use this file

Re-read this after you've poked around the actual code in `src/`. If
any section still doesn't make sense with a specific line of code,
that's a good, specific question to bring back — much more useful than
"explain embeddings" in the abstract.
