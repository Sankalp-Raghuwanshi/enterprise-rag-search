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

---
---

# Part 2: Phases 2-5 — Hybrid Retrieval, Reranking, Citations, Evaluation

---

## 10. `keyword_index.py` — BM25 and why exact terms need their own method

Vector search (Section 4) is built to match *meaning*. BM25 is built to
match *terms* — it counts how often query words appear in each chunk,
weighted so rare/distinctive words matter more than common ones ("the,"
"and" barely count; "RESPIRA," "Terraform," an application number count
a lot). No embeddings, no neural network - just counting, with some
statistical weighting.

**Why BM25 alone isn't enough either**: a query like "does this person
have any interest in India-Japan relations?" shares almost no exact
words with the resume text that actually answers it (which talks about
"Kaizen," "engineering culture," "collaboration between India and
Japan"). BM25 would likely miss this. This is why neither method alone
is sufficient - each is strong exactly where the other is weak.

**Persistence note**: unlike Chroma, the `rank_bm25` library has no
built-in save/load. `keyword_index.py` handles this itself by writing
every chunk to a `.jsonl` file on disk and rebuilding the in-memory BM25
index from that file on startup (`_load_existing()` /
`_rebuild_bm25()`). This is a deliberately simple approach - fine at
this project's scale (thousands of chunks), not how you'd do it at much
larger scale.

---

## 11. `hybrid_retrieval.py` — combining two incompatible scoring systems

**The core problem**: vector search returns *cosine distance* (roughly
0-2, lower = better). BM25 returns an *unbounded score* that depends on
your specific corpus (higher = better). You cannot add these two numbers
together meaningfully - they're not the same unit, not even the same
direction (one is "lower is better," the other "higher is better").

**The fix: Reciprocal Rank Fusion (RRF)**. Instead of using each
method's raw score, RRF only looks at *rank position* - was this chunk
#1, #2, #3... in each list? Formula:

```
fused_score(chunk) = sum over each method of  1 / (60 + rank)
```

A chunk ranked #1 by either method scores highly. A chunk ranked #1 in
*both* scores even higher (the fusion is additive). This sidesteps the
scale-mismatch problem entirely - rank position is comparable across
any two ranking methods, regardless of their internal scoring math.

**Why pull a wider candidate pool first** (`candidates_per_method=20`
in `HybridRetriever.query()`): if you only fused each method's top-5,
a chunk ranked #7 by vector search but #1 by BM25 would never even be
considered for fusion - it got cut off before fusion had a chance to
reward its strong BM25 rank. Casting a wider net before fusing is what
lets one method "rescue" a result the other method underrated.

---

## 12. `reranker.py` — the precision pass hybrid retrieval can't do alone

Vector search and BM25 both score the query against each chunk
*independently* and fast - that's what lets them scale to searching
across thousands of chunks in milliseconds. But scoring independently
has a ceiling: neither method ever actually reads the query and a
candidate chunk *together* and reasons about them jointly.

A **cross-encoder** does exactly that - it takes (query, chunk) as one
combined input and outputs a single relevance score, with much richer
reasoning than "how close are these two vectors" or "how many words
overlap." It's far more accurate, but also far slower - way too slow to
run against every chunk in a large corpus.

**The pattern**: use fast-but-approximate methods (vector + BM25) to
narrow thousands of chunks down to a shortlist (in this project: top 15
candidates via `retrieval_pool` in `pipeline.py`), then use the
slow-but-accurate cross-encoder only on that small shortlist. This
"retrieve wide, rerank narrow" pattern is standard in real search
systems, not something invented for this project.

---

## 13. Citation filtering (Phase 4) — showing only what was actually used

Originally, the Sources panel showed every chunk that was *retrieved*,
whether or not the LLM's answer actually referenced it. That's
misleading - it implies more of the answer is grounded than it actually
is.

**The fix** (`answer_query()` in `pipeline.py`): after generating the
answer, we parse it for `[Chunk N]` markers using a regex
(`re.findall(r"\[Chunk (\d+)\]", answer)`), then only include those
specific chunks in the returned `sources` list. If the model doesn't
use the citation format at all (which can happen - LLMs don't always
follow formatting instructions perfectly), we fall back to showing
everything retrieved rather than an empty, unhelpful sources list.

This is a small function, but it's a meaningful trust signal: the
Sources panel now tells the truth about what was actually used.

---

## 14. Evaluation harness (Phase 5) — from "I built RAG" to "I measured RAG"

This is the single highest-value piece of the whole project for
interview credibility, and it's worth understanding exactly what it
proved.

**What it measures**: for each test question, did the *correct* source
page appear anywhere in the top-5 retrieved results? This is called
**hit rate @ k** (here, k=5) - a standard retrieval evaluation metric.
It's a binary yes/no per question, averaged into a percentage across
the whole test set.

**Why this specific metric, not answer-quality grading**: retrieval and
generation are two separate failure points in a RAG system. If
retrieval finds the wrong chunk, no amount of good prompting can produce
a correct answer - the LLM literally never sees the right information.
Measuring retrieval accuracy in isolation (before even looking at the
LLM's final answer) tells you specifically whether *that* stage is
working, which is a more diagnostic, more precise measurement than
just eyeballing whether final answers "look right."

### The real result, from your own resume (12 test questions, top-5):

```
Vector-only : 91.7%  (11/12 - missed one)
BM25-only   : 100.0%
Hybrid      : 100.0%
```

**What the miss actually revealed**: vector search failed on *"What is
this person's major and where do they study?"* - a plain, natural
factual question, not an edge case you'd expect to trip up semantic
search. With only 13 total chunks, several other page-5 chunks were
all reasonably "close" in meaning, and the correct one got pushed just
outside the top-5 window. BM25 caught it because the answer chunk
contains near-exact matching words ("major," specific university name).
**Hybrid retrieval recovered this miss and reached 100%.**

This is a genuinely good result to have measured yourself, and a good
story: hybrid retrieval didn't just help on the "obvious" case (rare
keywords like Terraform) - it recovered a failure on an ordinary
semantic query too, which is a more surprising and more convincing
demonstration of its value.

**Caveat worth knowing for interviews**: 12 questions against a
13-chunk document is a small, illustrative evaluation, not a rigorous
benchmark - be honest about that scale if asked. The *method* (measuring
hit rate, comparing retrieval strategies head-to-head) is the
transferable, real skill; the specific 100% number is a small-sample
result on one document, not a claim about the system's accuracy in
general.

---

## 15. `api.py` — why a separate FastAPI layer exists at all

Streamlit is a UI framework, not a service framework - it's not
designed to be called by other programs, only rendered as a page in a
browser. Wrapping the same pipeline functions (`ingest_file`,
`answer_query`) in FastAPI turns them into a real, independently
callable HTTP service - something another application (or a deployed
frontend, or a script, or `curl`) could call directly, with proper
request/response schemas (Pydantic models here) and interactive docs
(FastAPI auto-generates a `/docs` page from your code).

This is also what makes cloud deployment meaningful: you deploy the
*API*, not the Streamlit demo - the Streamlit app becomes just one
possible client of that API, not the only way to use the system.

---

## 16. Terraform (Phase 7) — EC2, not Lambda, and why that's a real decision

AWS Lambda is often the "default" answer for deploying small services,
but it has a hard deployment package size limit, and PyTorch +
sentence-transformers + model weights blow well past it. Lambda also
"cold starts" - spinning up fresh for each request - which is a poor
fit for models you want to stay loaded in memory between calls.

An EC2 instance sidesteps both problems: normal-sized dependencies,
and the models load once when the service starts, then stay warm.
The Terraform config in `terraform/` provisions exactly that - one
small EC2 instance, a security group opening only the ports actually
needed (22 for SSH, 8000 for the API), and a boot script
(`user_data.sh.tpl`) that installs everything and starts the API as a
`systemd` service (so it restarts automatically if it crashes or the
instance reboots).

This is a real, explainable infrastructure decision with a tradeoff you
can articulate - not just "I used Terraform because the internship
listing mentioned it."

---

## 17. `agent.py` — an agentic layer, and why it's hand-written, not a framework

Everything up to this point answers ONE question against retrieved
context in ONE generation call. The agent adds a layer of *decision-making
before* that: should we even search documents for this? Does this
question need to be broken into pieces first?

**Three separate LLM calls, each doing one simple job:**

1. **Route** (`needs_retrieval()`): a strict yes/no classification -
   does this question need document search at all? Kept deliberately
   narrow (one-word answer) rather than open-ended, so this step is
   cheap, fast, and can't itself hallucinate content - it's only ever
   choosing between two paths.

2. **Decompose** (`decompose_question()`): breaks a compound question
   ("what is X and what is Y") into independent sub-questions that each
   retrieve well on their own. This is the exact same pattern as the
   query decomposition in the NL-to-SQL project - break a complex ask
   into simpler pieces, solve each piece, combine the results - just
   applied to unstructured document search instead of SQL.

3. **Synthesize**: retrieve context for EVERY sub-question separately
   (each sub-question gets its own hybrid retrieval + rerank pass), then
   merge and deduplicate all the retrieved chunks, and generate ONE
   final answer across the combined context - not a list of separate
   sub-answers stitched together, but one coherent response citing
   whichever chunks actually mattered.

**Why deduplication happens at two different levels** - this tripped up
an early test, worth understanding: `_dedupe_chunks()` merges by chunk
*id* (so the same exact chunk retrieved by two different sub-questions
isn't processed twice). But two DIFFERENT chunks can still share the
same source page (remember: `chunker.py` can split one page into
several chunks). So there's a second, separate dedup step in the final
sources list - by (source_file, page_number) - so the user-facing
Sources panel doesn't show the same page repeated just because it was
split into multiple retrieved chunks.

**Why no LangChain or other agent framework**: at this project's scope
- three LLM calls in a fixed sequence, no tool-calling loop, no long-horizon
multi-step planning - a framework adds abstraction layers without adding
real capability. Writing the control flow by hand (`run_agent()` is
maybe 40 lines) means every decision point is a plain, readable function
you can point to and explain line-by-line, rather than "the framework
handles that part." This is the same reasoning as the earlier decision
to hand-write chunking instead of using a library - transparency over
convenience, at this scale.

**What would justify a real framework later**: if the agent needed to
dynamically choose from many different tools (not just "retrieve or
don't"), loop and re-plan based on intermediate results, or run
open-ended multi-step reasoning - that's when a framework's abstractions
(state management, tool-calling protocols) start earning their
complexity cost. Recognizing *when* a framework is worth it - not just
defaulting to one - is itself the more valuable engineering judgment.

