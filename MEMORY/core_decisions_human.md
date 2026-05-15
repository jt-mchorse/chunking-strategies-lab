# Core Decisions

Strategic decisions for this repo, with reasoning. Append-only — superseded decisions are marked, not removed.

## D-001 — Scope locked to portfolio handoff §2 (2026-05-10)
**Decision:** Scope of this repo is fixed by the portfolio handoff document, section 2.

**Why:** The handoff spec was deliberated; ad-hoc scope expansion within a session is the failure mode this prevents.

**Alternatives considered:** None — this is a baseline.

**Reversibility:** Expensive. Scope changes require a deliberate revisit and a new decision entry.

**Related issues:** —

## D-002 — Pin the shared substrate: corpus, queries, embedding model (2026-05-14)
**Decision:** All chunking-strategy comparisons in this repo run on a single fixed substrate:

- **Corpus:** 5 hand-authored Markdown technical articles in `data/corpus/` (MIT, ~600–1200 words each, structurally varied with headings/paragraphs/code blocks).
- **Queries:** 12 verbatim-snippet queries in `data/queries.jsonl`. Each carries an `expected_doc` (filename in the corpus) and an `expected_snippet` (literal text from that document).
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, Apache 2.0), pinned via `chunking_lab.CANONICAL_EMBEDDING_MODEL`.

**Why:** A chunking-strategy comparison is only legible if the strategy is the only variable. If the corpus differs across strategies, or the query set is per-implementer, or the embedder is "whatever you have on hand," the resulting numbers can't be compared — and a repo whose contribution *is* the comparison has nothing to ship. Pinning the substrate forces every strategy author to compete on the same ground. The hand-authored corpus avoids external license complexity; the verbatim-snippet queries give the retrieval matrix (#3) a hard pass/fail signal per query.

**Alternatives considered:**
- Use an external corpus (TREC, BEIR, MS MARCO) — rejected for licensing complexity and corpus-size mismatch with this repo's CI-budget.
- Leave the substrate per strategy-implementor — rejected; that's the failure mode this decision exists to prevent.

**Reversibility:** Expensive. Any edit to a corpus document, a query snippet, or the embedding-model constant invalidates every downstream benchmark number from #3. Such an edit must come with: (a) a new D-NNN that supersedes D-002, (b) a clear explanation of why the substrate changed, and (c) a rerun of #3's metrics matrix.

**Related issues:** #1, #2, #3.

## D-003 — Real embedder behind an optional extra; dep-free reference by default (2026-05-14)
**Decision:** `MiniLMEmbedder` (the adapter for the canonical model) lives behind the `[sbert]` optional extra. The package's only baseline embedder is `HashEmbedder` (deterministic SHA-256 hashing, 384-d to match the canonical model), which has zero dependencies.

**Why:** CI needs to be fast and not pay model-download cost on every PR. `HashEmbedder` keeps the test job hermetic. Users running the metrics matrix (#3) install the extra and get the real embedder. Pinning `HashEmbedder` to the same 384-d as the canonical model means strategy code parameterized over the `Embedder` Protocol transfers cleanly from test runs to real benchmark runs.

**Alternatives considered:**
- Bundle `sentence-transformers` as a required dep — rejected; downloads model weights on every fresh install, hits CI runtime, hits PyPI cache size, and adds heavy transitive deps (`torch`, `transformers`).
- Ship only `HashEmbedder` and tell users to BYO real embedder — rejected; the canonical model needs a clean adapter so the substrate's "single embedding model" promise is honored by code, not just by docs.

**Reversibility:** Cheap. The split is one dependency-set change away from any other configuration.

**Related issues:** #1, #3.

## D-004 — Each strategy is its own module; `Strategy` is a single-method Protocol (2026-05-15)
**Decision:** Five chunking strategies in separate modules under `chunking_lab/strategies/` (one per file). Shared `Strategy` Protocol with one method: `chunk(text, *, source_doc_id) -> list[Chunk]`. No ABC, no inheritance.

**Why:** Cookbook principle — copy one strategy without dragging in siblings or Protocol machinery. Same single-method Protocol pattern as the rest of the portfolio. Metrics matrix (#3) iterates uniformly via the Protocol; everywhere else they're just classes with a `chunk` method.

**Alternatives considered:**
- Single file with all strategies — rejected: copy-one means copy-all.
- Abstract base class — rejected: ABC ceremony for one-method seam.
- sklearn-style estimators — rejected: no fit step.

**Reversibility:** Cheap.

**Related issues:** #2, #3

## D-005 — `Chunk` carries `start_offset`/`end_offset` into source text (2026-05-15)
**Decision:** Every `Chunk` carries inclusive `start_offset` and exclusive `end_offset` byte offsets into the source document.

**Why:** The metrics matrix (#3) attributes retrieved chunks to documents and to specific spans. Offsets are the universal join key — any strategy can be evaluated against any retrieval system because chunk identity is `(source_doc_id, start_offset, end_offset)`. Without offsets the matrix has to re-tokenize, which costs CPU and risks tokenization drift.

**Alternatives considered:**
- Chunk id only — rejected: #3 can't attribute retrievals without re-tokenizing.
- Separate offset index keyed by chunk id — rejected: same problem with extra indirection.

**Reversibility:** Cheap.

**Related issues:** #2, #3

## D-006 — Late chunking returns `(Chunk, vector)` pairs; other strategies return `Chunk` only (2026-05-15)
**Decision:** `LateChunkingStrategy.chunk_with_vectors()` returns `LateChunk` (chunk + vector); the four non-late strategies return `Chunk` (no vector).

**Why:** Late chunking's defining property is that each chunk's vector is derived from *document-level* context, not from the chunk text in isolation. The caller can't recompute the vector by `embedder.embed(chunk.text)` — that's the non-late path. So late chunking has to expose the vector alongside the chunk, otherwise the document-level signal is lost. The other four don't need this: their chunks' vectors are computed by the caller via `embedder.embed(chunk.text)`. Forcing all five to return vectors would waste compute for four-of-five.

**Alternatives considered:**
- Late chunking returns chunks only — rejected: loses document-level signal, defeats the strategy.
- All strategies return `(chunk, vector)` — rejected: wasted compute, forces `embedder` arg on strategies that don't need it.

**Reversibility:** Cheap. Both shapes coexist via `LateChunk`/`Chunk`.

**Related issues:** #2, #3
