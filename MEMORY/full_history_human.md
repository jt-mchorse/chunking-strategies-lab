# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---

## 2026-05-14 — Issue #1: pin the shared substrate
**Duration:** ~50 min · **Branch:** `session/2026-05-14-1435-issue-01`

- Committed five hand-authored Markdown technical articles to
  `data/corpus/` (MIT) covering HNSW, RRF, Anthropic prompt caching,
  evaluation-harness design, and async LLM pipelines. Multi-paragraph,
  structurally varied — chunking strategies will visibly differentiate.
- Committed 12 verbatim-snippet queries to `data/queries.jsonl`. Each
  query's `expected_snippet` is literal text from its `expected_doc`,
  and a test (`test_each_expected_snippet_is_verbatim_in_its_document`)
  is the gate that prevents the corpus and queries from drifting apart.
- Pinned `sentence-transformers/all-MiniLM-L6-v2` as
  `CANONICAL_EMBEDDING_MODEL`. Ships an `Embedder` Protocol with a
  dep-free `HashEmbedder` (matched to 384-d so strategies tested under
  it transfer to the real embedder) and a `MiniLMEmbedder` adapter
  behind the `[sbert]` optional extra.
- `docs/setup.md` documents the full substrate + the change protocol
  (any corpus/query/model edit invalidates downstream numbers and
  requires a deliberate revisit of D-002).
- 18 tests, 92% coverage on `chunking_lab/`. Real CI (ruff +
  pytest matrix py3.11/3.12).
- Two decisions recorded: D-002 (the substrate pin itself, the issue's
  explicit acceptance criterion) and D-003 (real embedder behind the
  optional extra, dep-free reference for CI).

**Why this work, this session:** Issue #1 is the gate for #2 and #3 —
without a pinned substrate, the strategy comparison numbers in #3
aren't reproducible.

**Open questions / blockers:** None.

**Next session:** Issue #2 — the five chunking strategies, each
exposing `chunk(text, **opts) -> list[Chunk]` against this substrate.

## 2026-05-15 — Issue #2: Implement 5 chunking strategies
**Duration:** ~75 min · **Branch:** `session/2026-05-15-1810-issue-02`

- Shipped `chunking_lab/strategies/`: shared `Strategy` Protocol (D-004), `Chunk` dataclass with offsets (D-005), `LateChunk` for the late-chunking output shape (D-006).
- Five standalone strategy modules: `fixed.py` (sliding window + overlap), `recursive.py` (separator-hierarchy split with greedy merge), `semantic.py` (embedding-boundary cosine-peak split with min/max chunk caps), `late.py` (chunks + document-blended vectors via configurable `document_weight`), `structure.py` (markdown-heading-bounded sections with title metadata + per-section size cap).
- 29 new hermetic tests + 18 from #1 = 47/47 passing. Acceptance criteria from #2 all met: common interface (`Strategy.chunk`), per-strategy unit tests, runtime sanity-check across the 5-doc corpus.
- `__init__.py` re-exports all new strategy classes alongside the substrate.
- README "Strategies (#2 · this PR)" section with the 5-row table + a use snippet.

**Why this work, this session:** Strategies are the central deliverable of this repo; without them the substrate from #1 is just static data. Locking the `Strategy` Protocol shape and the `Chunk` offset contract now lets the metrics matrix (#3) iterate strategies uniformly without re-tokenizing.

**Open questions / blockers:** None. Real per-strategy quality numbers (recall@k, faithfulness) are deferred to #3 since they need real-embedder runs and the metrics-matrix harness; this PR ships the runtime sanity-check (every strategy completes the corpus in under 5 seconds on CI).

**Next session:** Issue #3 (retrieval metrics matrix) is the natural sibling — iterates the strategies over the substrate and emits recall@k + faithfulness numbers per strategy.

## 2026-05-16 — Issue #3: Retrieval metrics matrix
**Duration:** ~40 min · **Branch:** `session/2026-05-16-0429-issue-3`

- Shipped `chunking_lab/metrics.py`: `evaluate_strategy(strategy, corpus, queries, embedder, ks)` returns a `RetrievalRun` with per-query results + recall@k and snippet-hit@k aggregates (D-007 — pure functions, no SQLite, same shape as llm-eval-harness's `diff-json`). Late-chunking is routed through `chunk_with_vectors()` so its blended document vectors are used directly; everything else embeds chunk text with the evaluator's embedder. Documented constraint: late-chunking's embedder must match the evaluator's embedder for the cosine scores to be meaningful.
- D-008 frames snippet-hit@k as the answer-faithfulness proxy: structural (substring match against retrieved chunks), not semantic (LLM judge). LLM-judge faithfulness belongs downstream in eval-harness; this layer ships the cheap, hermetic, dep-free proxy.
- `scripts/run_matrix.py` is the single-command runner: builds all 5 strategies, calls `evaluate_strategy` for each, writes `results/<timestamp>__<strategy>.json` per strategy plus `results/summary.md` aggregating the matrix. Defaults to `HashEmbedder` (dep-free, CI-safe); `--embedder minilm` opts into the real one via the `[sbert]` extra. The summary markdown carries an explicit disclosure when in CI mode: numbers reflect plumbing, not retrieval quality. Per the no-fabricated-benchmarks rule.
- 9 new tests in `tests/test_metrics.py`: required-fields presence, per-query shape correctness, recall-math against a forced-routing stub strategy, snippet-hit short-circuits on no-substring queries, empty-corpus edge case, late-chunking routes through its own vectors, JSON round-trip schema stability, run_matrix script writes one JSON per strategy + summary.md, summary.md contains the HashEmbedder disclosure when applicable. Suite total: 56/56 pass; ruff lint+format clean.
- README: new "Retrieval metrics matrix (#3 · this PR)" section with the single-command recipe + the honest CI-vs-real disclosure. Benchmarks section explicitly says no curve quoted until `docs/benchmarks.md` exists from a real run.

**Why this work, this session:** #3 is the last load-bearing piece of chunking-strategies-lab. With it, the repo's v0.1 story is complete: substrate (#1) + 5 strategies (#2) + matrix evaluator (#3) + honest framing about what numbers are real and which require operator action.

**Open questions / blockers:** None. Real recall numbers per strategy require an operator's `[sbert]` install + a few-minute run; nothing engineering can do about that without burning carbon-irrelevant API budget on a model that runs locally for free anyway.

**Next session:** Either #4 if filed (no current open issue beyond #3) or move to a different repo.
