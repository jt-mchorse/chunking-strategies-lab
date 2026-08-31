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

## D-007 — Metrics are pure functions, no SQLite (2026-05-16)
**Decision:** `evaluate_strategy(...)` is a pure function over `(Strategy, corpus, queries, embedder)` returning a `RetrievalRun`. The runner script writes one JSON per strategy + a markdown summary; nothing else is persisted. Same shape as `llm-eval-harness`'s `diff-json` (D-010 there): CI runners are ephemeral, the artifact-URL is the deployment story, and one current-vs-baseline JSON-pair comparison covers the use case.

**Why:** This repo is a *lab*, not a service. Operators run the matrix manually when they want fresh numbers and commit the results as artifacts; nothing about that flow needs a relational store. The simpler shape also makes the runner easier to fork — anyone can swap in their own corpus + queries + embedder and re-run without learning about SQLite migrations.

**Alternatives considered:**
- Persist runs to SQLite — rejected: see above; no use case.
- Ship the DB as a workflow artifact — rejected: same as above, plus storage cost.
- Accumulate results in a module global — rejected: thread-unsafe, hidden state, bad ergonomics for callers.

**Reversibility:** Cheap. The pure functions are easy to wrap in a persistence layer later if a use case shows up.

**Related issues:** #3

## D-008 — Snippet-hit@k is the answer-faithfulness proxy here (2026-05-16)
**Decision:** The "answer faithfulness on the downstream RAG task" the issue body mentions is implemented as **snippet-hit@k**: was the expected snippet substring present in the concatenated text of the top-k retrieved chunks? The metric is structural (substring match), not semantic (LLM judge).

**Why:** A *semantic* faithfulness judge belongs in `llm-eval-harness`, not here — and pulling it in would force this repo to take an eval-harness dep. The structural proxy catches the failure mode this lab cares most about: strategies that fragment the relevant passage across chunks, so the retriever finds the right *document* but no individual chunk has the answer in full. That's the whole "chunking strategy choice matters" thesis the lab exists to demonstrate. Strategies whose chunks miss the expected snippet across all top-k chunks lose on this metric; that's the right kind of loss to be sensitive to here.

**Alternatives considered:**
- LLM-judge faithfulness in this layer — rejected: adds an eval-harness dep + an API call per query, defeats the dep-free CI path.
- No faithfulness metric at all — rejected: recall@k alone misses the fragmentation failure mode.

**Reversibility:** Cheap. Adding an LLM-judge metric is additive — just a new field on `RetrievalRun`.

**Related issues:** #3

## D-009 — `RetrievalRun` carries `wall_clock_ms`, measured by `evaluate_strategy`; default `0.0` for backward compat (2026-05-17)
**Decision:** Add a `wall_clock_ms: float = 0.0` field to `RetrievalRun`. `evaluate_strategy` measures the full chunk + embed + retrieve pipeline with `time.perf_counter()` and reports the total. The default of `0.0` preserves backward compatibility — `RetrievalRun` instances built from older code paths or pre-D-009 JSONs still construct cleanly; the comparison notebook reads the field via `.get("wall_clock_ms", 0.0)` so older JSONs render as zero-latency rather than crashing.

**Why:** Issue #4's third acceptance bullet is a latency chart. `evaluate_strategy` is the only place with end-to-end visibility into the chunk + embed + retrieve pipeline; pushing the measurement up to a caller would require the caller to know which strategies use blended document vectors versus chunk-text embeddings (D-006) and would couple chart consumers to strategy internals. Default `0.0` is a deliberate choice — a missing/zero number is the right signal for "not measured", not a fake number that would corrupt a chart.

**Alternatives considered:**
- Time each phase separately (chunk, embed, retrieve) — rejected: more surface than the chart needs; the consumer cares about total cost, and phase timings can be added later as an additive field.
- Time only the chunking step — rejected: chunking is fast; the *embedding* cost dominates and varies per chunk count, which is exactly what the chart needs to show.
- Leave timing to the caller — rejected: couples chart consumers to strategy internals (late-chunking vs. standard) and loses the standardized number that makes cross-strategy comparison meaningful.

**Reversibility:** Cheap. One field, default `0.0`, additive on the JSON shape.

**Related issues:** #4

## D-010 — `[notebook]` extra (matplotlib + jupyter + nbformat); parallel to D-003's `[sbert]` pattern (2026-05-17)
**Decision:** Ship the comparison notebook behind an optional `[notebook]` extra that pulls in `matplotlib>=3.8`, `jupyter>=1.0`, and `nbformat>=5.0`. The matrix runner stays dep-free. The notebook tests use `pytest.importorskip("nbformat")` so the base CI matrix without extras still passes.

**Why:** Same posture as D-003 (`[sbert]` for MiniLM). The lab's core deliverable is the matrix runner + its results; the notebook is an optional visualization surface for operators who want to render the comparison. Forcing matplotlib + jupyter into the base install would pull in numpy, ipykernel, and a dozen indirect deps on every CI run, slowing the dep-free path that the rest of the portfolio expects. Splitting matplotlib and jupyter into separate extras would be over-engineering for a notebook that needs both.

**Alternatives considered:**
- Matplotlib in the base install — rejected: breaks the dep-free default, slows CI, pulls in numpy + fonttools transitively on every run.
- Render charts as static PNGs via a script — rejected: loses the interactive notebook artifact that issue #4 explicitly asks for; the markdown + chart interleaving is the readable form.
- Separate `[plot]` and `[jupyter]` extras — rejected: needless split; the notebook needs both, and the small overhead of installing jupyter when only matplotlib is needed isn't worth the extras-list complexity.

**Reversibility:** Cheap. One line in `pyproject.toml`. The notebook file is committed and works as long as the operator has the listed packages installed by any means.

**Related issues:** #4

## D-011 — Enforce late-chunking embedder consistency at the runner

- **Date.** 2026-05-22
- **Decision.** `evaluate_strategy` validates that, when the strategy is a `LateChunkingStrategy`, the strategy's own embedder and the runner's `embedder` report the same `model_name`. If they disagree, the runner raises `ValueError` immediately.
- **Why.** The constraint was documented at length in `_materialize_vectors`'s docstring ("the runner doesn't enforce this; it's a documented constraint of the late-chunking pattern"), but a documented constraint that doesn't fail loud is a footgun. Someone calling `LateChunkingStrategy(embedder=HashEmbedder())` paired with `evaluate_strategy(..., embedder=MiniLMEmbedder())` got a recall@k curve out the other side that *looked* plausible but was numerically meaningless — the chunk vectors lived in hash-space, the query vectors lived in MiniLM-space, cosine was noise. For a repo whose pitch is "the strategy choice is the only variable", a silent numerical-quality bug in one of the five strategies is exactly what undermines that pitch. Closes #19.
- **Alternatives considered.**
  - *Keep the documented-only status quo.* Rejected: documented-only constraints fail silently in operator code; the credibility cost is too high for a research-flavored repo.
  - *Auto-infer the strategy's embedder from `evaluate_strategy(embedder=...)`.* Rejected: the strategy constructor takes an embedder for a reason — it's part of the late-chunking surface contract (D-006). Silently rebuilding the strategy hides the failure rather than surfacing it.
  - *Identity check (`strategy.embedder is embedder`).* Rejected: two separate `HashEmbedder()` instances are not identity-equal but are functionally identical (both report `model_name="HashEmbedder"`); identity is too strict.
- **Reversibility.** Cheap. A caller who deliberately wants mismatched spaces can be unblocked by adding an explicit `allow_mismatch=True` flag to `evaluate_strategy` — but YAGNI for now; this defaults to safe.

## D-012 — Atomic-write helper lives in `chunking_lab/io_utils.py` (2026-05-26)
**Decision:** Atomic-write helper for this repo lives at `chunking_lab/io_utils.py`, exposing `atomic_write_text(path, text, encoding="utf-8")`. Matches the 2026-05-26 portfolio atomic-write arc.

**Why:** Two production write sites in `scripts/run_matrix.py` needed atomicity. Particularly nasty failure mode on canonical-fixture writes: a half-written `canonical__<strategy>.json` either fails the snapshot test loudly or, worse, gets committed and silently changes the published numbers. Package-level placement matches the four prior siblings (`rag_kit/io_utils`, `eval_harness/io_utils`, `emb_shootout/io_utils`, `async_pipelines/io_utils`) and centralizes the test surface.

**Alternatives considered:**
- File-private helper in `run_matrix.py` — rejected; the pattern is shared across the portfolio and other modules could need the helper later.
- Inline the pattern at the call sites — rejected; ~25 lines × 2 sites that could drift; no central seam.

**Reversibility:** Cheap.

**Related issues:** #33

---

## D-013 — A non-strict `mypy` gate over `chunking_lab`, in CI and as a test

**Date:** 2026-08-24

**Decision.** Adopt the non-strict `mypy` baseline gate: `python_version =
"3.11"`, `files = ["chunking_lab"]`, `warn_unused_ignores`,
`warn_redundant_casts`, no blanket `ignore_missing_imports`. It runs in the CI
lint job and again as `tests/test_mypy_clean.py`, both invoking a bare `mypy` so
they read exactly the `[tool.mypy]` block in `pyproject.toml`.

**Why.** The rationale here is deliberately *not* the one the two sibling repos
give. `llm-eval-harness` (D-016) and `llm-cost-optimizer` (D-014) both justify
their gate by shipping a `py.typed` marker — their annotations are visible to
downstream type-checkers, so drift is a broken contract with a consumer.
`chunking_lab` ships no such marker, so that argument doesn't apply.

What applies instead is **latent green** rot: the annotations were already
written, nothing machine-checked them, and the code was correct today with
nothing to say when it stopped being. #164 is the proof rather than the
hypothesis — two `no-redef` errors sat in
`chunking_lab/strategies/recursive.py` unnoticed, on a repo whose CI was green
the entire time, because no gate ran. They were found by someone running `mypy`
by hand while working on an unrelated issue. That is not a discovery mechanism.

Wiring it in *both* places is the substance rather than belt-and-braces. A CI
step alone means a developer never sees the failure until after pushing. A test
alone means a future change to the pytest scope silently bypasses it. And
invoking a bare `mypy` in both — rather than passing a file list — is what keeps
the test, the CI step, and a developer's terminal checking the same thing; a
test that passed its own file list would be testing a scope nothing else uses.

Two smaller calls fell out of it. The per-module override for
`sentence_transformers` (D-003's `[sbert]` extra) replaced an inline
`# type: ignore[import-not-found]` in `embedder.py`, because with
`warn_unused_ignores` on that inline ignore would itself become an error the
moment someone installs the extra — an override is stable across both states,
an inline ignore is a time bomb. And the absence of a blanket
`ignore_missing_imports` is deliberate: it would silence a *typo'd* import just
as effectively as an optional one.

**Scope limitation, stated rather than papered over.** `scripts/` and `tests/`
are outside the gate. `mypy chunking_lab scripts tests` fails before checking
anything with `Source file found twice under different module names:
"run_matrix" and "scripts.run_matrix"`, which needs an `__init__.py` or
`--explicit-package-bases` decision of its own. Tracked as a separate issue.

**Alternatives considered.**
- *Full strict mode now* (`disallow_untyped_defs` et al.) — rejected; both
  siblings took the baseline-first posture and tightening is a follow-up with
  its own churn.
- *Blanket `ignore_missing_imports`* — rejected; silences real mistakes.
- *Keep the inline `# type: ignore` instead of a per-module override* — rejected;
  goes stale the moment the extra is installed.
- *Just fix the two `no-redef` errors and move on* — rejected; that closes the
  instance and leaves the class open, which is exactly what happened last time.
- *CI step only, or test only* — rejected for the reasons above.

**Reversibility:** Cheap. A config block, a CI line, and a test file.

**Related issues:** #164

## D-014 — the mypy gate covers `scripts/`, and the collision it was blocked on was a real bug

**Date.** 2026-08-28 · **Issue.** #165 · **Reversibility.** Cheap.

**Decision.** Extend D-013's `mypy` gate to `scripts/`, using `mypy_path = "."`
plus `explicit_package_bases = true`, and normalize the test suite to import
`scripts/run_matrix.py` under a single module name. Do not add
`scripts/__init__.py`.

**Why.** D-013 left `scripts/` out because `mypy chunking_lab scripts` did not
merely report errors there — it stopped before checking anything at all, with
"Source file found twice under different module names". So the repo did not
know whether `scripts/` was clean, only that it was unchecked, and
`scripts/run_matrix.py` is not a throwaway: it writes the `results/*.json` that
the README's comparison table and the notebook are derived from.

Investigating turned the premise around. That error was a true finding, not a
layout quirk to configure away. The test suite really did import that one file
both ways — four bare `from run_matrix import …` sites in `test_metrics.py`
after a `sys.path` insert, and `scripts.run_matrix` everywhere else. Python
treats those as unrelated modules: the file's body executes twice, and a
`monkeypatch.setattr` on one copy is invisible to the other. Nothing was
failing, but only because each group of tests happened to patch and call the
same copy — a property of which tests exist, not one anyone had enforced.

So the fix has two halves and neither alone is enough. The configuration keys
make the *mapping* unambiguous; normalizing the imports removes the *cause*.
Once the gate could run it found exactly one real error — a `# type: ignore`
that had been dead all along, on a name that is defined unconditionally — which
was removed rather than silenced. Notably, `pyproject.toml`'s own comment
already argued against inline ignores for precisely this reason; the script was
simply out of scope, so nothing checked it.

**Alternatives considered.** `scripts/__init__.py` was the other option the
issue listed: it changes how `python scripts/run_matrix.py` resolves, and it
would not have removed the duplicate module. Either half of the fix on its own
leaves the other problem standing. Extending the gate to `tests/` in the same
change was rejected as too large — it reports 12 errors and one of them needs a
new dev dependency — but it is now measured rather than unknown, which was the
issue's actual complaint, and tracked separately.

**Portfolio note.** The same probe across the six repos with a `scripts/`
directory found the identical collision in `llm-cost-optimizer`
(`_io` vs `scripts._io`). Filed there rather than fixed here.
