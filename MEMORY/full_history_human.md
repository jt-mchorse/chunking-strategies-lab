# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---

## 2026-05-19 — Issue #13: snapshot test for comparison.ipynb cell sources
**Duration:** ~30 min · **Branch:** `session/2026-05-19-1520-issue-13` · **PR:** #14

- Refactored `notebooks/_build_notebook.py` to expose `build_notebook() -> NotebookNode` as a pure function; `main()` is now `build_notebook()` + `nbformat.write(...)`. The refactor makes the build script callable from tests without writing tempfiles or shelling out.
- Added `tests/test_build_notebook_snapshot.py` (11 tests, parametrized per cell): reads the committed `comparison.ipynb`, builds a fresh `NotebookNode`, asserts cell count + each cell's `(cell_type, source.rstrip())` matches in order. Outputs, `execution_count`, and notebook metadata are intentionally ignored — `test_notebook_executed_outputs_exist` already covers output presence, and pixel content + env-dependent metadata aren't worth locking. `pytest.importorskip("nbformat")` keeps base CI green when the `[notebook]` extra isn't installed.
- Drive-by fix: updated `_LOAD_CELL`'s final `print()` call to the multi-line shape already on disk in `comparison.ipynb`. The committed notebook had been ruff-formatted after the original build, so the snapshot test caught this drift on its first run; rather than reformat the notebook (which would also strip executed outputs), I updated the source script to match.
- Tamper-verified by editing `_INTRO` to add ` TAMPER`; cell-0 assertion fired with the regen hint before any rebuild, then reverted to green.

**Why this work, this session:** Third snapshot test in this repo's lineage after `test_summary_snapshot.py` and `test_metrics.py`'s fixture-locked rows. The portfolio's "no fabricated benchmarks" rule generalizes to "no silent drift between an authoring artifact and its rendered output"; the `_build_notebook.py` → `comparison.ipynb` pipeline was the last unguarded surface in this repo. Also surfaced and corrected a real drift the snapshot caught on first run.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Both this repo's authoring surfaces (`results/summary.md` renderer, `_build_notebook.py` build script) are now drift-locked against their committed artifacts. Continue the multi-issue loop into the next portfolio repo (vector-search-at-scale is next in §8, but it has only a low-priority demo issue — pick another zero-issue repo or work the demo issue's capture-script subset).

## 2026-05-19 — Issue #11 (cont.): un-block PR #12 by tracking canonical fixtures
**Duration:** ~30 min · **Branch:** `session/2026-05-18-1939-issue-11`

- Root cause of PR #12's CI failure (7 failed in `tests/test_summary_snapshot.py`) was the `results/` line in `.gitignore` — the 5 strategy JSONs and `summary.md` that the snapshot test reads were never on the branch. Test ran in CI, got an empty `results/` directory, and failed with "Missing committed result JSONs for strategies: [...]. Found: []."
- Fix: `.gitignore` now ignores `results/*` but un-ignores `results/summary.md` and `results/canonical__*.json`. Renamed the existing five files from a timestamped prefix (`20260516T162215__`) to `canonical__` so they're caught by the negation rule and tracked from now on. The snapshot test's `*.json` glob and `split("__")` parsing both still work.
- Added `--canonical-out` flag to `scripts/run_matrix.py` so the canonical fixtures can be refreshed with one command. Default behavior (no flag) writes timestamp-prefixed JSONs **and** a timestamp-prefixed summary so the regen scratch can't accidentally clobber the tracked `summary.md` (a footgun I noticed during testing — fixed it before committing).
- Updated the two integration tests in `test_metrics.py` that exercised the old default behavior to use `--canonical-out`, and added one new test asserting the default path writes only timestamped scratch (zero tracked filenames).
- README Quickstart now documents both paths.

**Why this work, this session:** PR #12 is the existing work for #11, blocked on CI. Fixing on the same branch keeps the snapshot test and its fixtures in one PR.

**Open questions / blockers:** None. CI rerun should be green now.

**Next session:** Whatever Phase A selection produces next; this branch should merge once CI lands.

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

## 2026-05-17 — Issue #4: Comparison notebook + wall-clock metric
**Duration:** ~70 min · **Branch:** `session/2026-05-16-2319-issue-4`

- Extended `RetrievalRun` with a `wall_clock_ms` field (D-009): `evaluate_strategy` now wraps the full chunk + embed + retrieve pipeline in `time.perf_counter()` and reports total milliseconds. The latency a downstream consumer sees if they swap in this strategy on this corpus + embedder. Default `0.0` keeps backward compat with pre-D-009 JSONs. `run_matrix.py` surfaces the column in `summary.md` and the per-strategy log line.
- Deleted the old `20260515*` results JSONs (no wall_clock_ms) and re-ran the matrix, committing fresh `20260516T162215__*.json` files with the new field. Real numbers from the HashEmbedder run: fixed-size 20ms, recursive 19ms, semantic 75ms (3× the others — 84 chunks vs. ~30), late-chunking 22ms, structure-aware 19ms. These latency numbers are *real* because chunking + cosine retrieval is real work even with HashEmbedder; only the quality numbers carry the embedder caveat.
- Shipped `notebooks/comparison.ipynb`, built programmatically from `notebooks/_build_notebook.py` (uses `nbformat` + `textwrap.dedent` so the Python source stays reviewable, and the `.ipynb` JSON is regenerable). Three matplotlib charts: recall@k grouped bars across strategies (chart 1), snippet-hit@k grouped bars (chart 2 — the D-008 answer-faithfulness proxy), wall-clock-ms single-bar chart with value labels (chart 3). Markdown cells frame the embedder caveat above the charts and the honest takeaways below. The committed notebook ships with executed outputs (rendered chart PNGs inline) via `jupyter nbconvert --to notebook --inplace --execute` so reviewers see the artifact without installing jupyter.
- D-010 records the `[notebook]` extra: `matplotlib + jupyter + nbformat`, parallel to D-003's `[sbert]`. Base install stays dep-free; the matrix runner works without the notebook; `test_notebook.py` uses `pytest.importorskip("nbformat")` so base CI without extras still passes.
- 7 new tests: 1 in `tests/test_metrics.py` asserting `wall_clock_ms` is recorded and positive (and round-trips through `to_json`), and 6 in `tests/test_notebook.py` covering committed-file presence, nbformat validation, ≥3 `plt.show()` cells (acceptance criterion), takeaways + honest-numbers disclosure, results-loader correctness against the real `results/` directory, and presence of executed outputs on the committed notebook. Suite 63/63 pass; ruff lint+format clean.
- README "Comparison notebook" subsection under Benchmarks/Results with the embedded link + three-step regenerate command (install extras, re-run matrix, re-execute notebook).

**Why this work, this session:** Issue #4 was the only open issue in this repo and lives naturally on top of #3's `RetrievalRun` shape. Closing it with a programmatically-built notebook (rather than a hand-edited `.ipynb`) means the comparison surface is itself versioned as code — future strategy additions or metric changes regenerate the notebook deterministically. Adding `wall_clock_ms` to `RetrievalRun` was the smallest possible change that satisfied the third-chart acceptance bullet without sprawl.

**Open questions / blockers:** None. Real-quality charts wait for an operator's `[sbert]` install + `--embedder minilm` run; the notebook automatically picks up new results on the next nbconvert.

**Next session:** Loop to a different portfolio repo per the multi-issue session prompt. The natural next is `vector-search-at-scale` (next in build sequence among untouched repos) or `python-async-llm-pipelines`.

## 2026-05-18 — Issue #9: Architecture doc covers all 4 shipped layers
**Duration:** ~20 min · **Branch:** `session/2026-05-18-1540-issue-9` · **PR:** [#10](https://github.com/jt-mchorse/chunking-strategies-lab/pull/10) (ready)

- Rewrote `docs/architecture.md` so every shipped layer (#1 substrate, #2 five strategies, #3 metrics matrix, #4 comparison notebook) has its own section with prose, a mermaid diagram of its own slice, D-NNN references back to MEMORY (D-002 through D-010), and a "composes with" line. Integrated diagram at the top is all-green and now includes the notebook + the wall-clock-ms axis (D-009).
- README Architecture stub got a one-line update so it stops saying "shipped vs pending" — points at the now-real doc.
- Mermaid hygiene: every label with parens is fully double-quoted, matching the lint applied to llm-cost-optimizer (#13) and rag-production-kit (#17) architecture docs earlier this session.

**Why this work, this session:** Every original priority:high issue is closed; the comparison notebook (#4) shipped on 2026-05-17. The architecture doc was the most visible remaining §1 quality-bar gap — it still labelled #2, #3, and the strategies as :::pending. Filling that gap is the cleanest move toward v0.1.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Pick up the next zero-open-issue repo in §8 build sequence (python-async-llm-pipelines), or wait for the in-flight architecture-doc PRs across the portfolio to merge first.

## 2026-05-18 — Issue #11: snapshot test for `results/summary.md`
**Duration:** ~15 min · **Branch:** `session/2026-05-18-1939-issue-11`

- Added `tests/test_summary_snapshot.py` (7 tests). The headline test loads all five committed `results/*.json` files, reconstructs `RetrievalRun` instances in strategy-build order (fixed-size, recursive, semantic, late-chunking, structure-aware), calls `scripts.run_matrix._render_summary`, and asserts byte equality with the committed `summary.md`. Per-strategy parametrized loader tests guard against schema drift in `RetrievalRun`.
- `summary.md` was previously generated by `_render_summary` but not test-locked — a future tweak to the renderer (column order, decimal places, formatting) would silently desync the committed table from what the script produces.
- Failure messages name the regen command. Verified the failure path by tampering one cell (fixed-size recall@1 `0.333` → `0.999`); the snapshot fired with the regen hint visible.
- Total tests: 70 (was 63).

**Why this work, this session:** Fourth snapshot test landed today across the portfolio (cost-optimizer for `docs/savings.{json,md}` + README, prompt-regression-suite for `docs/regression_demo.html`, rag-kit for the README rewriter table, now chunking-lab for `results/summary.md`). Handoff §10 commits the portfolio to "no fabricated benchmarks"; snapshot tests are the structural enforcement.

**Open questions / blockers:** None — PR ready for review.

**Next session:** All four "benchmark/demo-output-vs-committed-doc" gaps surveyed this session are now closed across the portfolio. Future portfolio sessions can focus on shipping the still-pending live-API integrations or layered features.

## 2026-05-20 — Issue #15: lock chunking_lab public surface
**Duration:** ~20 min · **Branch:** `session/2026-05-20-0321-issue-15`

- Added `tests/test_public_surface.py` (8 test items) and `__version__ = "0.0.1"` on `chunking_lab`. Five axes: semver, all-bound-non-None, all-matches-imports (relative-import AST filter), README quickstart union across both quoted snippets (lines 74 and 94 → six unique names), and one anchor per re-exported submodule. `metrics` deliberately excluded from the anchor list — dotted-path-only by design.
- Tamper-verified three axes: bad version, drop `"Document"` from `__all__`, alias-rename `CANONICAL_EMBEDDING_MODEL as EM_MODEL` (fires bound-and-non-none + README-quickstart simultaneously).
- Full suite 90/90 (was 82; +8 new).

**Why this work, this session:** Sixth strike of the portfolio-wide public-surface hygiene pattern. The README quotes two separate import snippets and the strategy class table (5 names); the union of all three is fully covered by `__all__` ↔ imports + the six-name README test + the `strategies` anchor.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Apply the pattern to `python-async-llm-pipelines` and the Python example in `mcp-server-cookbook`.

## 2026-05-21 — Issue #17: 60-second demo capture script
**Duration:** ~22 min · **Branch:** `session/2026-05-21-1916-issue-17` · **PR:** #18

- Added `scripts/capture_demo.sh` driving the two surfaces — `python scripts/run_matrix.py --results-dir <tmp>` walks all five strategies against the pinned corpus + queries with `HashEmbedder` and prints `recall@5 / snippet-hit@5 / wall_clock` per strategy live, then the same run's `<ts>__summary.md` is cat'd. Per-run tempdir trapped on EXIT/INT/TERM. Full runtime is ~0.2s — fits 60 seconds with banners and recording pauses.
- Added `tests/test_capture_demo_smoke.py` (3 tests) that runs the script with `PACE=0` in CI and asserts: all five strategy names appear in the matrix-step output; `recall@5=` / `snippet-hit@5=` / `wall_clock=` keys each appear ≥5 times; the summary markdown header signature matches what `test_summary_snapshot.py` locks separately; every strategy has a data row in the rendered summary; script exists and is executable.
- README "Demo" section replaces the `*60-second demo pending — depends on issues #2 and #3.*` placeholder with one paragraph framing the two surfaces plus the HashEmbedder-vs-MiniLM distinction. 93/93 tests pass, ruff clean.

**Why this work, this session:** Seventh repo (out of twelve) to land the `scripts/capture_demo.sh` pattern this week — closing the last unchecked item on the six-item v0.1 quality bar for this repo. The repo had zero open issues at session start, so the issue itself was filed against the §2 quality-bar gap before any code.

**Open questions / blockers:** None. The capture is hermetic so it can be re-recorded any time without coordination.

**Next session:** Continue the multi-issue loop on the remaining stale repos (vector-search-at-scale, python-async-llm-pipelines, agent-orchestration-platform, etc.). The `[demo]` capture issues already filed on those repos at `priority:low` are the obvious next round.

## 2026-05-22 — Make the silent late-chunking footgun fail loud (#19, D-011)

**Duration:** ~30 min. **Issue:** [#19](https://github.com/jt-mchorse/chunking-strategies-lab/issues/19). **PR:** TBD.

`evaluate_strategy` shipped with a documented but unenforced constraint: when the strategy is `LateChunkingStrategy`, the strategy's own embedder (used to compute chunk vectors with document-level blending) and the runner's `embedder` (used to embed queries) must produce vectors in the same space. The metrics.py docstring even warned about it — but a constraint that fails silently is a footgun, not a constraint. The example failure mode is `LateChunkingStrategy(embedder=HashEmbedder())` paired with `evaluate_strategy(..., embedder=MiniLMEmbedder())`: cosine between hash-space chunk vectors and MiniLM-space query vectors is meaningless, so the recall@k curve out the other side *looked* plausible but was garbage. For a repo whose pitch is "every published shootout hides something; this one doesn't", a silent numerical-quality bug in one of the five strategies is the credibility leak.

The fix is small and conservative. `evaluate_strategy` now calls `_check_late_chunking_embedder_consistency` before any work, which compares `_embedder_model_name(strategy.embedder)` to `_embedder_model_name(runner_embedder)` and raises a clear `ValueError` on mismatch. The check uses model name, not Python identity — so two `HashEmbedder()` instances both report `HashEmbedder` and pass through correctly; only a real embedder-space disagreement trips it. Four tests cover the surface: two-`HashEmbedder` pass case, mismatched-name fail case, non-late-strategy unaffected, and the error message names both `LateChunkingStrategy` and `D-011` so the operator finds the rationale fast.

Why prioritized: the existing comparison notebook and the matrix runner both happen to construct the strategy with the runner's embedder, so the bug was latent — but it's exactly the kind of thing an outside reader would step on the first time they wired up the lab against a real embedder, which is the explicit handoff path for honest numbers (`pip install -e '.[sbert]'`). Open questions / followups: an explicit `allow_mismatch=True` opt-out would let a curious caller deliberately probe what happens — filed as a backlog idea but not opened as an issue; YAGNI for now.

## 2026-05-23 — README drift fix + snapshot lock (#23)

**Duration:** ~25 min. **Issue:** [#23](https://github.com/jt-mchorse/chunking-strategies-lab/issues/23). **PR:** [#24](https://github.com/jt-mchorse/chunking-strategies-lab/pull/24).

This was the **last portfolio repo without a README snapshot/hygiene lock** — verified against 11 sister repos this session. Authoring the lock surfaced four real drift sites in README.md: a pre-shipping `This PR pins the substrate...will share` paragraph at L19; an architecture-doc-summary line at L41 citing `D-002…D-010` (D-011 added 2026-05-22 was omitted); and two `## Section (#N · this PR)` headers at L59 and L109.

Rewrote the four sites. The lock test contributes one novel pattern to the portfolio: an **active-decision-range upper-bound test** that anchors the README's `D-002…D-NNN` citation to the highest non-superseded `D-NNN` in `MEMORY/core_decisions_ai.md`. A future D-012 landing without the README updating fails this test loud with a regen hint. Tamper-verified three ways.

**Why this work, this session:** Sixth issue in the night sweep. The combination of arch-doc lock (PR #22) + this README lock brings this repo to the same hygiene posture as the rest of the portfolio.

**Open questions / blockers:** none. **Portfolio-wide:** architecture-doc lock and README lock are both now at 12-of-12 coverage.

## 2026-05-24 — Issue #25: `run_matrix.py --strategy` filter

**Duration:** ~20 min. **Issue:** [#25](https://github.com/jt-mchorse/chunking-strategies-lab/issues/25). **Branch:** `session/2026-05-24-0344-issue-25`.

`scripts/run_matrix.py` always ran all five strategies and always wrote five JSONs. There was no way to iterate on one strategy (e.g. tuning `SemanticBoundaryStrategy`'s threshold) without re-running and re-writing the other four — and under `--canonical-out`, a single-strategy iterative run would silently clobber the other four canonical files that `tests/test_summary_snapshot.py` pins.

`--strategy fixed-size|recursive|semantic|late-chunking|structure-aware` filters the strategies list before the eval loop. The names match each strategy's `.name` attribute and the `canonical__<name>.json` file convention so there's only one identifier in the surface area. When the filter is set, the script skips the summary.md write entirely — a single-row summary would invalidate the snapshot lock and would be misleading next to the canonical five-row aggregate. That "no summary when filtered" rule is what keeps the filter from being a one-way decision.

Six new tests pin the contract: filter writes only the chosen strategy's JSON, other canonical files in the same dir are not clobbered (test pre-seeds them with sentinel bytes and re-reads after the run), no summary lands in either the default or `--canonical-out` mode, the unfiltered path still writes all five + summary (regression guard), and `argparse`'s `choices=` rejects unknown strategy names with exit 2.

**Why this work, this session:** Sixth issue in the night-session multi-issue loop. Parallel to `rag-production-kit` #32's `--suite` filter landed earlier this session — same dev-iteration shape, different orchestrator.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue to build-sequence #7 (`vector-search-at-scale`).

## 2026-05-25 — Issue #27: evaluate_strategy validates ks per-element at function entry
**Duration:** ~15 min · **Branch:** `session/2026-05-24-issue-27`

- `evaluate_strategy` at `chunking_lab/metrics.py:114` validated only that the `ks` *sequence* was non-empty implicitly via `max(ks) if ks else 5` — but non-positive elements flowed through `retrieved_docs[:k]` slicing at lines 170 and 172 without raising. `k=0` produced tautological `recall@0=0.0`; `k<0` silently miscounted via "all but the last N" slicing. **The wrong number, not an absent number** — same harm wording as the sister fix in `embedding-model-shootout` `run_sweep` (PR #28).
- Added an entry guard that raises `ValueError("ks must be non-empty")` for an empty sequence and `ValueError(f"every k in ks must be positive; got {sorted(bad_k)}")` for any non-positive elements, with all offenders collected in one pass so operators copy-paste the fix in canonical sorted form. Message shape matches the emb-shootout sister fix exactly so the two retrieval-comparator repos in the portfolio raise identically.
- Seven new tests in `tests/test_metrics.py` under a `#27` block: empty raises with `"ks must be non-empty"`; zero raises with `[0]`; negative raises with `[-1]`; mixed `(-3, 0, 5)` lists both bad values as `[-3, 0]` sorted ascending; parametrized positive acceptance over `(1,), (3, 5), (1, 3, 5, 10)` produces `recall_at_k` keys exactly equal to the input set. `_eval_with_ks(ks)` helper centralizes the fixture so each negative test only varies the field under test. Full suite 126/126 (was 119 after #25).

**Why this work, this session:** Direct mirror of `embedding-model-shootout` PR #28 shipped earlier this same day session. The two retrieval-comparator repos in the portfolio now defend their result-JSON shapes consistently against the silent-recall-corruption harm class. Fifth Phase B+C target after `llm-eval-harness` #40, `llm-cost-optimizer` #34, `rag-production-kit` #36, `embedding-model-shootout` #29.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the day-session loop. Build sequence #7 (`vector-search-at-scale`) is the natural next pickup if there's time before the 180-min cap.

## 2026-05-25 — Issue #29: strategy dataclasses + HashEmbedder isinstance(int) guards
**Duration:** ~25 min · **Branch:** `session/2026-05-24-issue-29`

- Five constructors validated `chunk_chars` / `overlap_chars` / `min/max_chunk_chars` / `dim` with sign-only checks. Non-int (float, NaN, fractional, bool) slipped through and failed deep in the chunking loop. The worst case: `overlap_chars = NaN` passed both sign-only checks and the `overlap >= chunk` comparison (NaN comparisons false), then `stride = chunk - NaN = NaN`, then `start += NaN`, then `while start < len(text)` is undefined for NaN — infinite-loop scenario in the worst case. Fractional `chunk_chars` silently truncated via slicing-int coercion, producing wrong-sized chunks. `HashEmbedder.dim` non-int reached the `% 8` check and raised with a misleading "must be a multiple of 8" message.
- Tightened `FixedSizeStrategy`, `LateChunkingStrategy`, `RecursiveStrategy`, `SemanticBoundaryStrategy.__post_init__` and `HashEmbedder.__init__` to require `isinstance(x, int)` (bool excluded explicitly since Python's bool subclasses int) before the existing sign comparisons. Existing message matchers ("chunk_chars", "overlap_chars", "dim") survive unchanged so no pre-existing test required updating.
- 35 new tests (7 parametrize tables × 5 bad values) in `tests/test_strategies.py` under a `#29` block: per-constructor per-field rejection over `[1.5, NaN, +Infinity, True, "5"]` + a boundary acceptance regression that proves the five-constructor valid-int construction still works. Test count 162.

**Why this work, this session:** Tenth Phase B+C target in the 360-min night session. Second PR in chunking-strategies-lab tonight; the first was via the Phase A fixup-merge of #28 (sign-only `ks` per-element validation in `evaluate_strategy`). The two together complete the constructor-input + comparator-input contract-tightening arc on the chunking surface.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop across remaining unvisited-tonight-for-second-iteration repos: `vector-search-at-scale`, `python-async-llm-pipelines`. Each had a Phase A fixup-merge today but no Phase B+C finiteness sweep yet.

## 2026-05-26 — Issue #31: StructureAwareStrategy completes the #29 sweep
**Duration:** ~25 min · **Branch:** `session/2026-05-25-2200-issue-31`

- `StructureAwareStrategy` was the only strategy constructor #29 missed. Its `__post_init__` at `chunking_lab/strategies/structure.py:36-40` used a range-only check on `max_heading_level` and a sign-only `<= 0` on `max_chunk_chars`. Brought both fields into the portfolio's `isinstance(int) + reject bool` pattern; the existing range and positive errors remain reachable (tests pin this).
- Silent failure modes closed: `max_heading_level=True` silently bound to `1` (chunker degraded to splitting only on top-level `#`, semantic bug with no error); `max_heading_level=2.0` silently bound as float; `max_chunk_chars=True/NaN/Inf/4000.5` silently bound, then surfaced as misleading internal-site errors from the FixedSizeStrategy section-too-large fallback path (#29's tightened FixedSize caught them, but the error pointed at the wrong site).
- Six new parametrize blocks in `tests/test_strategies.py` following the existing `_BAD_INT` pattern: two type matrices (`max_heading_level`, `max_chunk_chars`), two acceptance matrices over the documented ranges, and two preservation pins ensuring the original "must be in [1, 6]" and "must be positive" errors remain reachable for plain out-of-range ints. 28 new collected cases; full suite 162 → 190. Ruff clean (`structure.py` ran through `ruff format` to normalize the new long isinstance lines).

**Why this work, this session:** Third Phase B+C target in the 360-min night session, continuing the portfolio-wide validation sweep started in Phase A's four PR merges. Picked via build-sequence #6 among repos with un-swept constructors after `prompt-regression-suite#38` (Phase B+C target #2).

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop. `vector-search-at-scale` (build #7) had only one PR earlier today and may have un-swept constructors too; `python-async-llm-pipelines` (build #8) similarly.

## 2026-05-26 — Issue #33: Add `chunking_lab/io_utils.atomic_write_text`, route `run_matrix.py` through it
**Duration:** ~15 min · **Branch:** `session/2026-05-26-1944-issue-33`

Two production sites in `scripts/run_matrix.py` (per-strategy `RetrievalRun` JSON; markdown summary) used `Path.write_text`. The canonical-fixture write path was particularly load-bearing: a half-written `canonical__<strategy>.json` either fails the snapshot test loudly or gets committed and silently changes published numbers. New `chunking_lab/io_utils.py` matches the portfolio standard; both sites routed; 6 unit + 1 integration test added (suite 190 → 197). D-012 codifies the placement.

**Why this work, this session:** Fifth Phase B issue of today's DAY session. Portfolio atomic-write coverage now at 10 of 12 repos.

**Open questions / blockers:** none.

**Next session:** `vector-search-at-scale` (5 sites) is the last remaining repo. `nextjs-streaming-ai-patterns` has no on-disk write paths to harden, so completing vector-search-at-scale would saturate the portfolio at 12 of 12 atomic-write coverage.

## 2026-05-27 — Issue #35: CONTRIBUTING.md cadence-wording propagation
**Duration:** ~3 min · **PR:** #36

- Replaced pre-D-008 `~60-minute session cap` line with D-008 (180/360 min, multi-issue loop) and D-004 (Phase A PR auto-merge) wording, matching the bootstrap template post-portfolio-ops#3.

**Why this work, this session:** Iteration in the autonomous NIGHT session propagation arc for portfolio-ops#3.

**Open questions / blockers:** none.

**Next session:** continue portfolio propagation.

## 2026-06-01 — Issue #37: Queries-JSONL collecting-mode validator + cross-file corpus check
**Duration:** ~17 min · **Branch:** `session/2026-06-01-2320-issue-37`

- Shipped `chunking_lab.validate.validate_queries(path, corpus_dir=None)` — frozen `ValidationReport` with `n_rows`, `n_valid`, findings tuple. Sixteen finding codes covering `json.loads` failures (`malformed_json`, `not_an_object`), per-field schema gaps for all four required fields (`missing_<f>` / `non_string_<f>` / `empty_<f>`), uniqueness (`duplicate_id`), empty-file, and — when `corpus_dir` is provided — the cross-file `expected_doc_not_found` invariant.
- The cross-file check is the highest-leverage finding in this validator: a typo'd `expected_doc` silently invalidates recall (the run completes, the number becomes meaningless). Letting validate walk the corpus alongside the queries catches it.
- Wired `python -m chunking_lab.validate <path> [--corpus-dir DIR] [--json]` (no top-level console_script — D-004 minimalism kept; `scripts/` invocation style is the established shape). Exit codes 0 / 1 / 2 uniform with the sister validators.
- `tests/test_validate.py` is 33 cases — happy path against the shipped substrate (both standalone and with `corpus_dir`), accumulating-errors (no fail-fast), one parametrized positive per finding code (14 row-level codes), duplicate-`id`, blank-line skip, empty-file, FileNotFoundError on both missing path and missing corpus_dir, JSON-stable `to_dict`, frozen-dataclass lock, `REQUIRED_FIELDS` tuple lock, three `--corpus-dir` variants, and six CLI end-to-end cases.
- README "Architecture" tree gains the new module line; `docs/architecture.md` gains a `validate` pre-flight paragraph under §1 Pinned substrate and a `validate.py` reference in the "Where to look next" Substrate bullet. Architecture-doc lock passes structurally.
- Live-tested against the real 12-row `data/queries.jsonl`: exit 0 in one pass, both standalone and with `--corpus-dir data/corpus`. Confirms no false positives on the shipped substrate and that the cross-file invariant holds. Full suite 230 / 230 pass, ruff clean.

**Why this work, this session:** Second iteration of the day-session loop. Iteration 1 shipped the validate pattern in `embedding-model-shootout` (#45/#46). The chunking-lab `load_queries` fail-fast read is the same shape, and the 4-field schema with an `expected_doc` cross-file constraint made this the natural next target — and the first repo in the validate propagation arc where the cross-file invariant adds substantive value beyond JSONL shape lint.

**Open questions / blockers:** None — ready for review.

**Next session:** Continue the day-session loop. Candidate repos with fail-fast JSONL or data loaders: `vector-search-at-scale`, `python-async-llm-pipelines`, `agent-orchestration-platform`. Pick the next one not touched since 2026-05-27 and check for the same pattern.

## 2026-06-17 — Issue #39: Workflow YAML-parseability lock
**Duration:** ~10 min · **Branch:** `session/2026-06-17-1914-issue-39`

Added `tests/test_workflows_yaml_parseable.py` and pulled `pyyaml>=6.0`
into `[project.optional-dependencies].dev`. 3 tests today (1 smoke + 1
parse + 1 jobs for `ci.yml`).

**Why this work, this session:** Same rationale as the prior two hops
(`llm-eval-harness#60`, `rag-production-kit#52`) — propagating the
`portfolio-ops#30` lock that closed the 21-day silent CI outage in
`portfolio-ops#27`.

**Open questions / blockers:** none — full `pytest` (233 → 236) +
`ruff` clean locally; PR #40 open and waiting for CI.

**Next session:** continue propagation to the remaining 9 repos.

## 2026-06-18 — Issue #41: timeout-minutes guard + lock test
**Duration:** ~20 min · **Branch:** `session/2026-06-18-0318-issue-41`

- Added `timeout-minutes: 15` to every job in `ci.yml` (`lint`, `test`,
  `memory-check`). Uniform 15-min ceiling — no workload here approaches
  it and there's no current justification for tighter or looser bounds.
- Added `tests/test_workflows_timeout_minutes.py` — 10 new tests: 1
  smoke + 3 jobs × 3 parametrized invariants (`timeout-minutes` is
  present, is an int (not bool/str), is in policy band `[1, 30]`).

**Why this work, this session:** GitHub Actions defaults to 360 min/job
when `timeout-minutes` is unset, so a hung job (network stall during
`pip install`, infinite loop in a chunker test, stuck embedding-API
call) burns the full 6-hour ceiling. `llm-eval-harness` PR #63 shipped
the canonical first hop and the portfolio-ops audit (#36) added a
`--check missing-timeout` fingerprint that surfaces every unprotected
repo weekly. This PR is the third propagation hop after
`rag-production-kit` PR #55.

**Open questions / blockers:** none. Full pytest clean (+10 new tests),
ruff check + format both clean.

**Next session:** continue propagation across the remaining 8
unprotected repos. Next per D-009 + build-sequence:
nextjs-streaming-ai-patterns.

## 2026-06-18 — Issue #43: concurrency guard + lock test
**Duration:** ~10 min · **Branch:** `session/2026-06-18-1524-issue-43`

- Added top-level `concurrency: { group: ci-${{ github.ref }},
  cancel-in-progress: true }` to `ci.yml`.
- Copied `tests/test_workflows_concurrency.py` from llm-eval-harness;
  docstring origin updated to this repo's #43 via sed.

**Why this work, this session:** fourth per-repo hop in the
concurrency-lock propagation arc. Audit fingerprint shipped in
portfolio-ops #41 surfaces every workflow missing the lock.

**Open questions / blockers:** none. Test count 240 → 247.

**Next session:** continue propagation to remaining 8 repos.

## 2026-06-19 — Issue #45: validate CLI --out for sink-parity
**Duration:** ~22 min · **Branch:** `session/2026-06-19-0322-issue-45`

- Added `--out PATH` to `python -m chunking_lab.validate` so its
  output (human summary or `--json` payload) atomic-writes to disk
  instead of stdout. Sibling propagation of llm-eval-harness#66.
- `main()` routes through `chunking_lab/io_utils.atomic_write_text`
  when `--out` is set, else `sys.stdout.write(rendered)`. Findings
  print to stderr in human-readable mode regardless of `--out`.
- Exit-2 (file-not-found) raises before any rendering, so `--out`
  leaves no zero-byte sentinel.
- 6 new tests; README unchanged (no dedicated validate-CLI section
  exists in this repo's README).

**Why this work, this session:** propagation of the same recipe
landing in llm-eval-harness#66 minutes earlier. Canonical sink-parity
shape now propagated across two of the four repos that ship a
`validate` CLI.

**Open questions / blockers:** none. 247 → 253 pytest passes. PR #46
open and ready.

**Next session:** propagate to the remaining validate-CLI repos
(prompt-regression-suite, embedding-model-shootout) when scope allows.

## 2026-06-19 — Issue #47: RetrievalRun.from_json / QueryResult.from_json round-trip parity
**Duration:** ~30 min · **Branch:** `session/2026-06-19-issue-47`

- Filed issue #47 during this session's Phase A loop after spotting the asymmetry: `RetrievalRun` had `to_json()` but no symmetric reader; `tests/test_summary_snapshot.py` carried a 19-line hand-written reader helper that the snapshot tests used. Worked immediately.
- Added `QueryResult.from_json(payload)` and `RetrievalRun.from_json(payload)` classmethods. Both restore the frozen-tuple invariant on the read path; the `RetrievalRun` reader coerces `recall_at_k` / `snippet_hit_at_k` keys back from `str` to `int` (the write side stringifies them), defaults `wall_clock_ms` / `notes` to dataclass defaults for pre-D-009 JSON compatibility, and raises `KeyError` naming the missing field rather than silently filling defaults.
- Collapsed `tests/test_summary_snapshot.py:_load_run_from_json` from 19 lines onto `RetrievalRun.from_json`. The snapshot renderer doesn't read `per_query` so the previously hand-passed empty tuple was fine for the snapshot, but the rebuilt instance now carries the full `per_query` round-trip identity for the cross-check tests.
- 7 new round-trip tests in `tests/test_metrics.py`: identity for both classmethods, populated/empty `per_query`, str→int key coercion, missing-default-field acceptance, missing-required-field raise, and a cross-check against the committed `results/canonical__*.json` files (5 strategies, byte-for-byte round-trip through `to_json`).
- Ruff caught a UP037 quoted-self type annotation on the classmethod return types (incompatible with the file's `from __future__ import annotations`); dropped quotes, suite clean. Format check clean.

**Why this work, this session:** the portfolio is heavily saturated — most repos have zero open issues. The session's Phase A multi-issue loop demands substantive engineering, and this was a real API-completeness gap surfaced by reading the snapshot test helper. Pattern continues the dogfood→issue→PR loop from earlier sessions: spot the asymmetry in the test helper, file an issue, close it in the same session with the lock-test inverse-safety-net (round-trip + missing-required-key tests).

**Open questions / blockers:** none. 253 → 260 pytest passes. PR #48 merged.

**Next session:** the metrics module's serialization contract is now complete (`to_json` ↔ `from_json` round-trip, locked by 7 new tests). `QueryResult` and `RetrievalRun` remain `chunking_lab.metrics`-internal (not re-exported in `chunking_lab/__init__.py`); promoting them to the top-level API is a separate decision and not blocked. If similar asymmetries exist in sibling repos' \`*_run.to_json()\` surfaces (rag-production-kit `PhaseTimings`, llm-eval-harness `RunRecord`, etc.), the from_json propagation arc is a natural sibling pattern for future sessions.

## 2026-06-22 — Issue #50: semantic chunker preserves inter-sentence whitespace (offset↔text contract)
**Duration:** ~30 min · **Branch:** `session/2026-06-22-0430-issue-50`

- Found by reading `semantic.py`: `_split_sentences_with_offsets` drops the whitespace between sentences, and `_emit_block` built multi-sentence chunks by joining those stripped sentences while computing `end_offset` from the shortened length. So for any chunk spanning 2+ sentences, `source[start_offset:end_offset] != chunk.text` and `end_offset` undercounted the true span — unlike fixed/recursive, which uphold that contract. Demonstrated empirically (a 4-sentence single block reported `end_offset=92` for a 95-char span, truncating the last word).
- The existing `test_chunks_have_valid_offsets` *already* asserts the contract for semantic (its comment even claims semantic preserves source text), but its data + `threshold=0.4` only ever yields single-sentence semantic chunks under `HashEmbedder`, so the concatenation path was never exercised — false confidence.
- Fix: thread the source `text` into `_emit_block` / `_merge_too_small` and slice each chunk's text from the original (`text[start:end]`), preserving whitespace, across the small-block / max-split / min-merge paths. This also repairs the snippet-hit faithfulness metric, which couldn't match a snippet straddling a sentence boundary against whitespace-stripped text. 4 new tests; suite 260 → 264, no snapshot ripple. PR #51 ready.

**Why this work, this session:** the portfolio is saturated; this was a real correctness bug that the test suite *claimed* was covered, plus a faithfulness-metric undercount in exactly the multi-sentence-passage case the lab exists to measure — high value, not a synthetic fill.

**Open questions / blockers:** none.

**Next session:** a trailing semantic chunk shorter than `min_chunk_chars` with no successor to merge into still gets emitted (forward-only merge). Debatable-by-design; low-pri if a future session wants filler here.

## 2026-06-22 — Issue #52: semantic chunker — min-merge must respect the max_chunk_chars hard ceiling
**Duration:** ~25 min · **Branch:** `session/2026-06-22-1116-issue-52`

- Found during Phase A (an Explore subagent flagged it after I'd cleared metrics/fixed/recursive/structure/late myself): `SemanticBoundaryStrategy` documents `max_chunk_chars` as a hard ceiling and `_emit_block` honors it, but the `min_chunk_chars` merge pass ran afterward and could merge a too-small chunk with its successor past the ceiling — and because merges chain greedily, the result grew well over it. Reproduced 127- and 106-char chunks against a 100-char cap.
- Fix: on conflict the hard ceiling wins (`min` is a soft floor). `_merge_too_small` now merges only when the combined source span stays within `max_chunk_chars`, else leaves the small chunk as-is. Span measured source-side, so the offset↔text contract (#50) still holds.
- 3 new tests (ceiling-never-breached + offset contract, merge-still-happens-when-it-fits via merged-vs-unmerged count). Suite 263 → 266, ruff clean. PR #53 ready.

**Why this work, this session:** the portfolio is saturated (only binary demo-capture tasks open). This was a real contract violation in the core chunker — the merge step silently undid the ceiling guarantee `_emit_block` works to maintain — and was untested. Higher value than a synthetic fill.

**Open questions / blockers:** none.

**Next session:** no specific lead — the semantic/structure/fixed/recursive/late strategies are all well-hardened. `metrics.evaluate_strategy`'s recall@k is effectively hit-rate@k (one relevant doc per query, which is fine); score-tie ordering in the chunk ranking is deterministic insertion order. If a future session needs work here, the from_json symmetry across `metrics.RetrievalRun` (already has from_json) vs. consumers is the remaining surface.
