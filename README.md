# chunking-strategies-lab
> Empirical comparison of chunking strategies (fixed, recursive, semantic, late-chunking, structure-aware) on retrieval and downstream RAG faithfulness.

![CI](https://github.com/jt-mchorse/chunking-strategies-lab/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## What this is

Chunking is the choice in a RAG pipeline that hides in plain sight. The
embedder gets the credit, the reranker gets the blame, and the chunker
quietly throws away half the signal — by splitting answers across
boundaries, by glomming unrelated paragraphs together, or by ignoring
the document's own structure. `chunking-strategies-lab` runs five
common strategies (fixed-size, recursive, semantic, late-chunking,
document-structure-aware) on the same corpus, the same queries, and the
same embedding model — so the strategy choice is the only variable left
to compare.

This PR pins the substrate that the five strategies (issue [#2]) and
the retrieval metrics matrix (issue [#3]) will share. The corpus is
five hand-authored technical articles (`data/corpus/*.md`), the
downstream task is twelve verbatim-snippet retrieval queries
(`data/queries.jsonl`), and the embedding model is pinned at
`sentence-transformers/all-MiniLM-L6-v2`. Everything ships behind a
small Python API: `load_corpus()`, `load_queries()`, and an `Embedder`
Protocol with a dep-free `HashEmbedder` reference plus a
`MiniLMEmbedder` adapter behind an `[sbert]` extra.

See [`docs/setup.md`](docs/setup.md) for the full substrate spec and
the change protocol — any edit to a corpus document, a query snippet,
or the embedding model invalidates downstream benchmark numbers and
requires a deliberate revisit of D-002.

[#2]: https://github.com/jt-mchorse/chunking-strategies-lab/issues/2
[#3]: https://github.com/jt-mchorse/chunking-strategies-lab/issues/3

## Architecture

```
data/
├── corpus/*.md          ← 5 hand-authored technical articles (MIT)
└── queries.jsonl        ← 12 verbatim-snippet queries against the corpus

chunking_lab/
├── corpus.py            ← load_corpus()
├── queries.py           ← load_queries()
├── embedder.py          ← Embedder Protocol + HashEmbedder + MiniLMEmbedder
└── strategies/          ← #2: 5 chunking strategies, common Strategy interface
    ├── __init__.py      ← Chunk + LateChunk + Strategy Protocol
    ├── fixed.py         ← FixedSizeStrategy (sliding window + overlap)
    ├── recursive.py     ← RecursiveStrategy (separator-hierarchy split)
    ├── semantic.py      ← SemanticBoundaryStrategy (cosine-peak split)
    ├── late.py          ← LateChunkingStrategy (chunks + doc-blended vectors)
    └── structure.py     ← StructureAwareStrategy (markdown-heading split)
```

See **[`docs/architecture.md`](docs/architecture.md)** for the integrated comparison flow, per-layer detail across all four shipped layers (substrate #1, strategies #2, metrics matrix #3, comparison notebook #4), and the design decisions behind each one (D-002…D-010).

## Strategies (#2 · this PR)

Five chunking strategies, each in its own module under
`chunking_lab/strategies/`, sharing one Protocol so the metrics matrix
(#3) iterates them uniformly:

| Strategy | Module | What it does | Returns |
|----------|--------|--------------|---------|
| `FixedSizeStrategy` | `fixed.py` | Sliding-window over chars with overlap | `Chunk[]` |
| `RecursiveStrategy` | `recursive.py` | Recursive split on a hierarchy of separators | `Chunk[]` |
| `SemanticBoundaryStrategy` | `semantic.py` | Embed sentences, split at cosine-distance peaks | `Chunk[]` |
| `LateChunkingStrategy` | `late.py` | Chunks + vectors blended with document embedding | `Chunk[]` (or `LateChunk[]` for vectors) |
| `StructureAwareStrategy` | `structure.py` | Markdown-heading-bounded sections | `Chunk[]` (with `title` + `heading_level` metadata) |

```python
from chunking_lab import (
    HashEmbedder, FixedSizeStrategy, SemanticBoundaryStrategy, load_corpus,
)

docs = load_corpus()
embedder = HashEmbedder()
for strategy in (FixedSizeStrategy(chunk_chars=600),
                 SemanticBoundaryStrategy(embedder=embedder)):
    chunks = strategy.chunk(docs[0].text, source_doc_id=docs[0].filename)
    print(f"{strategy.name}: {len(chunks)} chunks")
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Verify the substrate is reachable:
python -c "
from chunking_lab import load_corpus, load_queries, CANONICAL_EMBEDDING_MODEL
docs = load_corpus()
qs = load_queries()
print(f'corpus: {len(docs)} docs ({sum(d.char_count for d in docs)} chars)')
print(f'queries: {len(qs)}')
print(f'embedding model: {CANONICAL_EMBEDDING_MODEL}')
"

# Tests + lint:
pytest && ruff check . && ruff format --check .

# Real-embedding work (issue #3) will additionally need:
pip install -e '.[sbert]'
```

## Retrieval metrics matrix (#3 · this PR)

`scripts/run_matrix.py` runs all 5 strategies over the pinned corpus +
query set, computes **recall@k** and **snippet-hit@k** (the
answer-faithfulness proxy, D-008), and writes one JSON per strategy
plus `results/summary.md`. Single command:

```bash
# Hermetic / CI: dep-free HashEmbedder. Numbers reflect plumbing, not
# real retrieval quality — the disclosure is in results/summary.md.
python scripts/run_matrix.py                    # regen scratch (gitignored)
python scripts/run_matrix.py --canonical-out    # refresh tracked canonical fixtures

# Honest numbers: same script, real embedder.
pip install -e '.[sbert]'
python scripts/run_matrix.py --embedder minilm
```

Default output is timestamp-prefixed `{stamp}__<strategy>.json` and
`{stamp}__summary.md` — those filenames are gitignored regen scratch
and can be deleted freely. `--canonical-out` writes
`canonical__<strategy>.json` and (overwrites) `results/summary.md`,
which are the **tracked** fixtures that
`tests/test_summary_snapshot.py` locks. Refresh those when (and only
when) the matrix needs to move:

```
results/
  canonical__fixed-size.json
  canonical__recursive.json
  canonical__semantic.json
  canonical__late-chunking.json
  canonical__structure-aware.json
  summary.md
```

Each JSON file is the `RetrievalRun.to_json()` shape: per-strategy
metadata, per-query retrieved-doc rankings, per-rank snippet-hit flags,
and the aggregate recall/snippet-hit at each k. `summary.md` is the
markdown table consumers paste into a blog post or PR description.

**Honest disclosure.** The dep-free `HashEmbedder` produces vectors
that are effectively random per text — the runner *works*, but the
absolute recall numbers it reports are mostly base-rate. The whole
matrix is meaningful only against a real embedder; the markdown
summary explicitly warns about this when run in CI mode. Strategies
differentiate cleanly under `--embedder minilm`; that's how the
operator captures the real curve for `docs/benchmarks.md`.

## Benchmarks / Results

The retrieval matrix runner ships here. Real numbers from
`--embedder minilm` against the pinned corpus + queries land in
`docs/benchmarks.md` when the operator runs them — per the
no-fabricated-benchmarks rule, the README doesn't quote a curve until
that file exists.

### Comparison notebook

[`notebooks/comparison.ipynb`](notebooks/comparison.ipynb) renders the
current `results/` set into three charts — **recall@k**,
**snippet-hit@k** (the answer-faithfulness proxy from
[D-008](MEMORY/core_decisions_human.md)), and **wall-clock latency** per
strategy — plus an honest takeaways section that calls out exactly what
the loaded embedder *can* and *cannot* support claims about. The
loader picks the most recent JSON per strategy, so re-running the
matrix and committing the new files automatically refreshes the next
notebook render. The notebook ships with executed outputs so reviewers
see the charts without installing jupyter.

Install + regenerate:

```bash
pip install -e '.[notebook]'
python scripts/run_matrix.py                      # fills results/ (hash embedder)
jupyter nbconvert --to notebook --inplace --execute notebooks/comparison.ipynb
```

(Use `python scripts/run_matrix.py --embedder minilm` after
`pip install -e '.[sbert]'` for honest quality numbers.)

## Demo

*60-second demo pending — depends on issues [#2] and [#3].*

## Why these decisions

See [`MEMORY/core_decisions_human.md`](MEMORY/core_decisions_human.md).

## License

MIT.
