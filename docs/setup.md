# Experimental setup (pinned by D-002)

This document pins the three things every chunking strategy in this repo
is compared on. Changes to any of these invalidate downstream benchmark
numbers (#3) and require a deliberate revisit per the portfolio handoff.

## Corpus

**Location:** `data/corpus/*.md`
**License:** MIT (authored for this repo).
**Documents:** 5 multi-paragraph technical articles, 600–1200 words each.

| File                       | Topic                                       |
| -------------------------- | ------------------------------------------- |
| `01_hnsw.md`               | HNSW indexes: parameters and trade-offs     |
| `02_rrf.md`                | Reciprocal Rank Fusion                      |
| `03_prompt_caching.md`     | Anthropic prompt caching                    |
| `04_eval_harness.md`       | LLM evaluation harness design               |
| `05_async_pipelines.md`    | Async LLM pipeline patterns                 |

Each document has Markdown headings, multi-paragraph prose, and code
blocks — the structural diversity that makes chunking strategies
differentiate. The corpus is small on purpose: it has to be tractable
to chunk under every strategy variation we'll run in #3 in one CI
minute, so we can run the matrix on every PR without skipping.

## Downstream task

**Location:** `data/queries.jsonl`
**Format:** one JSON object per line:

```json
{
  "id": "q01",
  "question": "What parameter controls the candidate list size during HNSW build?",
  "expected_doc": "01_hnsw.md",
  "expected_snippet": "ef_construction"
}
```

`expected_snippet` is **verbatim text** from `expected_doc`. The
retrieval metrics in #3 score a strategy by whether the chunk(s) it
retrieves contain the snippet. The test
`test_each_expected_snippet_is_verbatim_in_its_document` is the gate
that prevents the snippets and the corpus from drifting apart.

The current set is 12 queries spanning every document. The queries are
worded as a developer would phrase them — they don't paste keywords
from the answer; they ask the question in natural language.

## Embedding model

**Pinned:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, Apache 2.0,
CPU-friendly).

The model id is exposed as `chunking_lab.CANONICAL_EMBEDDING_MODEL`.
The test `test_canonical_model_is_pinned` is the gate that prevents
silent drift.

The package provides three embedder classes:

- `HashEmbedder` (dep-free) — deterministic 384-d hash-based vectors.
  Tests use this; not for production-grade retrieval quality.
- `MiniLMEmbedder` — wraps the canonical sentence-transformers model.
  Ships behind the `[sbert]` optional extra (D-003) so CI doesn't
  download model weights on every run.
- `Embedder` (Protocol) — the type all of the above satisfy. Strategy
  code in #2 should be parameterized over this protocol, not over a
  concrete class.

## Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Verify the substrate loads:
python -c "
from chunking_lab import load_corpus, load_queries, CANONICAL_EMBEDDING_MODEL
docs = load_corpus()
qs = load_queries()
print(f'docs: {len(docs)}, queries: {len(qs)}, model: {CANONICAL_EMBEDDING_MODEL}')
"

# Real-embedding numbers (issue #3) will require the sbert extra:
pip install -e '.[sbert]'
```

## Change protocol

Any change to:

- a corpus document's text,
- a query's expected_snippet,
- the `CANONICAL_EMBEDDING_MODEL` constant,

invalidates the strategy-comparison numbers in `results/` (when #3
lands). A change like that ships as its own PR with a corresponding
revisit of D-002 (`superseded_by` a new D-NNN) and a rerun of the
matrix.
