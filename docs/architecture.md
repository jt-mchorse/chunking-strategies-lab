# Architecture

```mermaid
flowchart LR
    classDef shipped fill:#dcffe4,stroke:#22863a,color:#000
    classDef pending fill:#fff5b4,stroke:#c69400,color:#000

    Docs["data/corpus/*.md<br/>5 hand-authored articles"]:::shipped --> CorpusLoader["load_corpus()"]:::shipped
    Queries["data/queries.jsonl<br/>12 verbatim-snippet queries"]:::shipped --> QueriesLoader["load_queries()"]:::shipped

    CorpusLoader --> Pipeline["Strategy comparison pipeline<br/>(#2 + #3)"]:::pending
    QueriesLoader --> Pipeline
    Embedder["Embedder Protocol<br/>HashEmbedder (dep-free)<br/>MiniLMEmbedder ([sbert])"]:::shipped --> Pipeline

    Pipeline --> S1["fixed-size<br/>(#2)"]:::pending
    Pipeline --> S2["recursive<br/>(#2)"]:::pending
    Pipeline --> S3["semantic<br/>(#2)"]:::pending
    Pipeline --> S4["late-chunking<br/>(#2)"]:::pending
    Pipeline --> S5["structure-aware<br/>(#2)"]:::pending

    S1 & S2 & S3 & S4 & S5 --> Metrics["recall@k, faithfulness,<br/>runtime per strategy (#3)"]:::pending
```

## Shipped (this PR — issue #1)

The pinned substrate:

- **Corpus.** Five Markdown technical articles in `data/corpus/`,
  authored for this repo (MIT). Heterogeneous structure (headings,
  paragraphs, code blocks) so chunking strategies' differences show up.
- **Queries.** Twelve verbatim-snippet queries in `data/queries.jsonl`.
  A query passes when a strategy retrieves a chunk that contains the
  `expected_snippet` from the `expected_doc`.
- **Embedder.** `Embedder` Protocol + `HashEmbedder` (dep-free
  reference) + `MiniLMEmbedder` (behind `[sbert]` optional extra).
  Canonical model: `sentence-transformers/all-MiniLM-L6-v2` (384-d).

## Pending

- **Issue #2:** the five strategies, each as a module exposing
  `chunk(text, **opts) -> list[Chunk]` with unit tests and a
  per-strategy runtime number.
- **Issue #3:** the retrieval metrics matrix. One command runs all
  five strategies across all twelve queries, persists per-strategy
  numbers to `results/`, surfaces them in `docs/benchmarks.md`.
