# Chunking strategies — retrieval metrics matrix

_embedder_: `HashEmbedder` · _n_queries_: 12

> **Note.** HashEmbedder is the dep-free CI embedder; its vectors are effectively random per text. Absolute recall numbers below reflect the runner working, **not** the strategies' real retrieval quality. Run with `--embedder minilm` (after `pip install -e '.[sbert]'`) for honest numbers.

| strategy | n_chunks | recall@1 | recall@3 | recall@5 | snippet-hit@1 | snippet-hit@3 | snippet-hit@5 | wall-clock (ms) |
| -------- | -------: | -------: | -------: | -------: | ------------: | ------------: | ------------: | --------------: |
| fixed-size | 29 | 0.333 | 0.500 | 0.917 | 0.083 | 0.083 | 0.333 | 20 |
| recursive | 30 | 0.583 | 0.750 | 0.750 | 0.000 | 0.083 | 0.250 | 22 |
| semantic | 84 | 0.167 | 0.750 | 0.833 | 0.083 | 0.167 | 0.167 | 82 |
| late-chunking | 29 | 0.083 | 0.250 | 0.500 | 0.083 | 0.083 | 0.167 | 24 |
| structure-aware | 28 | 0.000 | 0.250 | 0.583 | 0.000 | 0.000 | 0.000 | 21 |
