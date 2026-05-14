# Reciprocal Rank Fusion

## What it is

Reciprocal Rank Fusion (RRF) is a parameter-light strategy for combining several ranked retrieval results into a single ranked list. It was introduced by Cormack, Clarke, and Buettcher in a 2009 SIGIR paper showing that it outperformed Condorcet-style voting and a number of supervised rank-learning approaches on TREC-style information-retrieval benchmarks.

The formula is:

```
score(doc) = sum over rankers r where doc appears: 1 / (k + rank_r(doc))
```

`k` is a small constant — the original paper uses 60 — that smooths the contribution of each ranker. Documents that the top of one ranker but never see in another still get a contribution; documents that all rankers agree on get the highest scores.

## Why it works

RRF doesn't require the participating rankers to produce comparable score scales. That's the property that makes it useful for hybrid retrieval pipelines that mix BM25-style lexical scores (which are positive log-odds-like quantities) with dense cosine similarities (which are bounded in `[-1, 1]`). The two scalars aren't directly comparable; their ranks are.

The smoothing constant `k` prevents the top of any single ranker from dominating the fused result. Without `k`, a doc ranked #1 by ranker A and absent from rankers B and C would beat a doc ranked #2 by all three rankers — which is the opposite of what consensus should mean.

## When to use it

RRF is the right default fusion strategy for retrieval pipelines that combine 2–4 rankers (e.g., BM25 + dense, BM25 + dense + reranker, BM25 + dense + sparse-lexical). It handles correlated rankers gracefully and adds no per-corpus tuning step.

It is the wrong tool when:

- One ranker is consistently and substantially better than the others, in which case weighted score fusion (with normalization) extracts more signal.
- The rankers produce calibrated probabilities that can be sensibly multiplied, in which case product-of-experts is theoretically better grounded.

## Operating notes

Surface the per-ranker rank in the fused result. Without it, the only way to answer "why did this doc appear in the top-5?" is by re-running the underlying queries — which is fine to do but tedious. The cost of carrying the per-ranker ranks through is trivial (a dict per doc) and pays back the first time you have to debug a surprising fusion.
