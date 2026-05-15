# Hierarchical Navigable Small World (HNSW) Indexes

## Overview

HNSW is a graph-based approximate nearest-neighbor index that gives sub-millisecond query latency on tens of millions of high-dimensional vectors, in exchange for some memory overhead and a build-time cost. It's the index pgvector reaches for by default in modern Postgres deployments, and it's the index Qdrant and Weaviate use as their primary structure. Understanding what its parameters actually do is most of what you need to operate it well.

The graph is built in layers. Each inserted vector lands in some top layer chosen by an exponential distribution, then connects to its nearest neighbors in every layer down to layer zero. Queries enter at the top, greedily walk toward the query vector at each layer, and descend until they hit the bottom — at which point they switch from a single best-neighbor walk to a wider beam search controlled by `ef_search`.

## Build-time parameters

The build is controlled by two parameters:

- **M**: the number of bidirectional connections each node gets in the upper layers. Typical values are 12–48. Higher M gives better recall and faster queries but uses more memory and slows build.
- **ef_construction**: the size of the dynamic candidate list during build. Typical values are 64–512. Higher values give a better-quality graph but slow the build linearly. Past about 200 the returns diminish on most workloads.

A common pgvector configuration is `M=16, ef_construction=64`, which is a sensible starting point for embedding sizes in the 384–1024 range.

## Query-time parameters

At query time the knob is **ef_search**, which controls the size of the candidate list maintained during the layer-0 beam search. Higher values give better recall but proportionally slower queries. For most production stacks `ef_search` between 40 and 200 is the right zone — go higher if you're seeing recall holes, lower if latency is dominated by ANN.

`ef_search` can be tuned per-query without rebuilding the index, which is the property that makes HNSW operationally pleasant compared to IVF-flat variants.

## Trade-offs

HNSW shines when:

- The corpus fits in RAM (or you have a fast SSD and `mmap` works for you).
- You need sub-10ms p95 query latency.
- The vectors are 256–1024 dimensions.

HNSW is the wrong tool when:

- The corpus is so large that the graph itself doesn't fit in memory, in which case IVF-PQ variants are usually better.
- You need exact recall — there is no `ef_search` value that gives you 100% recall on every query, only convergence toward it.

## Operating notes

Re-index when you change the embedding model — the geometry changes and the graph from the old embeddings is meaningless under the new ones. Re-index on a copy and atomically swap; HNSW build is expensive enough that you don't want to hold the write path during it.
