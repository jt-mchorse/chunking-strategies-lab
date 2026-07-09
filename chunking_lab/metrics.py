"""Retrieval metrics matrix (#3).

Pure-function evaluator: takes a `Strategy`, a corpus, a query set, and
an embedder, returns a `RetrievalRun` with per-query results + recall@k
and snippet-hit@k aggregates.

The runner is intentionally embedder-agnostic. CI uses `HashEmbedder`
to verify *the matrix produces structurally-correct output* (every
strategy runs, JSON schema is stable, no exceptions on the edge cases).
Real numbers — which strategy wins on a given corpus — require running
the same `evaluate_strategy(...)` against a real embedder
(`MiniLMEmbedder` or any conforming backend). The dep-free CI path is
plumbing-only, per the portfolio's no-fabricated-benchmarks rule.

Two metrics:

- **recall@k** — the expected-doc filename appears in the top-k
  retrieved chunks' `source_doc_id` list. Standard IR metric over the
  document granularity the queries file gives us.
- **snippet-hit@k** — the expected-snippet *substring* is present in
  *some single* top-k retrieved chunk's text (per-chunk, not the
  concatenation of the top-k). This is the answer-faithfulness proxy
  for this layer (D-008): structural rather than semantic, but cheap,
  hermetic, and it gates strategies that fragment the relevant passage
  across chunk boundaries — which is the whole concern the lab exists
  to measure. The per-chunk check is load-bearing: a snippet split
  across two adjacent chunks is absent from every single chunk and so
  scores a **miss**, even though concatenating the chunks would re-join
  it. Matching against the concatenated top-k text instead would *heal*
  fragmentation and invert the metric, so it is deliberately not done.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .corpus import Document
from .embedder import Embedder
from .queries import Query
from .strategies import Chunk, LateChunkingStrategy, Strategy


@dataclass(frozen=True)
class QueryResult:
    """One query's evaluation against one strategy."""

    query_id: str
    expected_doc: str
    expected_snippet: str
    retrieved_doc_ids_in_rank_order: tuple[str, ...]
    # Per-rank flags: True if this rank-position's chunk text contained
    # `expected_snippet`. Length matches `retrieved_doc_ids_in_rank_order`.
    snippet_hits_in_rank_order: tuple[bool, ...]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> QueryResult:
        """Inverse of the dict shape emitted by ``RetrievalRun.to_json``."""
        # A non-object per_query row makes the `payload[...]` lookups below raise
        # `TypeError` ('int'/'str' object is not subscriptable), escaping the
        # loud KeyError/ValueError contract — parity with the RetrievalRun
        # metric-map guard and the load_queries non-object guard (#110/#111).
        if not isinstance(payload, dict):
            raise ValueError(f"per_query row must be a JSON object, got {type(payload).__name__}")
        # A present-but-non-array rank-order field (a JSON scalar/null) makes
        # `tuple(...)` raise a raw `TypeError` ('int' object is not iterable),
        # escaping the loud KeyError/ValueError contract — the same list-container
        # sibling as `per_query` in `RetrievalRun.from_json`. (A JSON string would
        # silently splat into a char-tuple; requiring a list rejects that too.)
        for _field in ("retrieved_doc_ids_in_rank_order", "snippet_hits_in_rank_order"):
            if not isinstance(payload[_field], list):
                raise ValueError(
                    f"{_field} must be a JSON array, got {type(payload[_field]).__name__}"
                )
        return cls(
            query_id=payload["query_id"],
            expected_doc=payload["expected_doc"],
            expected_snippet=payload["expected_snippet"],
            retrieved_doc_ids_in_rank_order=tuple(payload["retrieved_doc_ids_in_rank_order"]),
            snippet_hits_in_rank_order=tuple(payload["snippet_hits_in_rank_order"]),
        )


def _validate_metric_map(name: str, mapping: dict[int, float]) -> None:
    """Reject corrupt metric values on the read path.

    ``recall_at_k`` / ``snippet_hit_at_k`` are proportions in ``[0, 1]``
    (see :class:`RetrievalRun`). A hand-edited or externally-generated
    result file can carry a non-finite (``NaN``/``±Inf``) or out-of-range
    value; loaded silently, a ``NaN`` sorts unpredictably and an
    out-of-range ``recall@k`` can crown the wrong strategy in the
    comparison. Fail loud here, matching the loud-key contract of
    ``from_json`` and the loader-finiteness guards in sibling repos.
    """
    for k, v in mapping.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{k}] must be a number; got {v!r}")
        if not math.isfinite(v):
            raise ValueError(f"{name}[{k}] must be finite; got {v!r}")
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}[{k}] must be in [0, 1]; got {v!r}")


@dataclass(frozen=True)
class RetrievalRun:
    """Aggregate output of `evaluate_strategy`.

    `recall_at_k` and `snippet_hit_at_k` are keyed by integer k. Both
    are floats in [0, 1] — the proportion of queries whose expected
    doc / snippet appeared in the top-k.
    """

    strategy_name: str
    embedder_model: str
    dataset_version: str
    n_queries: int
    n_chunks_total: int
    recall_at_k: dict[int, float]
    snippet_hit_at_k: dict[int, float]
    per_query: tuple[QueryResult, ...]
    # Wall-clock for the full chunk + embed + retrieve pipeline. The
    # latency a downstream consumer sees if they swap in this strategy.
    # Defaults to 0.0 so older JSON files (pre-D-009) still load cleanly
    # via consumers that pass `wall_clock_ms=0.0` when reading.
    wall_clock_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "embedder_model": self.embedder_model,
            "dataset_version": self.dataset_version,
            "n_queries": self.n_queries,
            "n_chunks_total": self.n_chunks_total,
            "wall_clock_ms": self.wall_clock_ms,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "snippet_hit_at_k": {str(k): v for k, v in self.snippet_hit_at_k.items()},
            "per_query": [
                {
                    "query_id": q.query_id,
                    "expected_doc": q.expected_doc,
                    "expected_snippet": q.expected_snippet,
                    "retrieved_doc_ids_in_rank_order": list(q.retrieved_doc_ids_in_rank_order),
                    "snippet_hits_in_rank_order": list(q.snippet_hits_in_rank_order),
                }
                for q in self.per_query
            ],
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RetrievalRun:
        """Inverse of :meth:`to_json`.

        Restores the frozen-tuple invariants on the read path
        (``per_query`` becomes a tuple, each ``QueryResult`` is
        rebuilt with tuple-typed rank-order fields) and coerces
        ``recall_at_k`` / ``snippet_hit_at_k`` keys back from ``str``
        to ``int``. Defaults for ``wall_clock_ms`` and ``notes``
        match the dataclass defaults so older committed JSONs
        (pre-D-009) continue to load cleanly.

        Raises ``KeyError`` with the missing field name when a
        required key is absent, and ``ValueError`` when a metric value
        is non-numeric, non-finite, or outside ``[0, 1]`` — the failure
        mode is loud, not silent, on both the key and the value axis.
        """
        # A non-object top-level payload (a bare JSON array/scalar/null from a
        # hand-edit or a truncated write) makes the `payload[...]` lookups below
        # raise `TypeError`/`AttributeError`, escaping the documented loud
        # `KeyError`/`ValueError` contract. Guard the container axis first, same
        # as the sibling `load_queries` non-object guard (#110/#111).
        if not isinstance(payload, dict):
            raise ValueError(
                f"run JSON top-level value must be a JSON object, got {type(payload).__name__}"
            )
        # A present-but-wrong-shape metric map (a JSON array/scalar written by a
        # hand-edit or an external generator) makes the very next `.items()`
        # raise `AttributeError`, escaping the documented `KeyError`/`ValueError`
        # loud contract on the container axis — the same class of bug the sibling
        # `load_queries` non-object guard fixed (#110/#111). Convert it to the
        # documented `ValueError` before iterating.
        for _name in ("recall_at_k", "snippet_hit_at_k"):
            if not isinstance(payload[_name], dict):
                raise ValueError(
                    f"{_name} must be a JSON object, got {type(payload[_name]).__name__}"
                )
        recall = {int(k): v for k, v in payload["recall_at_k"].items()}
        snippet = {int(k): v for k, v in payload["snippet_hit_at_k"].items()}
        _validate_metric_map("recall_at_k", recall)
        _validate_metric_map("snippet_hit_at_k", snippet)
        # A present-but-non-array `per_query` (a JSON scalar/null from a hand-edit
        # or truncated write) makes the `for q in ...` below raise a raw
        # `TypeError`, escaping the documented `KeyError`/`ValueError` loud
        # contract — the list-container sibling of the metric-map guards above
        # (#114 guarded the top-level and both metric maps, not this container).
        per_query_raw = payload.get("per_query", ())
        if not isinstance(per_query_raw, (list, tuple)):
            raise ValueError(f"per_query must be a JSON array, got {type(per_query_raw).__name__}")
        # A present-but-non-array `notes` (a JSON scalar/null from a hand-edit or
        # external generator) reaches `list(...)` below and raises a raw
        # `TypeError`, escaping the documented `KeyError`/`ValueError` loud
        # contract; a JSON string silently char-splats into a per-character list.
        # This is the last list container built via `list(...)` on the read path
        # — the same sibling class #114 guarded for the top-level/metric maps and
        # #118 guarded for `per_query`/rank-order, neither of which touched `notes`.
        notes_raw = payload.get("notes", [])
        if not isinstance(notes_raw, list):
            raise ValueError(f"notes must be a JSON array, got {type(notes_raw).__name__}")
        return cls(
            strategy_name=payload["strategy_name"],
            embedder_model=payload["embedder_model"],
            dataset_version=payload["dataset_version"],
            n_queries=payload["n_queries"],
            n_chunks_total=payload["n_chunks_total"],
            recall_at_k=recall,
            snippet_hit_at_k=snippet,
            per_query=tuple(QueryResult.from_json(q) for q in per_query_raw),
            wall_clock_ms=payload.get("wall_clock_ms", 0.0),
            notes=list(notes_raw),
        )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    sim = dot / (na * nb)
    # Zero-norm is handled above, but a non-finite (NaN/±Inf) component in
    # `a`/`b` slips that guard and makes `sim` non-finite (#66). In
    # `evaluate_strategy` the score feeds `scored.sort(key=...)`, where a NaN
    # sorts into an implementation-defined slot (all NaN comparisons are
    # False) and silently corrupts the recall@k / snippet-hit@k ranking — the
    # worst outcome for a measurement lab. `Embedder` is a BYO Protocol, so a
    # poisoned vector is reachable; this helper is the only line of defense,
    # so fail loud rather than return a fallback that would hide it. Same
    # result-finiteness check as llm-cost-optimizer's cosine() (#88), but
    # raising. Sibling of rag-production-kit #82.
    if not math.isfinite(sim):
        raise ValueError(
            "non-finite cosine similarity from a NaN/Inf embedding component; "
            "the embedder returned a non-finite vector"
        )
    return sim


def evaluate_strategy(
    strategy: Strategy,
    corpus: list[Document],
    queries: list[Query],
    embedder: Embedder,
    *,
    ks: Sequence[int] = (1, 3, 5),
    dataset_version: str = "v0",
) -> RetrievalRun:
    """Run `strategy` over `corpus`, rank chunks per query, return metrics.

    Late-chunking is handled specially: its `chunk()` returns regular
    `Chunk`s but its `chunk_with_vectors()` returns `(Chunk, vector)`
    pairs whose vectors were derived from document-level context
    (D-006). When the strategy is `LateChunkingStrategy` and supports
    that, we use the pre-baked vectors; otherwise we fall back to
    embedding each chunk's text directly. The result shape is the
    same — only the vector source differs.

    For `LateChunkingStrategy`, the strategy's embedder and the runner's
    `embedder` must report the same model name. If they disagree, the
    chunk vectors and the query vectors live in different embedding
    spaces and the resulting cosine scores are meaningless. The runner
    enforces this consistency (D-011); a mismatch raises `ValueError`
    immediately rather than producing a plausible-looking but garbage
    recall@k curve.
    """
    _check_late_chunking_embedder_consistency(strategy, embedder)

    # Non-positive `k` flows through `retrieved_docs[:k]` slicing without
    # raising — k=0 silently produces recall@0=0.0 always; k<0 silently
    # miscounts ("all but the last N"). Empty `ks` silently produces an
    # empty `recall_at_k` dict. Surface every offender in one pass so
    # operators don't chase them one-at-a-time. Mirrors the run_sweep
    # k_values guard in embedding-model-shootout (#28).
    if not ks:
        raise ValueError("ks must be non-empty")
    bad_k = sorted({k for k in ks if k <= 0})
    if bad_k:
        raise ValueError(f"every k in ks must be positive; got {bad_k}")
    # Deduplicate (order-preserving) so the counter dicts, `max(ks)`, and the
    # per-query counting loop all operate on the same unique-key set. The hit
    # dicts below already collapse duplicate keys, but the raw counting loop
    # iterates `ks`, so a duplicate `k` would be counted once per occurrence and
    # push `recall_hits[k] / n` above 1.0 — yielding a RetrievalRun that its own
    # `from_json` validator rejects (#84). No effect for already-unique `ks`.
    ks = tuple(dict.fromkeys(ks))

    t_start = time.perf_counter()
    chunks_with_vecs = _materialize_vectors(strategy, corpus, embedder)
    n_chunks = len(chunks_with_vecs)

    per_query: list[QueryResult] = []
    max_k = max(ks)
    recall_hits = {k: 0 for k in ks}
    snippet_hits = {k: 0 for k in ks}

    for q in queries:
        query_vec = embedder.embed(q.question)
        scored: list[tuple[float, Chunk]] = []
        for chunk, vec in chunks_with_vecs:
            scored.append((_cosine(query_vec, vec), chunk))
        # Deterministic, order-independent ranking (#68). Sorting on the score
        # alone left score ties broken by insertion order — i.e. the order
        # `_materialize_vectors` happened to emit chunks (corpus iteration
        # order). For a measurement lab, recall@k / snippet-hit@k must be a pure
        # function of the (score, chunk) set, not of how the corpus was read, so
        # equal scores (common with HashEmbedder, near-duplicate chunks, or
        # genuine ties) must resolve the same way every run. Break ties on the
        # chunk's stable identity (source_doc_id, start/end offsets). Negate the
        # score for an ascending composite sort instead of reverse=True (which
        # would also reverse the tiebreak); scores are guaranteed finite by
        # `_cosine`, which raises on NaN/Inf (#66), so negation is safe.
        scored.sort(key=lambda r: (-r[0], r[1].source_doc_id, r[1].start_offset, r[1].end_offset))
        top = scored[:max_k]
        retrieved_docs = tuple(c.source_doc_id for _, c in top)
        snippet_in_chunk = tuple((q.expected_snippet in c.text) for _, c in top)
        per_query.append(
            QueryResult(
                query_id=q.id,
                expected_doc=q.expected_doc,
                expected_snippet=q.expected_snippet,
                retrieved_doc_ids_in_rank_order=retrieved_docs,
                snippet_hits_in_rank_order=snippet_in_chunk,
            )
        )
        for k in ks:
            if q.expected_doc in retrieved_docs[:k]:
                recall_hits[k] += 1
            if any(snippet_in_chunk[:k]):
                snippet_hits[k] += 1

    wall_clock_ms = (time.perf_counter() - t_start) * 1000.0
    n = len(queries)
    return RetrievalRun(
        strategy_name=strategy.name,
        embedder_model=_embedder_model_name(embedder),
        dataset_version=dataset_version,
        n_queries=n,
        n_chunks_total=n_chunks,
        recall_at_k={k: (recall_hits[k] / n if n else 0.0) for k in ks},
        snippet_hit_at_k={k: (snippet_hits[k] / n if n else 0.0) for k in ks},
        per_query=tuple(per_query),
        wall_clock_ms=wall_clock_ms,
    )


def _materialize_vectors(
    strategy: Strategy, corpus: list[Document], embedder: Embedder
) -> list[tuple[Chunk, list[float]]]:
    """Chunk + embed every document; return `(chunk, vector)` pairs.

    Late-chunking strategies provide their own vectors via
    `chunk_with_vectors` (the vectors are blended with the
    document-level embedding for context, per D-006); everything else
    falls through to embedding each chunk's text directly with the
    evaluator's embedder. **For the late-chunking path to produce
    vectors comparable to the query embedding, the operator must
    construct `LateChunkingStrategy(embedder=...)` with the same
    embedder passed to `evaluate_strategy`** — otherwise the chunk
    vectors live in a different space than the query vectors and the
    cosine scores are meaningless. `evaluate_strategy` enforces this by
    model name (D-011) and raises before this function is reached.
    """
    out: list[tuple[Chunk, list[float]]] = []
    use_late_vectors = isinstance(strategy, LateChunkingStrategy) and hasattr(
        strategy, "chunk_with_vectors"
    )
    for doc in corpus:
        if use_late_vectors:
            pairs = strategy.chunk_with_vectors(  # type: ignore[attr-defined]
                doc.text, source_doc_id=doc.filename
            )
            for lc in pairs:
                out.append((lc.chunk, list(lc.vector)))
        else:
            chunks = strategy.chunk(doc.text, source_doc_id=doc.filename)
            for ch in chunks:
                out.append((ch, list(embedder.embed(ch.text))))
    return out


def _embedder_model_name(embedder: Embedder) -> str:
    # Embedders aren't required to expose a model name; fall back to the
    # class name when one isn't available.
    name = getattr(embedder, "model_name", None)
    if isinstance(name, str) and name:
        return name
    return type(embedder).__name__


def _check_late_chunking_embedder_consistency(strategy: Strategy, embedder: Embedder) -> None:
    """Enforce D-011: late-chunking strategy + runner must share an embedding space.

    For non-late strategies this is a no-op; their chunk vectors come
    from `embedder` directly so consistency is automatic. For
    `LateChunkingStrategy`, the strategy's own embedder is what
    produced the document-level vectors, so a model-name disagreement
    means cosine ranking against `embedder`-derived query vectors is
    meaningless. Comparison is by `_embedder_model_name`, not Python
    identity — two `HashEmbedder()` instances both report `HashEmbedder`
    and are correctly accepted.
    """
    if not isinstance(strategy, LateChunkingStrategy):
        return
    strategy_model = _embedder_model_name(strategy.embedder)
    runner_model = _embedder_model_name(embedder)
    if strategy_model != runner_model:
        raise ValueError(
            "LateChunkingStrategy embedder mismatch: "
            f"strategy.embedder reports model_name={strategy_model!r} but "
            f"evaluate_strategy was passed embedder reporting model_name={runner_model!r}. "
            "Construct the strategy with the same embedder you pass to "
            "evaluate_strategy, e.g. "
            "`LateChunkingStrategy(embedder=emb); evaluate_strategy(..., embedder=emb)`. "
            "(D-011 — see MEMORY/core_decisions_human.md.)"
        )
