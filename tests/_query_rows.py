"""Per-query evidence for test fixtures, sized to the run's `n_queries` (#204).

`RetrievalRun` requires `n_queries == len(per_query)` since D-020. Before that,
most fixtures in this suite built a run as `n_queries=N, per_query=[]` -- a
record whose published denominator had no evidence behind it, which is the
shape D-020 refuses. These build the smallest rows that are valid `QueryResult`s:
the rows carry no retrievals, because no fixture that uses them asserts on
per-query content.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from chunking_lab.metrics import QueryResult


def query_rows(n: int) -> list[dict[str, Any]]:
    """`n` per-query rows in the JSON shape `RetrievalRun.from_json` reads."""
    return [
        {
            "query_id": f"q{i}",
            "expected_doc": "doc",
            "expected_snippet": "snippet",
            "retrieved_doc_ids_in_rank_order": [],
            "snippet_hits_in_rank_order": [],
        }
        for i in range(n)
    ]


def query_results(n: int) -> tuple[QueryResult, ...]:
    """`n` `QueryResult`s, for fixtures that call the constructor directly."""
    return tuple(QueryResult.from_json(row) for row in query_rows(n))


def _hits_for(name: str, rates: Mapping[int, float], n: int) -> list[tuple[int, int]]:
    """`(k, hits)` ascending in k, each `hits / n` equal to its rate exactly."""
    out: list[tuple[int, int]] = []
    for k in sorted(rates):
        rate = rates[k]
        if not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValueError(f"{name}[{k}]={rate!r} is not a finite number")
        hits = round(rate * n) if n else 0
        expected = hits / n if n else 0.0
        if expected != rate:
            raise ValueError(f"{name}[{k}]={rate!r} is not hits/{n} for any whole number of hits")
        if out and hits < out[-1][1]:
            raise ValueError(
                f"{name} falls from {out[-1][1]}/{n} at k={out[-1][0]} to {hits}/{n} at "
                f"k={k}; the [:k] slices nest, so no rows can produce that"
            )
        out.append((k, hits))
    return out


def smallest_n(*maps: Mapping[int, float]) -> int:
    """The smallest query count at which every rate in `maps` is `hits / n` exactly."""
    n = 1
    for rates in maps:
        for rate in rates.values():
            n = math.lcm(n, Fraction(rate).limit_denominator(10**6).denominator)
    return n


def evidence_rows(
    recall_at_k: Mapping[int, float],
    snippet_hit_at_k: Mapping[int, float],
    n: int | None = None,
) -> list[dict[str, Any]]:
    """`n` JSON rows whose own retrievals produce exactly these rates (#218, D-021).

    `RetrievalRun` refuses a rate its `per_query` rows contradict, so a fixture
    that wants `recall_at_k={1: 0.5, 3: 1.0}` needs rows that give it. Row `i` is
    a recall hit for every `k` at or above the rank its expected doc sits at, and
    the same for its snippet flag, so the `[:k]` slices nest the way
    `evaluate_strategy`'s do. `n` defaults to `smallest_n(...)`. A rate no rows
    can produce -- not `hits / n`, or falling as `k` grows -- raises here, at the
    fixture, rather than inside the constructor under test.
    """
    if n is None:
        n = smallest_n(recall_at_k, snippet_hit_at_k)
    ks = sorted(set(recall_at_k) | set(snippet_hit_at_k))
    depth = ks[-1] if ks else 0

    def rank_for(steps: list[tuple[int, int]], i: int) -> int | None:
        # The first k whose hit count covers row i; the row's hit sits AT that
        # rank, so it is inside every slice from that k up and outside the rest.
        for k, hits in steps:
            if i < hits:
                return k
        return None

    recall_steps = _hits_for("recall_at_k", recall_at_k, n)
    snippet_steps = _hits_for("snippet_hit_at_k", snippet_hit_at_k, n)
    rows = []
    for i in range(n):
        doc_rank = rank_for(recall_steps, i)
        snippet_rank = rank_for(snippet_steps, i)
        rows.append(
            {
                "query_id": f"q{i}",
                "expected_doc": "doc",
                "expected_snippet": "snippet",
                "retrieved_doc_ids_in_rank_order": [
                    "doc" if r == doc_rank else f"other{r}" for r in range(1, depth + 1)
                ],
                "snippet_hits_in_rank_order": [r == snippet_rank for r in range(1, depth + 1)],
            }
        )
    return rows


def evidence(
    recall_at_k: Mapping[int, float],
    snippet_hit_at_k: Mapping[int, float],
    n: int | None = None,
) -> tuple[QueryResult, ...]:
    """`evidence_rows`, as `QueryResult`s for fixtures that call the constructor."""
    return tuple(
        QueryResult.from_json(row) for row in evidence_rows(recall_at_k, snippet_hit_at_k, n)
    )
