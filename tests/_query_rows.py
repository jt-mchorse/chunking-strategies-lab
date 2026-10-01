"""Per-query evidence for test fixtures, sized to the run's `n_queries` (#204).

`RetrievalRun` requires `n_queries == len(per_query)` since D-020. Before that,
most fixtures in this suite built a run as `n_queries=N, per_query=[]` -- a
record whose published denominator had no evidence behind it, which is the
shape D-020 refuses. These build the smallest rows that are valid `QueryResult`s:
the rows carry no retrievals, because no fixture that uses them asserts on
per-query content.
"""

from __future__ import annotations

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
