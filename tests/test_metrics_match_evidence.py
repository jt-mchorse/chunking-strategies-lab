"""Every published rate is what its own `per_query` rows give (#218, D-021).

D-020 tied `n_queries` to `len(per_query)` and named the other half -- "whether
recall_at_k AGREES WITH the per_query evidence is a DERIVED VALUE invariant, A
DIFFERENT CLASS" -- without filing it. Measured on `main` before this rule:

    n_queries=0, recall_at_k={1: 1.0}, per_query=()             constructs, renders 1.000
    n_queries=1, recall_at_k={1: 1.0}, per_query=(<a miss>,)     constructs, renders 1.000
    n_queries=1, recall_at_k={5: 0.37}                           constructs, renders 0.370

`results/canonical__*.json` is the provenance for every published number
(#198, D-017), so a rate there that its rows contradict is a fabricated
measurement with the evidence sitting beside it.

The rule recomputes with `evaluate_strategy`'s own arithmetic and compares with
`==`. `test_a_tolerance_would_admit_a_near_miss` is the arm that rejects the
`math.isclose` neighbour; `test_every_committed_canonical_file_satisfies_it_exactly`
is the arm that shows `==` costs nothing on the files this repo ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import QueryResult, RetrievalRun, evaluate_strategy
from chunking_lab.queries import Query
from chunking_lab.strategies import FixedSizeStrategy, RecursiveStrategy
from tests._query_rows import evidence, evidence_rows

_ROOT = Path(__file__).resolve().parents[1]
_REFUSAL = r"but its per_query rows give \d+/\d+"


def _row(*, doc_rank: int | None, snippet_rank: int | None, depth: int = 5) -> QueryResult:
    """One query whose expected doc / snippet sits at the given rank (or nowhere)."""
    return QueryResult(
        query_id="q",
        expected_doc="doc",
        expected_snippet="snippet",
        retrieved_doc_ids_in_rank_order=tuple(
            "doc" if r == doc_rank else f"other{r}" for r in range(1, depth + 1)
        ),
        snippet_hits_in_rank_order=tuple(r == snippet_rank for r in range(1, depth + 1)),
    )


def _run(**overrides: Any) -> RetrievalRun:
    kwargs: dict[str, Any] = {
        "strategy_name": "fixed-size",
        "embedder_model": "hash",
        "dataset_version": "v0",
        "n_queries": 1,
        "n_chunks_total": 3,
        "recall_at_k": {1: 0.0, 5: 1.0},
        "snippet_hit_at_k": {1: 1.0, 5: 1.0},
        "per_query": (_row(doc_rank=3, snippet_rank=1),),
        "wall_clock_ms": 1.0,
    }
    kwargs.update(overrides)
    return RetrievalRun(**kwargs)


# ---------------------------------------------------------------------------
# The issue's three shapes, on both paths
# ---------------------------------------------------------------------------

_ISSUE_SHAPES = [
    pytest.param(
        {"n_queries": 0, "per_query": (), "recall_at_k": {1: 1.0}, "snippet_hit_at_k": {1: 1.0}},
        id="zero-queries-claiming-1.0",
    ),
    pytest.param(
        {
            "per_query": (_row(doc_rank=None, snippet_rank=1),),
            "recall_at_k": {1: 1.0},
            "snippet_hit_at_k": {1: 1.0},
        },
        id="recall-1.0-beside-a-miss",
    ),
    pytest.param(
        {
            "per_query": (_row(doc_rank=1, snippet_rank=1),),
            "recall_at_k": {5: 0.37},
            "snippet_hit_at_k": {5: 1.0},
        },
        id="0.37-over-one-query",
    ),
]


@pytest.mark.parametrize("overrides", _ISSUE_SHAPES)
def test_the_constructor_refuses_the_issues_shapes(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=_REFUSAL):
        _run(**overrides)


@pytest.mark.parametrize("overrides", _ISSUE_SHAPES)
def test_from_json_refuses_the_issues_shapes(overrides: dict[str, Any]) -> None:
    # A good run's payload with the issue's fields spliced in: the read path
    # builds through `cls(...)`, so it meets the same rule with no second copy.
    payload = _run().to_json()
    bad = {**_run().__dict__, **overrides}
    payload["n_queries"] = bad["n_queries"]
    payload["recall_at_k"] = {str(k): v for k, v in bad["recall_at_k"].items()}
    payload["snippet_hit_at_k"] = {str(k): v for k, v in bad["snippet_hit_at_k"].items()}
    payload["per_query"] = [
        {
            "query_id": q.query_id,
            "expected_doc": q.expected_doc,
            "expected_snippet": q.expected_snippet,
            "retrieved_doc_ids_in_rank_order": list(q.retrieved_doc_ids_in_rank_order),
            "snippet_hits_in_rank_order": list(q.snippet_hits_in_rank_order),
        }
        for q in bad["per_query"]
    ]
    with pytest.raises(ValueError, match=_REFUSAL):
        RetrievalRun.from_json(json.loads(json.dumps(payload)))


def test_the_snippet_map_is_checked_on_its_own() -> None:
    # Recall agrees; only the snippet rate is wrong. A rule that checked one
    # map and assumed the other would pass this.
    with pytest.raises(
        ValueError, match=r"snippet_hit_at_k\[1\] is 0\.0 but its per_query rows give 1/1"
    ):
        _run(snippet_hit_at_k={1: 0.0, 5: 1.0})


def test_the_message_names_the_map_the_k_and_the_fraction() -> None:
    with pytest.raises(ValueError, match="recall_at_k") as exc:
        _run(recall_at_k={1: 1.0, 5: 1.0})
    assert str(exc.value).startswith("recall_at_k[1] is 1.0 but its per_query rows give 0/1 = 0.0")


def test_every_k_is_checked_not_only_the_largest() -> None:
    # k=1 is the wrong one here; the largest k (5) agrees.
    with pytest.raises(ValueError, match=r"recall_at_k\[1\]"):
        _run(recall_at_k={1: 1.0, 5: 1.0})
    # And the largest k on its own.
    with pytest.raises(ValueError, match=r"recall_at_k\[5\]"):
        _run(recall_at_k={1: 0.0, 5: 0.0})


# ---------------------------------------------------------------------------
# `==`, not a tolerance
# ---------------------------------------------------------------------------


def test_a_tolerance_would_admit_a_near_miss() -> None:
    """The `math.isclose` neighbour's accept set, refused.

    One hit in three is `0.3333333333333333`. The next double up differs by
    one ULP -- `math.isclose` (rel_tol=1e-9) calls them equal, and no three rows
    produce it. Both sides of the real check are one IEEE division of the same
    two integers, so the producer's value is reproduced bit for bit and a
    one-ULP difference is a payload that did not come from these rows.
    """
    import math

    third = 1 / 3
    near = math.nextafter(third, 1.0)
    assert near != third
    assert math.isclose(near, third)
    rows = evidence({1: third}, {1: third}, 3)
    _run(n_queries=3, per_query=rows, recall_at_k={1: third}, snippet_hit_at_k={1: third})
    with pytest.raises(ValueError, match=_REFUSAL):
        _run(n_queries=3, per_query=rows, recall_at_k={1: near}, snippet_hit_at_k={1: third})


def test_every_committed_canonical_file_satisfies_it_exactly() -> None:
    """The rule moves no committed number: every canonical file loads as-is."""
    files = sorted((_ROOT / "results").glob("canonical__*.json"))
    assert len(files) == 5, files
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = RetrievalRun.from_json(payload)
        assert run.n_queries > 0, path.name
        assert json.dumps(run.to_json(), sort_keys=True) == json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# What stays legal
# ---------------------------------------------------------------------------


def test_a_zero_query_run_with_zero_rates_stays_legal() -> None:
    """`hits / n if n else 0.0` is the producer's formula, so #192's empty run
    is not reopened."""
    run = _run(n_queries=0, per_query=(), recall_at_k={1: 0.0}, snippet_hit_at_k={1: 0.0})
    assert RetrievalRun.from_json(run.to_json()) == run


def test_an_int_rate_equal_to_the_fraction_stays_legal() -> None:
    # JSON `1` for a proportion is accepted by the value rule (#182); `1 == 1.0`.
    run = _run(recall_at_k={1: 0, 5: 1}, snippet_hit_at_k={1: 1, 5: 1})
    assert RetrievalRun.from_json(run.to_json()).recall_at_k == {1: 0, 5: 1}


@pytest.mark.parametrize("ks", [(1,), (1, 3, 5), (5, 1, 3), (1, 1, 3), (2, 50)])
def test_evaluate_strategys_own_output_always_constructs(ks: tuple[int, ...]) -> None:
    """The rule restates the producer, so the producer can never trip it --
    including duplicate, unordered and beyond-the-chunk-count `ks`."""
    corpus = [
        Document(
            filename="apples.md", text="## Apples\n\nApples grow on apple trees. Gala is crisp.\n"
        ),
        Document(
            filename="pears.md", text="## Pears\n\nPears ripen off the tree. Bosc is brown.\n"
        ),
    ]
    queries = [
        Query(
            id="q1",
            question="Where do apples grow?",
            expected_doc="apples.md",
            expected_snippet="apple trees",
        ),
        Query(
            id="q2",
            question="Which pear is brown?",
            expected_doc="pears.md",
            expected_snippet="Bosc",
        ),
        Query(id="q3", question="unrelated", expected_doc="pears.md", expected_snippet="nowhere"),
    ]
    for strategy in (FixedSizeStrategy(chunk_chars=40, overlap_chars=10), RecursiveStrategy()):
        run = evaluate_strategy(strategy, corpus, queries, HashEmbedder(), ks=ks)
        assert RetrievalRun.from_json(run.to_json()) == run


# ---------------------------------------------------------------------------
# Order, and the test builder itself
# ---------------------------------------------------------------------------


def test_the_count_rule_runs_before_the_rate_rule() -> None:
    """D-020's mismatch is the more fundamental diagnosis and must win the
    message: with the counts disagreeing, a rate rule that ran first would
    divide by a count the rows do not have."""
    with pytest.raises(ValueError, match=r"n_queries is 2 but per_query has 1 rows"):
        _run(n_queries=2, recall_at_k={1: 1.0, 5: 1.0})


def test_the_builder_refuses_rates_no_rows_can_produce() -> None:
    """`tests/_query_rows.evidence` fails at the fixture, not inside the
    constructor under test, for the two shapes no rows give."""
    with pytest.raises(ValueError, match="not hits/3"):
        evidence_rows({1: 0.5}, {1: 0.5}, 3)
    with pytest.raises(ValueError, match="falls from"):
        evidence_rows({1: 1.0, 3: 0.0}, {1: 0.0, 3: 0.0}, 1)
