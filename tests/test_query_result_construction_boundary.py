"""`QueryResult` enforces the invariants its own field comment states (#184).

`metrics.py` has four construction boundaries and, until this, three rules.
#180 gave `RetrievalRun` the write-side half of its count rules, #181 extended
that to `wall_clock_ms`, #182 to the two metric maps — three instances of one
defect, a rule stated in one place and applied in another. `QueryResult` was the
last boundary with **no** rule at all, and its invariant was stated in a
comment:

    # Per-rank flags: True if this rank-position's chunk text contained
    # `expected_snippet`. Length matches `retrieved_doc_ids_in_rank_order`.
    snippet_hits_in_rank_order: tuple[bool, ...]

"Length matches" was enforced on neither path, and neither were the element
types. Measured on the unguarded class — every one of these constructed *and*
survived `to_json` → `from_json` unchanged:

    shape                  constructs   round trip
    (True,) for 2 ids      ok           ok, flags=(True,)
    (True, False, True)    ok           ok, flags=(True, False, True)
    () for 2 ids           ok           ok, flags=()
    (1, 0)                 ok           ok, flags=(1, 0)
    ("yes", "no")          ok           ok, flags=('yes', 'no')
    (None, None)           ok           ok, flags=(None, None)
    (1.0, 0.0)             ok           ok, flags=(1.0, 0.0)

The read path needs no separate call here, and that is the interesting part:
`from_json` builds through `cls(...)`, so the constructor **is** the shared
definition — the thing #180/#181/#182 each had to arrange by hand. Asserted
below rather than assumed.

**Scope.** The plain `str` fields and the `str` elements of
`retrieved_doc_ids_in_rank_order` stay unchecked, because
`RetrievalRun.__post_init__` does not type-check `strategy_name` either. That
boundary has its own test, so a later reader does not conclude the class is
fully guarded.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any

import pytest

from chunking_lab import metrics as metrics_module
from chunking_lab.metrics import QueryResult


def _query_result(**overrides: Any) -> QueryResult:
    base: dict[str, Any] = {
        "query_id": "q-001",
        "expected_doc": "doc-a",
        "expected_snippet": "the snippet",
        "retrieved_doc_ids_in_rank_order": ("doc-a", "doc-b"),
        "snippet_hits_in_rank_order": (True, False),
    }
    base.update(overrides)
    return QueryResult(**base)


def _payload(**overrides: Any) -> dict[str, Any]:
    """The dict shape `RetrievalRun.to_json` emits for one `per_query` row."""
    base: dict[str, Any] = {
        "query_id": "q-001",
        "expected_doc": "doc-a",
        "expected_snippet": "the snippet",
        "retrieved_doc_ids_in_rank_order": ["doc-a", "doc-b"],
        "snippet_hits_in_rank_order": [True, False],
    }
    base.update(overrides)
    # Through a real JSON round trip, so no row can depend on a Python object
    # that `json.loads` could never produce.
    return json.loads(json.dumps(base))


#: (case id, overrides, substring the field-named message must contain)
_REJECTED: list[tuple[str, dict[str, Any], str]] = [
    ("flags-shorter", {"snippet_hits_in_rank_order": [True]}, "must be the same length"),
    (
        "flags-longer",
        {"snippet_hits_in_rank_order": [True, False, True]},
        "must be the same length",
    ),
    ("flags-empty", {"snippet_hits_in_rank_order": []}, "must be the same length"),
    (
        "ids-empty",
        {"retrieved_doc_ids_in_rank_order": []},
        "must be the same length",
    ),
    ("int-flags", {"snippet_hits_in_rank_order": [1, 0]}, "must be a bool"),
    ("str-flags", {"snippet_hits_in_rank_order": ["yes", "no"]}, "must be a bool"),
    ("null-flags", {"snippet_hits_in_rank_order": [None, None]}, "must be a bool"),
    ("float-flags", {"snippet_hits_in_rank_order": [1.0, 0.0]}, "must be a bool"),
    (
        "one-bad-flag-in-second-position",
        {"snippet_hits_in_rank_order": [True, 0]},
        "snippet_hits_in_rank_order[1]",
    ),
]

#: Shapes that must keep constructing. The anti-vacuous control: a guard
#: written as "the flags must be non-empty" or "the ranking must be non-empty"
#: satisfies every row above and breaks a legitimate empty ranking.
_ACCEPTED: list[tuple[str, dict[str, Any]]] = [
    ("plain-row", {}),
    (
        "both-empty",
        {"retrieved_doc_ids_in_rank_order": [], "snippet_hits_in_rank_order": []},
    ),
    (
        "all-false",
        {"snippet_hits_in_rank_order": [False, False]},
    ),
    (
        "single-rank",
        {"retrieved_doc_ids_in_rank_order": ["doc-a"], "snippet_hits_in_rank_order": [True]},
    ),
    (
        "long-ranking",
        {
            "retrieved_doc_ids_in_rank_order": [f"doc-{i}" for i in range(10)],
            "snippet_hits_in_rank_order": [i % 2 == 0 for i in range(10)],
        },
    ),
]


@pytest.mark.parametrize(
    ("case", "overrides", "phrase"), _REJECTED, ids=[c for c, _, _ in _REJECTED]
)
def test_construction_rejects_the_shape(case: str, overrides: dict[str, Any], phrase: str) -> None:
    tupled = {
        key: tuple(value) if isinstance(value, list) else value for key, value in overrides.items()
    }
    with pytest.raises(ValueError, match="snippet_hits_in_rank_order") as excinfo:
        _query_result(**tupled)
    assert phrase in str(excinfo.value), (case, str(excinfo.value))


@pytest.mark.parametrize(
    ("case", "overrides", "phrase"), _REJECTED, ids=[c for c, _, _ in _REJECTED]
)
def test_from_json_rejects_the_same_shape_with_the_same_message(
    case: str, overrides: dict[str, Any], phrase: str
) -> None:
    """The read path answers identically because it goes through the same door.

    `from_json` calls `cls(...)`, so it inherits `__post_init__` rather than
    carrying a copy. That is asserted structurally below; this is the
    behavioural half.
    """
    with pytest.raises(ValueError, match="snippet_hits_in_rank_order") as excinfo:
        QueryResult.from_json(_payload(**overrides))
    assert phrase in str(excinfo.value), (case, str(excinfo.value))


@pytest.mark.parametrize(("case", "overrides"), _ACCEPTED, ids=[c for c, _ in _ACCEPTED])
def test_a_well_formed_row_still_constructs_and_round_trips(
    case: str, overrides: dict[str, Any]
) -> None:
    payload = _payload(**overrides)
    row = QueryResult.from_json(payload)
    assert len(row.snippet_hits_in_rank_order) == len(row.retrieved_doc_ids_in_rank_order)
    assert all(isinstance(f, bool) for f in row.snippet_hits_in_rank_order)
    # And the values survive, so the guard is not quietly normalising anything.
    assert list(row.snippet_hits_in_rank_order) == payload["snippet_hits_in_rank_order"]
    assert list(row.retrieved_doc_ids_in_rank_order) == payload["retrieved_doc_ids_in_rank_order"]


def test_the_populations_are_not_empty() -> None:
    """Anti-vacuous: an empty parametrize list asserts nothing."""
    assert len(_REJECTED) >= 9
    assert len(_ACCEPTED) >= 5
    assert len({c for c, _, _ in _REJECTED}) == len(_REJECTED)


def test_the_real_producer_still_satisfies_the_rule() -> None:
    """`evaluate_strategy` builds both tuples from one `top` list.

    That is why this issue was `priority:low` — the producer cannot diverge. If
    a future edit makes it diverge, this class now says so at the boundary
    rather than one consumer later, and this test is where that shows up.
    """
    source = inspect.getsource(metrics_module.evaluate_strategy)
    assert "retrieved_docs = tuple(" in source
    assert "snippet_in_chunk = tuple(" in source
    # Both derived from the same slice, which is the invariant in source form.
    assert source.count("for _, c in top") == 2


# --- the structural claim: one door, not two rules ------------------------


def test_from_json_states_no_rule_of_its_own() -> None:
    """The neighbour this catches passes every behavioural test above.

    Copying the length and bool checks into `from_json` — instead of letting
    `cls(...)` apply them — satisfies both parametrized suites and re-creates
    the exact drift that produced #180, #181 and #182: two paths describing
    each other rather than sharing a definition.
    """
    tree = ast.parse(inspect.getsource(metrics_module))
    from_json = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "from_json"
        and any(
            isinstance(c, ast.Constant) and "per_query row" in str(c.value) for c in ast.walk(node)
        )
    )
    raised = [
        message
        for node in ast.walk(from_json)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for message in [node.value]
    ]
    assert not any("must be the same length" in m for m in raised), (
        "from_json carries a copy of the length rule"
    )
    assert not any("must be a bool" in m for m in raised), (
        "from_json carries a copy of the flag-type rule"
    )
    # Anti-vacuous: the walk really did find `from_json`'s own messages.
    assert any("must be a JSON array" in m for m in raised)


@pytest.mark.parametrize("marker", ["must be the same length", "must be a bool, got"])
def test_each_rule_is_written_once(marker: str) -> None:
    source = inspect.getsource(metrics_module)
    assert source.count(marker) == 1, f"{marker!r} appears more than once"


# --- the scope boundary, stated and pinned --------------------------------


def test_the_str_fields_are_deliberately_unchecked() -> None:
    """Parity with `RetrievalRun`, which does not type-check `strategy_name`.

    Making this one class stricter than its sibling for no stated reason is how
    a module's bar becomes unknowable. If the module later decides to type-check
    its `str` fields, this test is the thing to update — not delete.
    """
    row = QueryResult(
        query_id=1,  # type: ignore[arg-type]
        expected_doc=None,  # type: ignore[arg-type]
        expected_snippet=2.5,  # type: ignore[arg-type]
        retrieved_doc_ids_in_rank_order=(7, 8),  # type: ignore[arg-type]
        snippet_hits_in_rank_order=(True, False),
    )
    assert row.query_id == 1

    run = metrics_module.RetrievalRun(
        strategy_name=7,  # type: ignore[arg-type]
        n_queries=0,
        n_chunks_total=0,
        recall_at_k={},
        snippet_hit_at_k={},
        wall_clock_ms=0.0,
        per_query=(),
        embedder_model="hash-64",
        dataset_version="v1",
    )
    assert run.strategy_name == 7


# --- the neighbours, built and run ----------------------------------------


def test_the_length_only_neighbour_accepts_every_int_flag() -> None:
    """ "The comment says 'length matches', so check the length" is the smallest
    change that closes the issue's headline row, and it leaves a field annotated
    `tuple[bool, ...]` holding `(1, 0)` — invisible to `any()` and `sum()`,
    which is the whole reason the bool axis matters here."""

    def length_only(ids: tuple[Any, ...], flags: tuple[Any, ...]) -> bool:
        return len(ids) == len(flags)

    assert length_only(("a", "b"), (1, 0)), "the neighbour accepts it"
    with pytest.raises(ValueError, match="must be a bool"):
        _query_result(snippet_hits_in_rank_order=(1, 0))


def test_the_isinstance_int_neighbour_accepts_exactly_what_the_rule_rejects() -> None:
    """`isinstance(flag, int)` reads as a type check and is the identity on the
    failing case: `True` is an `int` and so is `1`."""
    assert isinstance(1, int)
    assert isinstance(True, int)
    with pytest.raises(ValueError, match=r"snippet_hits_in_rank_order\[0\]"):
        _query_result(snippet_hits_in_rank_order=(1, 0))
    # And the correct value is untouched.
    assert _query_result(snippet_hits_in_rank_order=(True, False)) is not None
