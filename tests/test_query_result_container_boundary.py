"""`QueryResult.__post_init__` guarded elements and length, not containers (#188).

#187 gave `RetrievalRun.__post_init__` the container guards its own `from_json`
had carried since #114/#118, and stated the finding as a shape: `from_json`
guarded four containers, each with a comment naming the harm, and
`__post_init__` guarded zero.

`QueryResult` is the same table one class up. Its `from_json` guards both
container fields and its comment names both harms verbatim — "a raw `TypeError`
… escaping the loud KeyError/ValueError contract", and "(A JSON string would
silently splat into a char-tuple; requiring a list rejects that too.)". Its
`__post_init__` checked the *elements* (bool flags) and the *length parity*,
and neither container's type. `RetrievalRun.to_json` writes
`list(q.retrieved_doc_ids_in_rank_order)` — the identical `list(...)` #187
chased, on the element class it did not open.

Measured on `ab166ec`, with #187 merged::

    ids='abc',  3 flags   ROUND-TRIPPED -> ('a','b','c')     identical=False
    ids=b'abc', 3 flags   ROUND-TRIPPED -> (97, 98, 99)      identical=False
    ids=5                 raw TypeError: object of type 'int' has no len()
    ids=None              raw TypeError: object of type 'NoneType' has no len()
    flags=5               raw TypeError: 'int' object is not iterable
    flags=None            raw TypeError: 'NoneType' object is not iterable
    control: proper tuple ROUND-TRIPPED, identical
    control: empty        ROUND-TRIPPED, identical

Two things are sharper here than in #187's version of the defect.

**The raw `TypeError`s come out of the guard method itself.** In #187 the value
was constructed and `to_json` blew up a seam later. Here `len()` and
`enumerate()` are called inside `__post_init__`, before either of its
`raise ValueError` lines — so the method that exists to turn bad input into
this module's loud `ValueError` raises the exception type that contract exists
to convert.

**The length-parity invariant hides the string row.** `len("abc") == 3`, so
three real bool flags make the two "parallel per-rank arrays" agree *precisely
because* a string's length is its character count. #184's check, written to
catch divergence, is what made this input look correct — and it then round-trips
as three document ids `'a'`, `'b'`, `'c'`. The `bytes` row is worse: document
ids become the integers `97, 98, 99`.

The ACCEPT rows are the anti-vacuous control, and the two `#184` rows at the
bottom are the other half of it: a container guard written too broadly, or
placed wrongly in the sequence, satisfies every REJECT row here while breaking
the checks this class already had.
"""

from __future__ import annotations

from typing import Any

import pytest

from chunking_lab.metrics import QueryResult, RetrievalRun


def _query_result(**overrides: Any) -> QueryResult:
    base: dict[str, Any] = {
        "query_id": "q1",
        "expected_doc": "d1",
        "expected_snippet": "snippet",
        "retrieved_doc_ids_in_rank_order": ("d1", "d2"),
        "snippet_hits_in_rank_order": (True, False),
    }
    base.update(overrides)
    return QueryResult(**base)


def _run(qr: QueryResult) -> RetrievalRun:
    return RetrievalRun(
        strategy_name="s",
        embedder_model="e",
        dataset_version="v",
        n_queries=1,
        n_chunks_total=1,
        recall_at_k={1: 1.0},
        snippet_hit_at_k={1: 1.0},
        per_query=(qr,),
    )


# (label, overrides, the field name the message must point at)
REJECT: list[tuple[str, dict[str, Any], str]] = [
    # The two silent rows. Both round-trip cleanly on the unfixed class, so an
    # assertion that merely asked "did it raise" would have passed there.
    (
        "str ids char-splat, with matching flag count",
        {
            "retrieved_doc_ids_in_rank_order": "abc",
            "snippet_hits_in_rank_order": (True, False, True),
        },
        "retrieved_doc_ids_in_rank_order",
    ),
    (
        "bytes ids splat to ints, with matching flag count",
        {
            "retrieved_doc_ids_in_rank_order": b"abc",
            "snippet_hits_in_rank_order": (True, False, True),
        },
        "retrieved_doc_ids_in_rank_order",
    ),
    (
        "bytearray ids",
        {
            "retrieved_doc_ids_in_rank_order": bytearray(b"ab"),
            "snippet_hits_in_rank_order": (True, False),
        },
        "retrieved_doc_ids_in_rank_order",
    ),
    # The four that raised a raw TypeError out of __post_init__ itself.
    (
        "int ids",
        {"retrieved_doc_ids_in_rank_order": 5, "snippet_hits_in_rank_order": ()},
        "retrieved_doc_ids_in_rank_order",
    ),
    (
        "None ids",
        {"retrieved_doc_ids_in_rank_order": None, "snippet_hits_in_rank_order": ()},
        "retrieved_doc_ids_in_rank_order",
    ),
    (
        "mapping ids",
        {"retrieved_doc_ids_in_rank_order": {"a": 1}, "snippet_hits_in_rank_order": ()},
        "retrieved_doc_ids_in_rank_order",
    ),
    (
        "int flags",
        {"retrieved_doc_ids_in_rank_order": (), "snippet_hits_in_rank_order": 5},
        "snippet_hits_in_rank_order",
    ),
    (
        "None flags",
        {"retrieved_doc_ids_in_rank_order": (), "snippet_hits_in_rank_order": None},
        "snippet_hits_in_rank_order",
    ),
    (
        "str flags",
        {"retrieved_doc_ids_in_rank_order": ("a", "b"), "snippet_hits_in_rank_order": "ab"},
        "snippet_hits_in_rank_order",
    ),
]

ACCEPT: list[tuple[str, dict[str, Any]]] = [
    ("tuples, the shipped shape", {}),
    (
        "lists — from_json and the runner hold different concrete types",
        {"retrieved_doc_ids_in_rank_order": ["d1"], "snippet_hits_in_rank_order": [True]},
    ),
    ("both empty", {"retrieved_doc_ids_in_rank_order": (), "snippet_hits_in_rank_order": ()}),
    (
        "single rank",
        {"retrieved_doc_ids_in_rank_order": ("d1",), "snippet_hits_in_rank_order": (True,)},
    ),
]


@pytest.mark.parametrize(("label", "overrides", "field"), REJECT, ids=[r[0] for r in REJECT])
def test_a_non_container_is_refused_loudly(
    label: str, overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match="must be a sequence of per-rank values") as exc:
        _query_result(**overrides)
    # The field, not just a blanket sentence: a guard that rejected everything
    # with one message would satisfy `pytest.raises` on all nine rows.
    assert field in str(exc.value), f"{label}: message does not name the field"
    assert type(overrides[field]).__name__ in str(exc.value), (
        f"{label}: message does not name the offending type"
    )


@pytest.mark.parametrize(("label", "overrides"), ACCEPT, ids=[r[0] for r in ACCEPT])
def test_a_real_container_still_round_trips_by_value(label: str, overrides: dict[str, Any]) -> None:
    qr = _query_result(**overrides)
    back = RetrievalRun.from_json(_run(qr).to_json()).per_query[0]
    # By VALUE. `to_json` writes `list(...)` and `from_json` reads `tuple(...)`,
    # so the assertion is against the tuple form of what went in.
    assert back.retrieved_doc_ids_in_rank_order == tuple(qr.retrieved_doc_ids_in_rank_order), label
    assert back.snippet_hits_in_rank_order == tuple(qr.snippet_hits_in_rank_order), label


def test_the_length_invariant_is_what_hid_the_string_row() -> None:
    """The interaction, named, because it is why this survived #184.

    `QueryResult(retrieved_doc_ids_in_rank_order="abc", snippet_hits=(T,F,T))`
    satisfies every check #184 added: the flags are real bools, and the two
    "parallel per-rank arrays" are the same length — because a string's length
    IS its character count. The invariant written to catch divergence is what
    made this input look correct.
    """
    assert len("abc") == len((True, False, True)), (
        "the premise of this test has changed; it exists because a 3-char "
        "string and 3 flags agree on length"
    )
    # So the container check has to fire, and it has to fire on the CONTAINER,
    # not on a length or an element.
    with pytest.raises(ValueError, match="retrieved_doc_ids_in_rank_order") as exc:
        _query_result(
            retrieved_doc_ids_in_rank_order="abc",
            snippet_hits_in_rank_order=(True, False, True),
        )
    message = str(exc.value)
    assert "same length" not in message, (
        "the parity check answered first — it cannot, these two agree on length"
    )
    assert "must be a bool" not in message, "the element check answered first"


def test_the_container_check_runs_before_the_checks_that_would_crash() -> None:
    """Ordering is the fix, not an implementation detail.

    Both existing checks raise a raw `TypeError` on the inputs the container
    check rejects — `enumerate(5)` and `len(5)`. Placing the new guard after
    either of them leaves four of the nine rows raising `TypeError` out of the
    method whose contract is `ValueError`.
    """
    for overrides in (
        {"retrieved_doc_ids_in_rank_order": 5, "snippet_hits_in_rank_order": ()},
        {"retrieved_doc_ids_in_rank_order": (), "snippet_hits_in_rank_order": 5},
        {"retrieved_doc_ids_in_rank_order": None, "snippet_hits_in_rank_order": None},
    ):
        try:
            _query_result(**overrides)
        except ValueError:
            pass
        except TypeError as e:  # pragma: no cover - the regression this pins
            pytest.fail(f"raw TypeError escaped __post_init__ for {overrides}: {e}")
        else:  # pragma: no cover
            pytest.fail(f"no error at all for {overrides}")


def test_the_checks_184_added_still_fire() -> None:
    """A container guard written too broadly satisfies every REJECT row above
    and breaks these. Both keep their own messages."""
    with pytest.raises(ValueError, match="must be a bool"):
        _query_result(retrieved_doc_ids_in_rank_order=("d1",), snippet_hits_in_rank_order=(1,))
    with pytest.raises(ValueError, match="must be the same length"):
        _query_result(
            retrieved_doc_ids_in_rank_order=("d1", "d2"), snippet_hits_in_rank_order=(True,)
        )


def test_from_json_container_guard_is_not_redundant() -> None:
    """RUN the claim rather than restate #187's argument for it.

    `from_json` builds through `tuple(payload[field])`. If its own guard were
    removed on the reasoning that the constructor now covers it, a JSON string
    would be coerced to a perfectly good tuple of characters BEFORE the
    constructor ever sees it — and the constructor's rule would find nothing to
    object to. The coercion between the two launders the error, which is why
    both guards exist.
    """
    # The laundering, demonstrated on the raw values.
    assert tuple("abc") == ("a", "b", "c")
    from chunking_lab.metrics import _is_sequence_container

    assert not _is_sequence_container("abc"), "the constructor rejects the string"
    assert _is_sequence_container(tuple("abc")), (
        "and accepts what from_json's tuple() would have handed it — so "
        "removing from_json's guard would make this reachable again"
    )

    # And from_json still refuses it at its own seam, with its own message.
    payload = {
        "query_id": "q",
        "expected_doc": "d",
        "expected_snippet": "s",
        "retrieved_doc_ids_in_rank_order": "abc",
        "snippet_hits_in_rank_order": [True, False, True],
    }
    with pytest.raises(ValueError, match="must be a JSON array"):
        QueryResult.from_json(payload)


def test_the_container_rule_is_not_respelled_in_query_result() -> None:
    """Structural, because the copy passes every behavioural row.

    #187 measured exactly this: inlining the rule instead of sharing it passed
    all 1555 tests. `_is_sequence_container` is the one definition, and
    `QueryResult` is now its third caller.
    """
    import ast
    import inspect

    import chunking_lab.metrics as metrics_module

    tree = ast.parse(inspect.getsource(metrics_module))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "QueryResult")
    post_init = next(
        n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )

    calls = [n for n in ast.walk(post_init) if isinstance(n, ast.Call)]
    shared = [
        c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "_is_sequence_container"
    ]
    assert len(shared) == 1, (
        f"expected exactly one call to the shared predicate, found {len(shared)} — "
        "two calls is a loop unrolled, zero is the rule respelled inline"
    )

    # And no local isinstance-against-Sequence/str/bytes, which is what the
    # copy would look like.
    for call in calls:
        if isinstance(call.func, ast.Name) and call.func.id == "isinstance" and len(call.args) == 2:
            target = call.args[1]
            elts = target.elts if isinstance(target, ast.Tuple) else [target]
            names = {e.id for e in elts if isinstance(e, ast.Name)}
            assert not (names & {"Sequence", "str", "bytes", "bytearray"}), (
                f"QueryResult.__post_init__ respelled the container rule inline: {names}"
            )
    # Anti-vacuous: the walk really did find the method's body.
    assert len(calls) >= 3, "the AST walk found almost no calls — it is not reading the method"
