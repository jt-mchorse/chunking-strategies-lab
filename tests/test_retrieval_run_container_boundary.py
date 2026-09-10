"""`RetrievalRun`'s container fields are guarded on the write path too (#186).

`from_json` has guarded **four** containers since #114/#118 — the top-level
payload, both metric maps, `per_query`, and `notes` — each with a comment
saying why: a raw `TypeError`/`AttributeError` "escaping the documented
`KeyError`/`ValueError` loud contract". That phrase appears four times in one
method.

`__post_init__` guarded **zero** of them. It validated every numeric field —
`n_queries`/`n_chunks_total` (#180), `wall_clock_ms` (#181), the two metric maps
(#182) — and no container. The numeric axis was completed one field at a time by
four issues whose own words name this shape: "a rule stated in one place and
applied in another", and "guarding two of the three numeric fields would be the
half-fix this issue is about."

Measured on the unguarded class::

    per_query = 'not a list'              constructed, to_json RAW AttributeError
    per_query = 5                         constructed, to_json RAW TypeError
    per_query = None                      constructed, to_json RAW TypeError
    per_query = ('not a QueryResult',)    constructed, to_json RAW AttributeError
    notes = 5                             constructed, to_json RAW TypeError
    notes = None                          constructed, to_json RAW TypeError
    notes = 'abc'                         constructed, ROUND-TRIPPED
    notes = [1, 2]                        constructed, ROUND-TRIPPED

Six rows raise exactly the exception types `from_json`'s comments exist to
convert, one seam later, out of a method that had no such guard at all.

**The two silent rows are the reason this is an issue rather than six raw
tracebacks.** `to_json` writes `"notes": list(self.notes)`, so::

    RetrievalRun(..., notes="chunk overlap looks high").to_json()["notes"]
      -> ['c', 'h', 'u', 'n', 'k', ' ', 'o', 'v', ...]   24 entries
    from_json(that).notes
      -> the same 24 single-character notes

`from_json`'s guard for that very field names the harm — "a JSON string silently
char-splats into a per-character list" — and calls itself "the last list
container built via `list(...)` on the **read** path". The identical `list(...)`
on the write path was never asked about.
"""

from __future__ import annotations

from typing import Any

import pytest

from chunking_lab.metrics import QueryResult, RetrievalRun


def _query_result(query_id: str = "q1") -> QueryResult:
    return QueryResult(
        query_id=query_id,
        expected_doc="d1",
        expected_snippet="snippet",
        retrieved_doc_ids_in_rank_order=("d1", "d2"),
        snippet_hits_in_rank_order=(True, False),
    )


def _run(**overrides: Any) -> RetrievalRun:
    base: dict[str, Any] = {
        "strategy_name": "fixed",
        "embedder_model": "hash-64",
        "dataset_version": "v1",
        "n_queries": 1,
        "n_chunks_total": 2,
        "recall_at_k": {1: 1.0},
        "snippet_hit_at_k": {1: 1.0},
        "per_query": (_query_result(),),
        "wall_clock_ms": 1.5,
        "notes": ["ok"],
    }
    base.update(overrides)
    return RetrievalRun(**base)


# (label, field, value, substring the refusal must contain). Every row was
# verified to CONSTRUCT on the unguarded class; the last column is what it did
# one seam later.
REJECT_ROWS: tuple[tuple[str, str, Any, str], ...] = (
    ("per_query is a str", "per_query", "abc", "per_query must be a sequence"),
    ("per_query is an int", "per_query", 5, "per_query must be a sequence"),
    ("per_query is None", "per_query", None, "per_query must be a sequence"),
    ("per_query is a dict", "per_query", {"a": 1}, "per_query must be a sequence"),
    ("per_query holds a str", "per_query", ("x",), r"per_query\[0\] must be a QueryResult"),
    ("per_query holds a dict", "per_query", ({"query_id": "q"},), r"per_query\[0\] must be"),
    (
        "per_query holds a QueryResult then a str",
        "per_query",
        (_query_result(), "x"),
        r"per_query\[1\] must be a QueryResult",
    ),
    ("notes is a str", "notes", "abc", "notes must be a list"),
    ("notes is an int", "notes", 5, "notes must be a list"),
    ("notes is None", "notes", None, "notes must be a list"),
    ("notes is a tuple", "notes", ("a",), "notes must be a list"),
    ("notes holds an int", "notes", [1, 2], r"notes\[0\] must be a str"),
    ("notes holds a str then an int", "notes", ["ok", 2], r"notes\[1\] must be a str"),
    ("notes holds None", "notes", [None], r"notes\[0\] must be a str"),
)

ACCEPT_ROWS: tuple[tuple[str, str, Any], ...] = (
    ("per_query empty tuple", "per_query", ()),
    ("per_query empty list", "per_query", []),
    ("per_query as a list", "per_query", [_query_result()]),
    ("per_query with several", "per_query", (_query_result("a"), _query_result("b"))),
    ("notes empty", "notes", []),
    ("notes one string", "notes", ["chunk overlap looks high"]),
    ("notes several strings", "notes", ["a", "b", "c"]),
    ("notes with an empty string", "notes", [""]),
)


@pytest.mark.parametrize(
    ("label", "field", "value", "expected"), REJECT_ROWS, ids=[r[0] for r in REJECT_ROWS]
)
def test_a_bad_container_is_refused_at_construction(
    label: str, field: str, value: Any, expected: str
) -> None:
    """`ValueError`, this module's documented type — not the raw
    `TypeError`/`AttributeError` these shapes raised out of `to_json`.

    `match` on the field name, not a bare `ValueError`: without it a rejection
    raised by some *other* field of the baseline would pass the row for the
    wrong reason.
    """
    with pytest.raises(ValueError, match=expected):
        _run(**{field: value})


@pytest.mark.parametrize(("label", "field", "value"), ACCEPT_ROWS, ids=[r[0] for r in ACCEPT_ROWS])
def test_a_valid_container_still_round_trips(label: str, field: str, value: Any) -> None:
    """The control. Without these the reject table would be satisfied by a
    `__post_init__` that refused everything — and `per_query` legitimately
    arrives as a `list` from `from_json` and a `tuple` from the annotation, so
    both have to keep working.
    """
    run = _run(**{field: value})
    reloaded = RetrievalRun.from_json(run.to_json())
    assert list(reloaded.notes) == list(run.notes)
    assert len(reloaded.per_query) == len(run.per_query)
    assert [q.query_id for q in reloaded.per_query] == [q.query_id for q in run.per_query]


def test_the_tables_have_teeth_on_both_sides() -> None:
    assert len(REJECT_ROWS) >= 14
    assert len(ACCEPT_ROWS) >= 8
    assert sum(1 for r in REJECT_ROWS if r[1] == "per_query") >= 6
    assert sum(1 for r in REJECT_ROWS if r[1] == "notes") >= 6


# --- the two rows that were SILENT ----------------------------------------


def test_a_string_notes_is_not_one_note_per_character() -> None:
    """The sharpest row, and the reason the check reads the field rather than
    the payload.

    `to_json` does `list(self.notes)`, so a string became a valid `list[str]`
    with nothing left for a rule stated over the payload to object to. It
    round-tripped cleanly as 24 notes.
    """
    assert list("chunk overlap") == [
        "c",
        "h",
        "u",
        "n",
        "k",
        " ",
        "o",
        "v",
        "e",
        "r",
        "l",
        "a",
        "p",
    ]
    with pytest.raises(ValueError, match="notes must be a list"):
        _run(notes="chunk overlap looks high")


def test_a_non_str_note_is_refused() -> None:
    """`notes` is annotated `list[str]`, and `[1, 2]` round-tripped through
    `to_json` -> `from_json` unchanged. JSON has no problem carrying it; this
    class's own annotation does.
    """
    with pytest.raises(ValueError, match=r"notes\[0\] must be a str"):
        _run(notes=[1, 2])


# --- the writer cannot emit what its own reader refuses --------------------


def test_the_writer_cannot_produce_a_payload_its_own_reader_refuses() -> None:
    """The six loud rows, stated as the round trip they broke.

    Before #186 each of these constructed and then raised a raw
    `TypeError`/`AttributeError` out of `to_json` — not even reaching a payload
    for `from_json` to refuse. The assertion is that construction now stops
    them, and that the honest path still round-trips, because a guard that also
    broke the honest path would satisfy the first half for the wrong reason.
    """
    bad_per_query: tuple[Any, ...] = ("abc", 5, None, ({"query_id": "q"},))
    for value in bad_per_query:
        with pytest.raises(ValueError, match="per_query"):
            _run(per_query=value)
    bad_notes: tuple[Any, ...] = ("abc", 5, None, [1])
    for value in bad_notes:
        with pytest.raises(ValueError, match="notes"):
            _run(notes=value)

    run = _run()
    reloaded = RetrievalRun.from_json(run.to_json())
    assert reloaded.to_json() == run.to_json()


# --- the read path's own guards are NOT redundant --------------------------


@pytest.mark.parametrize(
    ("field", "bad", "expected"),
    [
        ("per_query", "abc", "per_query must be a JSON array"),
        ("notes", "abc", "notes must be a JSON array"),
    ],
    ids=["per_query", "notes"],
)
def test_from_json_still_guards_its_own_container(field: str, bad: Any, expected: str) -> None:
    """A coercion sits between `from_json`'s guard and `__post_init__`'s, and it
    launders the error.

    `list("abc")` is a perfectly good `list[str]`, and
    `tuple(QueryResult.from_json(q) for q in "abc")` iterates a string into
    characters — so by the time the constructor's rule runs there is nothing
    left for it to object to. Two guards, one rule, two different moments; the
    read-side one is load-bearing and its message is pinned by #114/#118's own
    tests.
    """
    payload = _run().to_json()
    payload[field] = bad
    with pytest.raises(ValueError, match=expected):
        RetrievalRun.from_json(payload)


def test_the_laundering_is_real_and_not_a_story() -> None:
    """Run the claim in the docstring above rather than asserting it in prose.

    If `from_json`'s container guards were removed, these are the values that
    would reach `__post_init__` — and they are valid, which is exactly why the
    read-side guard cannot be deleted as redundant.
    """
    assert list("abc") == ["a", "b", "c"]
    from chunking_lab.metrics import _validate_notes

    _validate_notes(list("abc"))  # no raise: the splat produced a valid list[str]
    with pytest.raises(ValueError, match="notes must be a list"):
        _validate_notes("abc")


# --- one definition, not a copy -------------------------------------------
#
# The copy-instead-of-share neighbour — inline the two rules in `__post_init__`
# rather than calling the shared validators — passes ALL 1555 tests. Built and
# run, not assumed. That is the fourth time in this module's history that shape
# has been the separating case (#180, #181, #182 each shipped a shared
# definition for the same reason), and a behavioural suite cannot distinguish
# one definition from two identical ones by construction.
#
# Same idiom as `test_the_shared_validator_is_the_only_place_the_parity_rule_lives`
# and `test_each_rule_is_written_once`, with one change: counted over AST string
# literals excluding docstrings, not over raw source text. A raw `.count` would
# false-hit the day someone quotes a message in a comment explaining the fix —
# which is exactly what the prose above this block does with "must be a str".


def _module_string_literals() -> list[str]:
    """Every string literal in `metrics.py` that is not a docstring."""
    import ast
    from pathlib import Path

    import chunking_lab.metrics as metrics_module

    tree = ast.parse(Path(metrics_module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def test_post_init_calls_the_shared_validators() -> None:
    import inspect

    source = inspect.getsource(RetrievalRun.__post_init__)
    assert "_validate_per_query" in source, "__post_init__ carries its own per_query rule"
    assert "_validate_notes" in source, "__post_init__ carries its own notes rule"


@pytest.mark.parametrize(
    "marker",
    [
        "per_query must be a sequence of QueryResult",
        "must be a QueryResult, got ",
        "notes must be a list of strings",
        "must be a str, got ",
    ],
    ids=repr,
)
def test_each_container_rule_is_written_once(marker: str) -> None:
    literals = _module_string_literals()
    occurrences = sum(1 for lit in literals if marker in lit)
    assert occurrences == 1, (
        f"{marker!r} appears in {occurrences} string literals; each rule must "
        "live once, in its shared validator"
    )


def test_the_literal_scan_is_not_vacuous() -> None:
    """A scan that found no literals would make every count above 0 != 1 — but a
    scan that found only *some* would silently make a genuine duplicate
    invisible. Anchor it on a message that certainly exists.
    """
    literals = _module_string_literals()
    assert len(literals) >= 30
    assert any("must be a JSON array" in lit for lit in literals)
