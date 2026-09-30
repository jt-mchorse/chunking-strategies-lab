"""A frozen record keeps what it validated (#200, D-018).

`frozen=True` prevents *rebinding* an attribute and says nothing about the
object the attribute points at. Four container fields on frozen dataclasses in
this package held the caller's object: `RetrievalRun.recall_at_k`,
`.snippet_hit_at_k`, `.notes`, and `Chunk.metadata`.

The consequence is sharper than "a caller can edit a frozen record", because
`RetrievalRun.__post_init__` **already validates every one of those fields** —
and then stored the caller's object, which made the validation a snapshot
rather than an invariant:

    run = RetrievalRun(..., recall_at_k=recall, notes=notes)   # validated, passes
    recall[5] = 999.0
    notes.append(123)

    run.to_json()["recall_at_k"]  ->  {"5": 999.0}
    RetrievalRun.from_json(...)   ->  ValueError: recall_at_k[5] must be in [0, 1]

**The writer emits a payload its own reader refuses, by the very rule
`__post_init__` just applied.** `to_json` is what writes
`results/canonical__*.json`, and #198 established those files as the provenance
for every published number in this repo.

It is also #186 one call upstream: that issue's `_validate_notes` docstring says
`to_json`'s `list(...)` "was never asked about" after `from_json`'s was, and the
constructor was the one nobody asked about after that.

Three shallow, one deep
-----------------------

A shallow copy is complete exactly when the element type is proved immutable,
and `RetrievalRun` *proves* it: `_validate_metric_maps` admits only non-bool
finite numbers, `_validate_notes` only `str`. `Chunk.metadata` is
`dict[str, Any]` with no validator, so nothing proves it and it takes
`copy_json_value`.

That premise lives in a **validator** rather than in an annotation, so
`test_the_shallow_copies_still_have_their_premise` asserts the validators still
reject a nested container. Weakening either one is what would silently turn
three correct shallow copies into the defect, and nothing else would say so.
This is the `embedding-model-shootout#133` triage question ("is the element type
validated?") deciding depth — and it answers the opposite way from
`rag-production-kit#227`, where nothing validated.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from chunking_lab.io_utils import copy_json_value
from chunking_lab.metrics import QueryResult, RetrievalRun, _validate_metric_maps, _validate_notes
from chunking_lab.strategies import Chunk

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "chunking_lab"


def _run(**overrides: Any) -> RetrievalRun:
    kwargs: dict[str, Any] = {
        "strategy_name": "fixed-size",
        "embedder_model": "hash-256",
        "dataset_version": "v1",
        "n_queries": 1,
        "n_chunks_total": 1,
        "recall_at_k": {5: 1.0},
        "snippet_hit_at_k": {5: 1.0},
        "per_query": (),
        "wall_clock_ms": 1.0,
        "notes": ["ok"],
    }
    kwargs.update(overrides)
    return RetrievalRun(**kwargs)


def _chunk(metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        text="t",
        start_offset=0,
        end_offset=1,
        source_doc_id="d",
        strategy_name="fixed-size",
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# The four rows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["recall_at_k", "snippet_hit_at_k"])
def test_a_metric_map_is_not_the_callers_object(field: str) -> None:
    supplied = {5: 1.0}
    run = _run(**{field: supplied})
    supplied[5] = 999.0
    assert getattr(run, field)[5] == 1.0, (
        f"{field} is the caller's dict; `frozen=True` stopped nothing because nothing was rebound"
    )


def test_notes_is_not_the_callers_object() -> None:
    supplied = ["ok"]
    run = _run(notes=supplied)
    supplied.append("added later")
    assert run.notes == ["ok"]


def test_chunk_metadata_is_not_the_callers_object() -> None:
    supplied: dict[str, Any] = {"heading": "Intro"}
    chunk = _chunk(supplied)
    supplied["INJECTED"] = 1
    assert "INJECTED" not in chunk.metadata


def test_chunk_metadata_is_copied_deeply() -> None:
    """The arm that rejects a shallow `dict(...)` on the one row that needs depth.

    `Chunk.metadata` is the only container field on a frozen record here whose
    element type nothing proves — `dict[str, Any]`, no validator — so `Any`
    admits a nested container and a one-level copy leaves it the caller's.
    """
    supplied: dict[str, Any] = {"nested": {"level": 1}, "listed": [{"level": 1}]}
    chunk = _chunk(supplied)
    supplied["nested"]["level"] = 99
    supplied["listed"][0]["level"] = 99
    assert chunk.metadata["nested"]["level"] == 1, "the copy is one level deep"
    assert chunk.metadata["listed"][0]["level"] == 1, "the copy does not recurse into lists"


# --------------------------------------------------------------------------
# The consequence: the writer stops emitting what its own reader refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("recall_at_k", lambda c: c.__setitem__(5, 999.0)),
        ("snippet_hit_at_k", lambda c: c.__setitem__(5, float("nan"))),
        ("notes", lambda c: c.append(123)),
    ],
    ids=["recall out of range", "snippet nan", "notes non-str"],
)
def test_a_post_construction_edit_cannot_make_to_json_unreadable(field: str, mutate: Any) -> None:
    """`__post_init__` validated it; the copy is what makes that durable.

    Each mutation below is one `from_json` already refuses by name. Before the
    copy, every one of them reached `to_json()` — and `to_json()` is what writes
    `results/canonical__*.json`.
    """
    supplied: Any = [] if field == "notes" else {}
    supplied = {5: 1.0} if field != "notes" else ["ok"]
    run = _run(**{field: supplied})
    mutate(supplied)
    payload = json.loads(json.dumps(run.to_json()))
    restored = RetrievalRun.from_json(payload)
    assert restored.to_json() == run.to_json(), (
        f"a post-construction edit to {field} changed what to_json() emits, and "
        f"from_json refuses the result by the same rule __post_init__ applied"
    )


def test_the_unfixed_shape_really_was_refused_by_from_json() -> None:
    """The defect's own consequence, pinned so the arms above are not tautologies.

    Constructs the payload the aliased record *used* to produce, and shows
    `from_json` rejecting it. If this ever stops raising, the read path relaxed
    and the arms above are checking a rule that no longer exists.
    """
    payload = _run().to_json()
    payload["recall_at_k"]["5"] = 999.0
    with pytest.raises(ValueError, match=r"recall_at_k\[5\] must be in \[0, 1\]"):
        RetrievalRun.from_json(payload)


# --------------------------------------------------------------------------
# The premise the three shallow copies rest on
# --------------------------------------------------------------------------


def test_the_shallow_copies_still_have_their_premise() -> None:
    """`dict(...)` / `list(...)` are complete only while the element type is immutable.

    Unlike `rag-production-kit#227`, where the premise is an *annotation* and a
    widening edit would break it, here the premise is a pair of **validators**.
    Weakening either is what silently turns three correct shallow copies into
    the defect, and nothing else in the suite would say so.
    """
    with pytest.raises(ValueError, match="recall_at_k"):
        _validate_metric_maps({5: {"nested": 1}}, {5: 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="snippet_hit_at_k"):
        _validate_metric_maps({5: 1.0}, {5: [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"notes\[0\] must be a str"):
        _validate_notes([{"nested": 1}])
    with pytest.raises(ValueError, match=r"notes\[0\] must be a str"):
        _validate_notes([["nested"]])


def test_the_validators_run_before_the_copy() -> None:
    """Order, and it is load-bearing rather than tidy — measured the hard way.

    This arm first asserted only that an invalid input is still refused, which
    it is either way, so it stayed **green** against an inverted order while six
    arms in `tests/test_retrieval_run_container_boundary.py` went red. The one
    that matters is silent:

        RetrievalRun(..., notes="chunk overlap looks high")

    Copying first runs `list("chunk overlap looks high")`, which does not raise
    — it **char-splats into 24 single-character notes**, every one of them a
    `str`, so `_validate_notes` then inspects the splatted list and *passes*.
    That is #186's harm exactly, reintroduced by the copy it is paired with.

    So the copies go last, and this arm tests the silent case rather than the
    loud one. `_validate_notes`'s own docstring is the repro; the existing
    boundary module owns the six loud rows.
    """
    with pytest.raises(ValueError, match="notes must be a list of strings"):
        _run(notes="chunk overlap looks high")
    with pytest.raises(ValueError, match=r"notes\[1\] must be a str"):
        _run(notes=["ok", 123])
    with pytest.raises(ValueError, match="recall_at_k"):
        _run(recall_at_k={5: 999.0})
    # And the honest path is unchanged, so the guard is not passing for the
    # wrong reason.
    assert _run(notes=["one", "two"]).notes == ["one", "two"]


# --------------------------------------------------------------------------
# The copy itself
# --------------------------------------------------------------------------


def test_a_cycle_is_copied_rather_than_exhausting_the_stack() -> None:
    """A recursive copy never terminates here — and `RecursionError` is not a
    `ValueError`, which `Chunk.__post_init__`'s own comment names as the class a
    caller catches at this boundary."""
    metadata: dict[str, Any] = {"self": None}
    metadata["self"] = metadata
    copied = copy_json_value(metadata)
    assert copied is not metadata
    assert copied["self"] is copied


def test_a_cyclic_metadata_does_not_raise_from_the_constructor() -> None:
    metadata: dict[str, Any] = {"self": None}
    metadata["self"] = metadata
    chunk = _chunk(metadata)
    assert chunk.metadata["self"] is chunk.metadata


def test_deep_nesting_does_not_raise() -> None:
    deep: dict[str, Any] = {}
    node = deep
    for _ in range(3000):
        node["n"] = {}
        node = node["n"]
    chunk = _chunk(deep)
    depth = 0
    node = chunk.metadata
    while "n" in node:
        node = node["n"]
        depth += 1
    assert depth == 3000


def test_the_copy_is_total_under_a_constrained_recursion_limit() -> None:
    """Assert the outcome, not the road: a shallow limit must not change the answer."""
    metadata: dict[str, Any] = {"a": [{"b": [{"c": 1}]}]}
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(60)
    try:
        assert copy_json_value(metadata) == metadata
    finally:
        sys.setrecursionlimit(original)


def test_shared_substructure_stays_shared_in_the_copy() -> None:
    """The memo preserves the input's sharing structure, so a DAG stays linear."""
    inner: dict[str, Any] = {"k": "v"}
    copied = copy_json_value({"left": inner, "right": inner})
    assert copied["left"] is copied["right"]
    assert copied["left"] is not inner


def test_a_mutable_container_inside_a_tuple_stays_shared() -> None:
    """A measured limitation, pinned so it reads as a decision.

    Rebuilding a tuple loses a `namedtuple`'s class, the measured reason
    `llm-eval-harness`' D-027 drew the same line.
    """
    inner: dict[str, Any] = {"k": "original"}
    chunk = _chunk({"pair": ("a", inner)})
    inner["k"] = "MUTATED"
    assert chunk.metadata["pair"][1]["k"] == "MUTATED", (
        "the tuple case is now covered — update this arm and D-018 rather than deleting it"
    )


# --------------------------------------------------------------------------
# The three validated tuple fields (#202, D-019)
# --------------------------------------------------------------------------


def _query_result(**overrides: Any) -> QueryResult:
    kwargs: dict[str, Any] = {
        "query_id": "q1",
        "expected_doc": "d1",
        "expected_snippet": "s",
        "retrieved_doc_ids_in_rank_order": ("d1",),
        "snippet_hits_in_rank_order": (True,),
    }
    kwargs.update(overrides)
    return QueryResult(**kwargs)


def test_per_query_is_not_the_callers_object() -> None:
    """The fourth thing `RetrievalRun.__post_init__` validates, and D-018's miss.

    A caller passing a `list` is not violating a contract — `_validate_per_query`
    documents that it "accepts any sequence, because the two paths hold different
    concrete types". D-018's exclusion was argued from the *element* type, which
    decides how deep a copy must be and not whether to make one.
    """
    supplied = [_query_result()]
    run = _run(n_queries=1, per_query=supplied)
    assert run.per_query is not supplied
    assert isinstance(run.per_query, tuple)
    supplied.append(_query_result(query_id="q2"))
    assert len(run.per_query) == 1


@pytest.mark.parametrize("field", ["retrieved_doc_ids_in_rank_order", "snippet_hits_in_rank_order"])
def test_a_rank_order_field_is_not_the_callers_object(field: str) -> None:
    """The other two fields whose `__post_init__` validates and then stored.

    Both go through the same `_is_sequence_container` check `per_query` does, so
    a `list` is accepted here for the same documented reason.
    """
    supplied: list[Any] = ["d1"] if field == "retrieved_doc_ids_in_rank_order" else [True]
    other = ["d1"] if field != "retrieved_doc_ids_in_rank_order" else [True]
    result = _query_result(
        **{
            field: supplied,
            (
                "snippet_hits_in_rank_order"
                if field == "retrieved_doc_ids_in_rank_order"
                else "retrieved_doc_ids_in_rank_order"
            ): other,
        }
    )
    assert getattr(result, field) is not supplied
    assert isinstance(getattr(result, field), tuple)


def test_the_guards_own_measured_failure_is_no_longer_one_append_away() -> None:
    """`_validate_per_query`'s docstring is the repro, and it still worked.

    That docstring records, of the *unguarded* class::

        per_query = ('not a QueryResult',)  constructed, to_json raw AttributeError

    and says all four such shapes "raise the exact exception types `from_json`'s
    comments exist to convert". The guard closed the construction-time case. The
    field it guards was the one field whose value it did not keep, so at
    `4600346` the same `AttributeError` came back out of `to_json` after one
    `append`.
    """
    supplied: list[Any] = [_query_result()]
    run = _run(n_queries=1, per_query=supplied)
    supplied.append("NOT A QueryResult")
    payload = run.to_json()  # must not raise AttributeError
    assert len(payload["per_query"]) == 1


def test_a_post_construction_edit_of_a_rank_order_field_cannot_make_to_json_unreadable() -> None:
    """D-018's headline harm, on `QueryResult`'s two fields.

    "The writer emits a payload its own reader refuses, by the very rule
    `__post_init__` just applied." Measured at `4600346`: appending `"nope"` to
    the flags list put `[True, "nope"]` into `to_json`, and `from_json` raised
    `ValueError: snippet_hits_in_rank_order[1] must be a bool` — the rule
    enforced twenty lines above the field it did not keep.
    """
    ids: list[Any] = ["d1"]
    hits: list[Any] = [True]
    result = _query_result(retrieved_doc_ids_in_rank_order=ids, snippet_hits_in_rank_order=hits)
    ids.append(42)
    hits.append("nope")
    payload = _run(n_queries=1, per_query=(result,)).to_json()
    assert payload["per_query"][0]["retrieved_doc_ids_in_rank_order"] == ["d1"]
    assert payload["per_query"][0]["snippet_hits_in_rank_order"] == [True]
    # The round trip the harm broke.
    RetrievalRun.from_json(payload)


def test_clearing_the_supplied_list_no_longer_empties_the_record() -> None:
    """The shape that needs no invalid value at all.

    `pq.clear()` after a passing construction published `"per_query": []` beside
    `"n_queries": 1`, into the file `#198`/D-017 established as the provenance
    for every number this repo publishes. Nothing refuses that payload, which is
    what made it worth measuring separately from the type-error shapes.
    """
    supplied = [_query_result()]
    run = _run(n_queries=1, per_query=supplied)
    supplied.clear()
    assert len(run.to_json()["per_query"]) == 1


def test_the_three_new_copies_run_after_their_checks() -> None:
    """Order is load-bearing here for the same reason D-018 measured for `notes`.

    A copy placed *before* its check hands the validator the coerced value
    instead of the caller's, and the coercion is not neutral. `tuple(5)` raises a
    raw `TypeError: 'int' object is not iterable` out of `__post_init__` — the
    exact exception class `_validate_per_query` and the rank-order container
    check exist to convert into this module's loud `ValueError`. And `tuple("ab")`
    splats into `('a', 'b')`, two perfectly valid doc ids, which is #188's harm
    restored silently rather than loudly.

    Both orders are also rejected repo-wide (the copy-before-validate neighbours
    went 2 and 8 red across `tests/`), but a module that ships a copy should own
    the arm for where it sits — D-018's own note is that its arm was green
    against the wrong order until it was rewritten to test the silent case.
    """
    for bad in (5, None):
        with pytest.raises(ValueError, match="per_query must be a sequence"):
            _run(per_query=bad)
        with pytest.raises(ValueError, match="must be a sequence of per-rank values"):
            _query_result(retrieved_doc_ids_in_rank_order=bad, snippet_hits_in_rank_order=())
    # The silent one: a `str` is a sequence, and three characters would agree in
    # length with three real flags.
    with pytest.raises(ValueError, match="must be a sequence of per-rank values"):
        _query_result(
            retrieved_doc_ids_in_rank_order="abc", snippet_hits_in_rank_order=(True, True, True)
        )


def test_the_in_package_producers_already_passed_a_tuple_so_nothing_republishes() -> None:
    """`tuple(...)` is a no-op on a tuple, and that is why the artifacts do not move.

    `metrics.py`'s two `per_query=` sites both wrap in `tuple(...)` already —
    `from_json` at the read path and `evaluate_strategy` at the write path. The
    committed `results/` files are pinned elsewhere; this arm states *why* they
    are unaffected, so the claim is checked rather than asserted in prose.
    """
    source = (_PACKAGE / "metrics.py").read_text(encoding="utf-8")
    call_sites = [line for line in source.splitlines() if "per_query=" in line]
    assert call_sites, "no per_query= call site found in metrics.py"
    assert all("tuple(" in line for line in call_sites), (
        f"an in-package producer no longer passes a tuple: {call_sites}. "
        f"`tuple(...)` on a tuple returns the same object, which is the whole "
        f"reason D-019 changes no published byte."
    )
    original = (_query_result(),)
    assert _run(n_queries=1, per_query=original).per_query is original


# --------------------------------------------------------------------------
# The population
# --------------------------------------------------------------------------

_MUTABLE_CONTAINERS = ("dict", "list", "set", "Mapping", "MutableMapping")


def _dataclasses_in_package() -> list[tuple[str, str, bool, list[tuple[str, str]]]]:
    out = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if not any("dataclass" in d for d in decorators):
                continue
            fields = [
                (stmt.target.id, ast.unparse(stmt.annotation))
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and stmt.annotation is not None
                and isinstance(stmt.target, ast.Name)
            ]
            out.append(
                (
                    str(path.relative_to(_ROOT)),
                    node.name,
                    any("frozen=True" in d for d in decorators),
                    fields,
                )
            )
    return out


def _is_mutable_container(annotation: str) -> bool:
    return annotation.split("[", 1)[0].split(".")[-1] in _MUTABLE_CONTAINERS


def test_every_frozen_record_with_a_mutable_container_field_copies_it() -> None:
    """Discover the population; do not trust the four the sweep listed.

    Keyed on `frozen=True` (the claim) plus a **mutable** container field (what
    the claim does not cover), walked with `rglob` so `strategies/` is in the
    corpus — a `glob("*.py")` would have missed `Chunk` entirely, which is the
    one row here that needs a deep copy.

    `tuple` is deliberately not a mutable container, which is why `per_query`,
    the two rank-order fields, `LateChunk.vector`, `RecursiveStrategy.separators`
    and `ValidationReport.findings` are not offenders.
    """
    offenders = []
    for module, cls, frozen, fields in _dataclasses_in_package():
        if not frozen:
            continue
        container_fields = [n for n, ann in fields if _is_mutable_container(ann)]
        if not container_fields:
            continue
        tree = ast.parse((_ROOT / module).read_text(encoding="utf-8"))
        body = next(n.body for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        post_init = "".join(
            ast.unparse(n)
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
        )
        for name in container_fields:
            if f"object.__setattr__(self, {name!r}" not in post_init:
                offenders.append(f"{module}:{cls}.{name}")
    assert not offenders, (
        f"these frozen records hold a mutable container field they do not copy: "
        f"{offenders}. `frozen=True` stops a rebind and nothing else, so anything "
        f"validated in `__post_init__` is a snapshot rather than an invariant "
        f"(#200)."
    )


def test_the_population_arm_found_the_four_rows() -> None:
    """A pass over an empty set is not a pass."""
    found = {
        f"{cls}.{name}"
        for _, cls, frozen, fields in _dataclasses_in_package()
        if frozen
        for name, ann in fields
        if _is_mutable_container(ann)
    }
    assert found == {
        "RetrievalRun.recall_at_k",
        "RetrievalRun.snippet_hit_at_k",
        "RetrievalRun.notes",
        "Chunk.metadata",
    }, f"the walk found {sorted(found)}; #200 triaged exactly four frozen rows"


_ANY_CONTAINER = _MUTABLE_CONTAINERS + ("tuple", "Sequence", "frozenset")


def _is_container(annotation: str) -> bool:
    return annotation.split("[", 1)[0].split(".")[-1] in _ANY_CONTAINER


def _post_init_parts(module: str, cls: str) -> tuple[str, str]:
    """`(checking_source, setattr_source)` for one class's `__post_init__`.

    Split by **scope**, not by text: the copy statements are identified as
    `object.__setattr__` calls and the docstring as the leading string
    expression, then removed. A text-keyed split would find every field name in
    the copy line it is trying to look past — and, since D-019, in a docstring
    that names the fields too.
    """
    tree = ast.parse((_ROOT / module).read_text(encoding="utf-8"))
    body = next(n.body for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    post_init = next(
        (n for n in body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None
    )
    if post_init is None:
        return "", ""
    checking: list[str] = []
    setattrs: list[str] = []
    for index, stmt in enumerate(post_init.body):
        if index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # the docstring
        text = ast.unparse(stmt)
        if text.startswith("object.__setattr__"):
            setattrs.append(text)
        else:
            checking.append(text)
    return "\n".join(checking), "\n".join(setattrs)


def test_every_validated_container_field_is_also_kept() -> None:
    """The rule D-018 should have used, derived rather than listed (#202, D-019).

    D-018 keyed its population on *"is the annotation a mutable container"*. That
    is not the property its own sentence describes — "every check above runs
    against the caller's own object and the record then stores that same object,
    so the validation was a **snapshot rather than an invariant**". The property
    is **"does `__post_init__` validate this field"**, and three `tuple`-annotated
    fields answered yes while the annotation-keyed rule looked past them.

    An annotation is a hint. `_validate_per_query` and the rank-order container
    check both accept any sequence *by design and in writing*, so a `tuple`
    annotation says nothing about what is actually stored — which is precisely
    why keying on it missed. Keyed on validation, the line falls in the right
    place with no list to maintain: `LateChunk.vector` and
    `ValidationReport.findings` are tuple-annotated and validated nowhere, so
    there is no snapshot there to turn into an invariant.
    """
    offenders = []
    for module, cls, frozen, fields in _dataclasses_in_package():
        if not frozen:
            continue
        checking, setattrs = _post_init_parts(module, cls)
        if not checking:
            continue
        for name, annotation in fields:
            if not _is_container(annotation):
                continue
            if name not in checking:
                continue
            if f"object.__setattr__(self, {name!r}" not in setattrs:
                offenders.append(f"{module}:{cls}.{name} ({annotation})")
    assert not offenders, (
        f"these frozen records validate a container field and then store the "
        f"caller's object: {offenders}. `frozen=True` stops a rebind and nothing "
        f"else, so the check is a snapshot rather than an invariant (#202)."
    )


def test_the_validated_and_unvalidated_container_fields_are_both_named() -> None:
    """A pass over an empty set is not a pass, and the split is the finding.

    Both halves are pinned by value: the validated set is what the arm above
    walks, and the unvalidated set is what it deliberately does not. Adding a
    validator to one of the bottom two moves it across on its own and the arm
    above will then require a copy — which is the behaviour a hand-written list
    of exclusions cannot have.

    **The two population arms are a pair, and neither is a superset.** A field
    must be copied if its annotation is a mutable container *or* if
    `__post_init__` validates it. `Chunk.metadata` satisfies only the first
    (nothing validates it, which is why its copy is the deep one) and
    `RetrievalRun.per_query` satisfies only the second — so covering either
    condition alone leaves a real row exposed, and D-018 covering only the first
    is how three rows stayed exposed.
    """
    validated: set[str] = set()
    unvalidated: set[str] = set()
    for module, cls, frozen, fields in _dataclasses_in_package():
        if not frozen:
            continue
        checking, _ = _post_init_parts(module, cls)
        for name, annotation in fields:
            if not _is_container(annotation):
                continue
            (validated if name in checking else unvalidated).add(f"{cls}.{name}")
    assert validated == {
        "RetrievalRun.recall_at_k",
        "RetrievalRun.snippet_hit_at_k",
        "RetrievalRun.notes",
        "RetrievalRun.per_query",
        "QueryResult.retrieved_doc_ids_in_rank_order",
        "QueryResult.snippet_hits_in_rank_order",
    }, f"the validated container fields are now {sorted(validated)}"
    assert unvalidated == {
        "Chunk.metadata",
        "LateChunk.vector",
        "ValidationReport.findings",
    }, (
        f"the unvalidated container fields are now {sorted(unvalidated)}. If one "
        f"gained a validator it needs a copy too; if one was added, decide it "
        f"rather than widening this literal."
    )
    # `Chunk.metadata` is the reason the two arms are a *pair* rather than one
    # rule. It is validated nowhere -- `dict[str, Any]`, which is exactly why its
    # copy has to be deep -- so this arm does not reach it, and the
    # mutable-annotation arm above does. Neither is a superset of the other, and
    # the two fields left in `unvalidated` below it are caught by neither, which
    # is the decision.
    assert "Chunk.metadata" not in validated
    chunk_setattrs = _post_init_parts("chunking_lab/strategies/__init__.py", "Chunk")[1]
    assert "object.__setattr__(self, 'metadata'" in chunk_setattrs
