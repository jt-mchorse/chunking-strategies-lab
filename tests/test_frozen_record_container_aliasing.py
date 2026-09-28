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
from chunking_lab.metrics import RetrievalRun, _validate_metric_maps, _validate_notes
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


def test_the_immutable_container_rows_are_cleared_by_name() -> None:
    """`tuple`-typed fields on frozen records, recorded as a result.

    Every one holds either a scalar or a frozen record whose own fields are
    tuples, so there is nothing a caller can edit in place. Named here so the
    next `portfolio-ops#71`-style sweep reads a decision, and so that retyping
    one of them to a `list` trips the arm above rather than passing quietly.
    """
    tuple_rows = {
        f"{cls}.{name}"
        for _, cls, frozen, fields in _dataclasses_in_package()
        if frozen
        for name, ann in fields
        if ann.split("[", 1)[0].split(".")[-1] == "tuple"
    }
    assert tuple_rows == {
        "QueryResult.retrieved_doc_ids_in_rank_order",
        "QueryResult.snippet_hits_in_rank_order",
        "RetrievalRun.per_query",
        "LateChunk.vector",
        "ValidationReport.findings",
    }, (
        f"the frozen tuple-typed fields are now {sorted(tuple_rows)}. A new one "
        f"needs the same check; a retyped one needs a copy."
    )
