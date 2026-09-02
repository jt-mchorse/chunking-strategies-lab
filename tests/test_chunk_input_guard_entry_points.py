"""Every public chunk-producing entry point validates its inputs (#167, #176).

`check_chunk_input` was introduced by #167 so the strategies cannot disagree
about what they accept, and its docstring stated the coverage claim as *"called
at the top of all five `chunk()` methods."* The count was right and the
population was wrong: there are five strategies but **six** entry points.
`LateChunkingStrategy.chunk_with_vectors` is the surface D-006 exists for, the
one its sibling `chunk()` delegates to, and the one `_materialize_vectors`
deliberately routes late chunking through -- and it did not call the guard.

So the shipped evaluator took the unguarded road for exactly one of the five
strategies. On a `Document` whose `filename` was a `Path` (the likely wrong
type, since the field is built from `path.name`) the other four raised, and
late chunking reported ``recall={1: 0.0, 3: 0.0}`` next to
``snippet={1: 1.0, 3: 1.0}`` -- the right chunk retrieved, and only the
attribution key failing to compare equal.

The lesson is in the shape of this file, not just its assertions: a *list* of
sites cannot see a new member, however carefully it was counted. The entry
points below are **discovered** from the exported strategy classes, so a
seventh is covered without anyone editing this file. Two arms guard the
discovery itself, because a discovery that quietly finds nothing passes
vacuously and is worse than the list it replaced.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from chunking_lab import strategies as strategies_module
from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import evaluate_strategy
from chunking_lab.queries import Query
from chunking_lab.strategies import (
    Chunk,
    FixedSizeStrategy,
    LateChunk,
    LateChunkingStrategy,
    RecursiveStrategy,
    SemanticBoundaryStrategy,
    StructureAwareStrategy,
)

_EMBEDDER = HashEmbedder()

#: One constructed instance per exported strategy class. Constructed here (not
#: discovered) because each takes different arguments; the *entry points* are
#: what gets discovered, and that is the axis #176 was missed on.
STRATEGY_INSTANCES: list[tuple[str, Any]] = [
    ("fixed", FixedSizeStrategy(chunk_chars=40, overlap_chars=5)),
    ("recursive", RecursiveStrategy(chunk_chars=40)),
    ("semantic", SemanticBoundaryStrategy(embedder=_EMBEDDER)),
    ("late", LateChunkingStrategy(embedder=_EMBEDDER, chunk_chars=40, overlap_chars=5)),
    ("structure", StructureAwareStrategy()),
]


def _exported_strategy_classes() -> list[type]:
    """Strategy classes on the package's `__all__`, minus the data types."""
    excluded = {"Chunk", "LateChunk", "Strategy"}
    out = []
    for name in strategies_module.__all__:
        if name in excluded:
            continue
        obj = getattr(strategies_module, name)
        if inspect.isclass(obj):
            out.append(obj)
    return out


def _chunk_entry_points(instance: Any) -> list[str]:
    """Public methods on `instance` that produce chunks.

    Discovered from the return annotation rather than the name, so a future
    `segment()` or `split_with_vectors()` is found too -- a name-prefix rule
    would be the same hand-listing mistake one level up. The annotation is a
    string here (`from __future__ import annotations` is on in every strategy
    module), which is exactly what makes this cheap and dependency-free.
    """
    found = []
    for name, member in inspect.getmembers(instance, predicate=callable):
        if name.startswith("_"):
            continue
        annotation = inspect.get_annotations(member).get("return")
        rendered = (
            annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
        )
        if Chunk.__name__ in str(rendered) or LateChunk.__name__ in str(rendered):
            found.append(name)
    return sorted(found)


ALL_ENTRY_POINTS: list[tuple[str, Any, str]] = [
    (label, instance, method)
    for label, instance in STRATEGY_INSTANCES
    for method in _chunk_entry_points(instance)
]


# --- arms guarding the discovery itself --------------------------------------


def test_the_discovery_finds_at_least_one_entry_point_per_strategy() -> None:
    """A discovery that finds nothing passes every table below vacuously."""
    for label, instance in STRATEGY_INSTANCES:
        assert _chunk_entry_points(instance), f"{label}: no chunk entry point discovered"


def test_the_discovery_finds_more_entry_points_than_strategies() -> None:
    """The whole point of #176: entry points outnumber strategy classes.

    If this ever reads "equal", either `chunk_with_vectors` stopped being
    discoverable or someone narrowed the discovery back to `chunk()` -- which
    is the exact state #167 shipped in and #176 found.
    """
    assert len(ALL_ENTRY_POINTS) > len(STRATEGY_INSTANCES), ALL_ENTRY_POINTS
    assert ("late", "chunk_with_vectors") in {
        (label, method) for label, _, method in ALL_ENTRY_POINTS
    }, ALL_ENTRY_POINTS


def test_every_exported_strategy_class_is_represented() -> None:
    """And the instance table cannot fall behind the package's `__all__`."""
    exported = {cls.__name__ for cls in _exported_strategy_classes()}
    covered = {type(instance).__name__ for _, instance in STRATEGY_INSTANCES}
    assert exported == covered, f"exported={sorted(exported)} covered={sorted(covered)}"


# --- the guard itself, over every discovered entry point ---------------------

#: #167's own input table, verbatim, plus `Path` -- the wrong type a caller is
#: most likely to actually produce, since `Document.filename` is built from
#: `path.name`.
BAD_TEXT: list[tuple[str, Any]] = [
    ("int", 123),
    ("bytes", "Café — 中文".encode()),
    ("list", ["ab", "cd"]),
    ("None", None),
]

BAD_DOC_ID: list[tuple[str, Any]] = [
    ("None", None),
    ("int", 123),
    ("Path", Path("a.md")),
    ("bytes", b"a.md"),
]

_IDS = [f"{label}.{method}" for label, _, method in ALL_ENTRY_POINTS]


@pytest.mark.parametrize(("label", "instance", "method"), ALL_ENTRY_POINTS, ids=_IDS)
@pytest.mark.parametrize(("kind", "bad_text"), BAD_TEXT, ids=[k for k, _ in BAD_TEXT])
def test_every_entry_point_rejects_a_non_str_text(
    label: str, instance: Any, method: str, kind: str, bad_text: Any
) -> None:
    with pytest.raises(ValueError, match="text must be a str"):
        getattr(instance, method)(bad_text, source_doc_id="doc")


@pytest.mark.parametrize(("label", "instance", "method"), ALL_ENTRY_POINTS, ids=_IDS)
@pytest.mark.parametrize(("kind", "bad_id"), BAD_DOC_ID, ids=[k for k, _ in BAD_DOC_ID])
def test_every_entry_point_rejects_a_non_str_source_doc_id(
    label: str, instance: Any, method: str, kind: str, bad_id: Any
) -> None:
    with pytest.raises(ValueError, match="source_doc_id must be a str"):
        getattr(instance, method)("some ordinary text to chunk", source_doc_id=bad_id)


@pytest.mark.parametrize(("label", "instance", "method"), ALL_ENTRY_POINTS, ids=_IDS)
def test_every_entry_point_still_accepts_the_control(
    label: str, instance: Any, method: str
) -> None:
    """The guard must not have hardened into refusing ordinary input."""
    out = getattr(instance, method)("# Heading\n\nSome ordinary prose.\n", source_doc_id="a.md")
    assert out, f"{label}.{method} produced no chunks for valid input"


# --- the harm, end to end through the evaluator ------------------------------


_QUERIES = [
    Query(
        id="q1",
        question="how are embeddings indexed",
        expected_doc="a.md",
        expected_snippet="index embeddings for similarity",
    )
]
_TEXT = "Vector databases index embeddings for similarity search. " * 8


@pytest.mark.parametrize(
    ("label", "instance"), STRATEGY_INSTANCES, ids=[label for label, _ in STRATEGY_INSTANCES]
)
@pytest.mark.parametrize("bad_filename", [Path("a.md"), None, 123], ids=["Path", "None", "int"])
def test_evaluate_strategy_fails_the_same_way_for_every_strategy(
    label: str, instance: Any, bad_filename: Any
) -> None:
    """The measured divergence, asserted as *agreement* rather than as a raise.

    Before #176 this table split in half: four strategies raised the guard's
    `ValueError` and late chunking returned a `RetrievalRun` scoring
    `recall={1: 0.0, 3: 0.0}` while `snippet={1: 1.0, 3: 1.0}`. The recall
    number is the one this repo publishes, and it said late chunking found
    nothing on a run where it had found everything.
    """
    corpus = [Document(filename=bad_filename, text=_TEXT)]
    with pytest.raises(ValueError, match="source_doc_id must be a str"):
        evaluate_strategy(instance, corpus, _QUERIES, _EMBEDDER, ks=(1, 3))


@pytest.mark.parametrize(
    ("label", "instance"), STRATEGY_INSTANCES, ids=[label for label, _ in STRATEGY_INSTANCES]
)
def test_evaluate_strategy_still_works_on_a_well_formed_corpus(label: str, instance: Any) -> None:
    """Anti-vacuous partner to the arm above: a `str` filename must still run.

    Without this, a guard that rejected *every* corpus would satisfy the whole
    table above -- and "every strategy now raises" is not the fix.

    Deliberately weak on the *scores*: whether a given strategy's chunking puts
    the expected snippet inside one chunk depends on its parameters, and pinning
    that here would make this file a metrics test that breaks when someone tunes
    `chunk_chars`. The score assertion that matters lives in the focused late
    arm below, where the numbers were actually measured.
    """
    corpus = [Document(filename="a.md", text=_TEXT)]
    run = evaluate_strategy(instance, corpus, _QUERIES, _EMBEDDER, ks=(1, 3))
    assert set(run.recall_at_k) == {1, 3}, run.recall_at_k
    for k, v in run.recall_at_k.items():
        assert 0.0 <= v <= 1.0, f"{label}: recall@{k} = {v}"


def test_late_chunking_attributes_correctly_on_a_str_filename() -> None:
    """The measured control, with the parameters the divergence was measured at.

    `recall={1: 0.0, 3: 0.0}` beside `snippet={1: 1.0, 3: 1.0}` was the
    signature of the bug: the right chunk retrieved, and only `source_doc_id`
    failing to compare equal to the query's `expected_doc`. With a well-formed
    filename both read 1.0, which is what makes the raise in
    `test_evaluate_strategy_fails_the_same_way_for_every_strategy` a *fix*
    rather than a blanket refusal.
    """
    late = LateChunkingStrategy(embedder=_EMBEDDER, chunk_chars=60, overlap_chars=10)
    corpus = [Document(filename="a.md", text=_TEXT)]
    run = evaluate_strategy(late, corpus, _QUERIES, _EMBEDDER, ks=(1, 3))
    assert run.snippet_hit_at_k[1] == 1.0, "the snippet was not retrieved; wrong control"
    assert run.recall_at_k[1] == 1.0, (
        f"snippet hit but recall is {run.recall_at_k[1]} -- the attribution key is wrong, "
        "which is the #176 signature"
    )
