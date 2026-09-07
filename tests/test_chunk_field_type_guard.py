"""`Chunk` is a construction boundary and must guard like every other (#180).

#29/#31 put `isinstance(x, int) and not isinstance(x, bool)` on nine numeric
fields across all five strategy classes. `Chunk.__post_init__` validated only
the *ordering* of its offsets — and its two numeric fields are the ones this
module's central invariant is about:

    source_text[start_offset:end_offset] == chunk.text

`Chunk(text="Hello", start_offset=True, end_offset=5)` was accepted and slices
`'ello'`. Silently: nothing raises, at construction or after.

The policy arm below is **discovered**, not listed. A hand-written table of
"classes that must reject `True`" is exactly the artifact that would have
missed `Chunk` — the same mistake, one level up, that #176 caught in #167's
"all five `chunk()` methods".
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import chunking_lab
from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies import Chunk

SRC = "Hello world, this is the source text."


def _chunk(**over: Any) -> Chunk:
    kwargs: dict[str, Any] = {
        "text": "Hello",
        "start_offset": 0,
        "end_offset": 5,
        "source_doc_id": "a.md",
        "strategy_name": "fixed-size",
    }
    kwargs.update(over)
    return Chunk(**kwargs)


# --------------------------------------------------------------------------
# The control, first — so nothing below can pass by refusing everything
# --------------------------------------------------------------------------


def test_a_well_formed_chunk_is_still_accepted_and_still_holds_the_invariant() -> None:
    c = _chunk()
    assert SRC[c.start_offset : c.end_offset] == c.text


# --------------------------------------------------------------------------
# The rows from the issue's measured table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("bool True/int", True, 5),
        ("bool False/bool True", False, True),
        ("int/bool", 0, True),
        ("float whole", 0.0, 5.0),
        ("float fractional", 0.5, 5.5),
        ("str numeral", "0", "5"),
        ("None", None, 5),
        ("complex", complex(0, 0), 5),
    ],
)
def test_a_non_int_offset_is_rejected_as_a_valueerror(label: str, start: Any, end: Any) -> None:
    """`ValueError`, not merely "raises".

    `'0'` already failed before this change — as a bare `TypeError` out of the
    `start_offset < 0` comparison. `check_chunk_input` raises `ValueError`, so a
    caller wrapping chunk production in `except ValueError` caught one road and
    not the other. Asserting the class is what makes the two agree.
    """
    with pytest.raises(ValueError, match="must be an int"):
        _chunk(start_offset=start, end_offset=end)


def test_the_bool_row_is_the_one_that_used_to_fail_silently() -> None:
    """Recorded as an executable fact, because it is why `bool` gets its own arm.

    Every other rejected row announced itself somewhere — a `TypeError` at the
    slice, or at the `<`. `True` produced a chunk whose recorded span does not
    contain its own text, and nothing anywhere noticed.
    """
    assert SRC[True:5] == "ello"
    assert SRC[True:5] != "Hello"
    with pytest.raises(ValueError, match="must be an int"):
        _chunk(start_offset=True)


@pytest.mark.parametrize("field", ["text", "source_doc_id", "strategy_name"])
@pytest.mark.parametrize(
    ("label", "value"),
    [("None", None), ("int", 123), ("bytes", "Café".encode()), ("Path", Path("a.md"))],
    ids=["None", "int", "bytes", "Path"],
)
def test_a_non_str_string_field_is_rejected(field: str, label: str, value: Any) -> None:
    """`check_chunk_input` guards the strategies' *inputs*; nothing guarded this.

    `Chunk` is on `chunking_lab.strategies.__all__` and constructible directly,
    so `source_doc_id=None` reached the metrics attribution key — the #176 harm
    — without going through a strategy at all.
    """
    with pytest.raises(ValueError, match="must be a str"):
        _chunk(**{field: value})


def test_the_ordering_guards_still_fire_and_still_come_second() -> None:
    """Range checks are unchanged, and only ever see real ints now.

    Order matters: `'0' < 0` raises `TypeError` before any field-named message
    could be produced, which is exactly what the `str` row above used to hit.
    """
    with pytest.raises(ValueError, match="start_offset must be >= 0"):
        _chunk(start_offset=-1, end_offset=5)
    with pytest.raises(ValueError, match="must be >= start_offset"):
        _chunk(start_offset=5, end_offset=2)


# --------------------------------------------------------------------------
# The neighbouring fix that reads as complete
# --------------------------------------------------------------------------


def test_a_plain_isinstance_int_check_would_not_have_been_enough() -> None:
    """`isinstance(v, int)` is the obvious guard and admits the silent row.

    `bool` subclasses `int` in Python, so the check passes for `True` — and
    `True` is the single row in the table that produces a wrong answer instead
    of an exception. That is why #29/#31 wrote the exclusion explicitly nine
    times rather than relying on `isinstance(..., int)`.
    """
    assert isinstance(True, int), "sanity: this is why the plain check reads as correct"
    neighbour_accepts = isinstance(True, int)
    assert neighbour_accepts
    with pytest.raises(ValueError, match="must be an int"):
        _chunk(start_offset=True)


# --------------------------------------------------------------------------
# The policy, discovered rather than listed
# --------------------------------------------------------------------------


def _package_dataclasses() -> list[type]:
    """Every dataclass defined in `chunking_lab`, found by walking the package."""
    found: dict[str, type] = {}
    pkg_dir = Path(chunking_lab.__file__).parent
    for mod_info in pkgutil.walk_packages([str(pkg_dir)], prefix="chunking_lab."):
        module = importlib.import_module(mod_info.name)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if dataclasses.is_dataclass(obj) and obj.__module__.startswith("chunking_lab"):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(found.values())


def _numeric_fields(cls: type) -> list[str]:
    """Fields annotated `int` or `float` (the annotation is a string under PEP 563).

    `float` is in the population deliberately. An earlier draft walked `int`
    only, found seven fields, and missed `RetrievalRun.wall_clock_ms` — which
    has exactly the same read/write asymmetry and whose bool case was already
    measured rendering as a fabricated `1` ms in `results/summary.md`. The
    annotation the field happens to carry is not the shape of the defect.
    """
    return [f.name for f in dataclasses.fields(cls) if str(f.type).strip() in ("int", "float")]


CLASSES_WITH_NUMERIC_FIELDS = [
    (cls, name) for cls in _package_dataclasses() for name in _numeric_fields(cls)
]


def _is_required(f: dataclasses.Field[Any]) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _synthesise(f: dataclasses.Field[Any]) -> Any:
    """A plausible valid value for a required field, from its annotation.

    Synthesised rather than skipped. An earlier draft of this file just
    `pytest.skip`ped any class it could not construct from a fixed kwargs table,
    and skipped **9 of the discovered fields** — including every field on the
    two embedder-taking strategies and on `RetrievalRun`. A policy arm that
    silently skips most of its population is the vacuous-discovery failure the
    docstring at the top of this file is about, one layer in.
    """
    annotation = str(f.type).strip()
    if annotation.startswith("Embedder") or f.name == "embedder":
        return HashEmbedder()
    known: dict[str, Any] = {
        "int": 2,
        "float": 0.5,
        "str": "x",
        "bool": False,
        "Path": Path("a.md"),
        "Chunk": Chunk("t", 0, 1, "a.md", "s"),
        "dict[int, float]": {1: 1.0},
        "dict[str, Any]": {},
        "tuple[float, ...]": (0.1, 0.2),
        "tuple[str, ...]": ("a",),
        "tuple[bool, ...]": (True,),
    }
    if annotation in known:
        return known[annotation]
    if annotation.startswith("tuple["):
        return ()
    if annotation.startswith(("dict[", "Mapping[")):
        return {}
    if annotation.startswith(("list[", "Sequence[")):
        return []
    if annotation.startswith(("str |", "Optional[")) or annotation.endswith("| None"):
        return None
    raise AssertionError(
        f"no synthesised value for {f.name}: {annotation!r} — teach _synthesise rather "
        "than skipping, or the policy arm silently stops covering this class"
    )


def test_the_discovery_finds_the_classes_the_sweep_covered() -> None:
    """A discovery that finds nothing passes the policy arm vacuously.

    Named floors rather than a bare count: these are the classes #29/#31 and
    #180 actually guard, so if the walk stops seeing them the arm below has
    stopped meaning anything.
    """
    found = {f"{cls.__name__}.{field}" for cls, field in CLASSES_WITH_NUMERIC_FIELDS}
    for expected in (
        "Chunk.start_offset",
        "Chunk.end_offset",
        "FixedSizeStrategy.chunk_chars",
        "StructureAwareStrategy.max_heading_level",
        "RetrievalRun.n_queries",
        "RetrievalRun.wall_clock_ms",
        "ValidationReport.n_valid",
    ):
        assert expected in found, f"discovery lost {expected}; found {sorted(found)}"


@pytest.mark.parametrize(
    ("cls", "field"),
    CLASSES_WITH_NUMERIC_FIELDS,
    ids=[f"{c.__name__}.{f}" for c, f in CLASSES_WITH_NUMERIC_FIELDS],
)
def test_every_int_annotated_dataclass_field_rejects_a_bool(cls: type, field: str) -> None:
    """The #29/#31 policy, applied to a population nobody has to maintain.

    A new dataclass with an `int` field is covered the day it is written, which
    is the property `Chunk` needed and a hand-listed table cannot have.
    """
    kwargs = {f.name: _synthesise(f) for f in dataclasses.fields(cls) if _is_required(f)}
    kwargs[field] = True
    try:
        cls(**kwargs)
    except ValueError:
        return  # the policy holds
    pytest.fail(f"{cls.__name__}.{field} accepted True; #29/#31 policy says int-and-not-bool")


# --------------------------------------------------------------------------
# The downstream harm the bool row caused
# --------------------------------------------------------------------------


def test_a_bool_offset_can_no_longer_collapse_the_metrics_tie_break() -> None:
    """`metrics.py`'s sort breaks score ties on "the chunk's stable identity".

    `True == 1`, so two chunks differing only there tie on all four key
    components and fall back to insertion order — the corpus-iteration-order
    dependence #68 exists to remove. Constructing the colliding pair is now
    impossible, which is asserted here rather than in `metrics`, because the
    boundary is where it is prevented.
    """
    assert (0, "a.md", True, 5) == (0, "a.md", 1, 5), "sanity: True and 1 tie in the sort key"
    ok = _chunk(start_offset=1)
    with pytest.raises(ValueError, match="must be an int"):
        _chunk(start_offset=True)
    assert ok.start_offset == 1
