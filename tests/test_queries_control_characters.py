"""The Cc block, swept whole (#171).

`#162` closed the Unicode *format* (`Cf`) hole and excluded *control* (`Cc`) on
one stated reason, pinned next door by
`test_snippet_still_accepts_an_embedded_newline`:

    A newline is category Cc, and a snippet spanning a line break is normal.
    This is why the rule is Cf-only rather than "all invisible categories".

The reason is true, and it justifies excluding **newline**. `Cc` has 65 members.
The reason reaches at most the ten `str.isspace()` already recognises; the other
55 -- NUL, BEL, BACKSPACE, ESC, DELETE, and the whole C1 block -- are invisible,
survive `.strip()`, and are not legitimate inside an id, a filename, or a
snippet. Measured on the harness `#162` itself used, changing nothing but one
appended character in the golden data::

    control (clean)     recall@3 1.000   snippet@3 1.000   validator CLEAN
    U+0000 NUL          recall@3 0.000   snippet@3 0.000   validator CLEAN
    U+0007 BEL          recall@3 0.000   snippet@3 0.000   validator CLEAN
    U+001B ESC          recall@3 0.000   snippet@3 0.000   validator CLEAN
    U+007F DELETE       recall@3 0.000   snippet@3 0.000   validator CLEAN
    U+009B CSI          recall@3 0.000   snippet@3 0.000   validator CLEAN

Byte-for-byte the failure `#162` exists to prevent, reached through the sibling
category.

**The block is swept whole rather than sampled.** `Cc` is 65 enumerable
codepoints; picking six and calling it covered is how a rule ends up as wide as
whoever wrote the examples rather than as wide as its own justification -- which
is the defect this file is fixing. Every character is built from `chr(cp)`, never
written as a literal: a literal control character does not survive a copy-paste,
and several of them (BACKSPACE, ESC, CSI) actively rewrite the rendering of
whatever is echoed after them.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import evaluate_strategy
from chunking_lab.queries import (
    MATCHED_FIELDS,
    Query,
    find_invisible_char,
    load_queries,
)
from chunking_lab.strategies import FixedSizeStrategy
from chunking_lab.validate import validate_queries

#: Every codepoint Python assigns general category Cc: U+0000-U+001F, U+007F,
#: U+0080-U+009F. Derived, not typed out, so the sweep cannot drift from the
#: category it claims to cover.
CC_BLOCK = [
    cp
    for cp in list(range(0x00, 0x20)) + [0x7F] + list(range(0x80, 0xA0))
    if unicodedata.category(chr(cp)) == "Cc"
]

#: The partition the rule is built on. `REJECTED` is every Cc character
#: `str.isspace()` does not recognise; `STILL_LEGAL` is the ten it does -- and
#: the ten are exactly what the newline argument covers.
REJECTED = [cp for cp in CC_BLOCK if not chr(cp).isspace()]
STILL_LEGAL = [cp for cp in CC_BLOCK if chr(cp).isspace()]


def _label(cp: int) -> str:
    return f"U+{cp:04X}"


def test_the_block_is_complete_and_partitioned() -> None:
    """Anti-vacuous for the sweep itself: if `CC_BLOCK` silently shrank, every
    parametrized test below would pass by covering nothing."""
    assert len(CC_BLOCK) == 65
    assert len(REJECTED) == 55
    assert len(STILL_LEGAL) == 10
    assert set(REJECTED) | set(STILL_LEGAL) == set(CC_BLOCK)
    assert not set(REJECTED) & set(STILL_LEGAL)
    # The ten are the ones the newline argument actually covers.
    assert {0x09, 0x0A, 0x0D} <= set(STILL_LEGAL)
    assert {0x00, 0x1B, 0x7F} <= set(REJECTED)


def _row(**over: object) -> dict:
    base: dict = {
        "id": "q1",
        "question": "How long is the refund window?",
        "expected_doc": "d1.md",
        "expected_snippet": "refund window",
    }
    base.update(over)
    return base


def _write(tmp_path, row: dict):
    p = tmp_path / "queries.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")
    return p


# --- the premise -------------------------------------------------------------


@pytest.mark.parametrize("cp", REJECTED, ids=_label)
def test_rejected_controls_survive_strip_and_are_invisible(cp: int) -> None:
    ch = chr(cp)
    assert unicodedata.category(ch) == "Cc"
    assert not ch.isspace()
    assert ch.strip() == ch  # the emptiness guard cannot see it
    assert not ch.isprintable()


# --- the rule, on both readers ----------------------------------------------


@pytest.mark.parametrize("cp", REJECTED, ids=_label)
@pytest.mark.parametrize("field", MATCHED_FIELDS)
def test_query_rejects_every_non_whitespace_control(field: str, cp: int) -> None:
    with pytest.raises(ValueError, match="invisible character"):
        Query(**_row(**{field: str(_row()[field]) + chr(cp)}))


@pytest.mark.parametrize("cp", [0x00, 0x07, 0x08, 0x1B, 0x7F, 0x9B], ids=_label)
@pytest.mark.parametrize("field", MATCHED_FIELDS)
def test_validator_reports_the_same_rows_the_loader_rejects(tmp_path, field: str, cp: int) -> None:
    """Loader/validator parity, the property `#162` established. A representative
    slice rather than all 55 x 3 x 2 -- the full block is swept on the loader
    above, and what this asserts is that the two readers agree, not that the
    rule is wide."""
    path = _write(tmp_path, _row(**{field: str(_row()[field]) + chr(cp)}))

    report = validate_queries(path)
    assert [f.code for f in report.findings] == [f"invisible_char_{field}"]

    with pytest.raises(ValueError, match="invisible character"):
        load_queries(path)


@pytest.mark.parametrize("cp", REJECTED, ids=_label)
def test_a_wholly_control_field_is_rejected(cp: int) -> None:
    """A lone control character is non-empty and strips to itself, so the
    "whitespace-only is as corrupting as empty" guard never fires on it -- the
    same reasoning `test_a_wholly_invisible_field_is_rejected` pins for ZWSP."""
    ch = chr(cp)
    assert ch.strip() != ""
    with pytest.raises(ValueError, match="invisible character"):
        Query(**_row(expected_snippet=ch))


def test_the_message_names_the_codepoint_and_index() -> None:
    with pytest.raises(ValueError, match="invisible character") as exc:
        Query(**_row(expected_doc="d1.md" + chr(0x1B)))
    msg = str(exc.value)
    assert "U+001B" in msg
    assert "index 5" in msg
    # The offending character must not be echoed into a message a terminal will
    # render -- ESC and CSI rewrite whatever follows them.
    assert chr(0x1B) not in msg


# --- what the newline argument actually protects ----------------------------


@pytest.mark.parametrize("cp", STILL_LEGAL, ids=_label)
def test_whitespace_controls_are_still_accepted(cp: int) -> None:
    """The exclusion's real motivation, kept intact and widened to its whole
    partition. `#162`'s reason was about a snippet spanning a line break; every
    codepoint here is one `str.isspace()` recognises, which is precisely the set
    that reason covers."""
    Query(**_row(expected_snippet="refund window" + chr(cp) + "is thirty days"))


def test_a_snippet_spanning_a_newline_still_loads() -> None:
    """The named case from `#162`, unchanged."""
    Query(**_row(expected_snippet="refund window\nis thirty days"))


@pytest.mark.parametrize("cp", REJECTED, ids=_label)
def test_question_still_accepts_every_control(cp: int) -> None:
    """`question` stays out of `MATCHED_FIELDS`. Its exclusion reason -- "only
    ever embedded, never compared" -- was re-checked rather than assumed:
    `metrics.evaluate_strategy` has the single use and it is
    `embedder.embed(q.question)`. Widening the character rule must not quietly
    widen the *field* scope too."""
    Query(**_row(question="How long is the refund window?" + chr(cp)))


# --- end to end: the corrupted row cannot reach the published metric ---------

CORPUS = [
    Document(filename="d1.md", text="The refund window is thirty days from delivery. " * 6),
    Document(filename="d2.md", text="Shipping takes three to five business days. " * 6),
]


def _run_metric(row: dict) -> tuple[float, float]:
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=10),
        CORPUS,
        [Query(**row)],
        HashEmbedder(),
        ks=(1, 3),
        dataset_version="v1",
    )
    return run.recall_at_k[3], run.snippet_hit_at_k[3]


def test_control_row_scores_a_genuine_one() -> None:
    """The baseline the corrupted rows were measured against -- without it, a
    0.000 below would prove nothing."""
    assert _run_metric(_row()) == (1.0, 1.0)


@pytest.mark.parametrize("cp", [0x00, 0x07, 0x08, 0x1B, 0x7F, 0x9B], ids=_label)
@pytest.mark.parametrize("field", ["expected_doc", "expected_snippet"])
def test_the_corrupting_rows_can_no_longer_reach_the_metric(field: str, cp: int) -> None:
    """Each of these produced a confident 0.000 on BOTH axes while
    `validate_queries` reported the file CLEAN."""
    with pytest.raises(ValueError, match="invisible character"):
        _run_metric(_row(**{field: str(_row()[field]) + chr(cp)}))


def test_find_invisible_char_reports_the_first_of_either_category() -> None:
    """Cf and Cc share one scan, so the index is the first invisible character
    of either kind -- not the first Cf then separately the first Cc."""
    zwsp = chr(0x200B)
    assert find_invisible_char("clean") is None
    index, label = find_invisible_char("ab" + chr(0x1B) + "cd" + zwsp)
    assert index == 2
    assert label.startswith("U+001B")
    index, label = find_invisible_char("ab" + zwsp + "cd" + chr(0x1B))
    assert index == 2
    assert label.startswith("U+200B")
