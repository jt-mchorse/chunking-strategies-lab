"""An invisible character in golden data must not be published as a measurement.

`Query.__post_init__` and `validate_queries` both rejected an empty or
whitespace-only field, with a comment explaining that "a whitespace-only field
is as corrupting as an empty one". That is true for every codepoint Python calls
whitespace — U+00A0, U+3000 and U+202F are all removed by `str.strip()`. It is
not true for the Unicode *format* (Cf) characters, which `str.isspace()`
deliberately excludes: U+200B ZERO WIDTH SPACE, U+FEFF, U+2060 WORD JOINER,
U+00AD SOFT HYPHEN, U+180E.

Those are invisible in an editor and survive `.strip()`, so before #162 a single
one of them in a golden query flipped a published recall@k or snippet-hit@k from
1.000 to 0.000 while the CI validator reported the file CLEAN. A 0.000 produced
that way is indistinguishable from an honest "this strategy retrieved nothing
relevant" — the same confusion #160 closed for the unmeasured case, reached here
through the golden data rather than through a missing key.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import evaluate_strategy
from chunking_lab.queries import Query, find_format_char, load_queries
from chunking_lab.strategies import FixedSizeStrategy
from chunking_lab.validate import validate_queries

ZWSP = "​"
BOM = "﻿"
WORD_JOINER = "⁠"
SOFT_HYPHEN = "­"
MONGOLIAN_VS = "᠎"

# Whitespace codepoints the pre-existing `.strip()` rule already handled. These
# are here to show the new rule is an extension, not a replacement.
ALREADY_COVERED = [" ", "　", " ", " ", " ", "\t"]

INVISIBLE = [ZWSP, BOM, WORD_JOINER, SOFT_HYPHEN, MONGOLIAN_VS]


def _row(**over):
    base = {
        "id": "q1",
        "question": "How long is the refund window?",
        "expected_doc": "d1.md",
        "expected_snippet": "refund window",
    }
    base.update(over)
    return base


def _write(tmp_path, row):
    p = tmp_path / "queries.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# The premise: which codepoints survive strip()
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ch", ALREADY_COVERED)
def test_whitespace_codepoints_are_removed_by_strip(ch):
    """Pins why the old rule looked sufficient — it covers a lot."""
    assert ch.isspace()
    assert ch.strip() == ""


@pytest.mark.parametrize("ch", INVISIBLE)
def test_format_codepoints_survive_strip(ch):
    """And pins the gap: these are invisible but `.strip()` keeps them."""
    assert not ch.isspace()
    assert ch.strip() == ch
    assert unicodedata.category(ch) == "Cf"


# ----------------------------------------------------------------------
# The guard, on both readers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ch", INVISIBLE)
@pytest.mark.parametrize("field", ["id", "expected_doc", "expected_snippet"])
def test_query_rejects_an_invisible_character_in_a_matched_field(field, ch):
    with pytest.raises(ValueError, match="invisible character"):
        Query(**_row(**{field: _row()[field] + ch}))


@pytest.mark.parametrize("ch", INVISIBLE)
@pytest.mark.parametrize("field", ["id", "expected_doc", "expected_snippet"])
def test_validator_reports_the_same_rows_the_loader_rejects(tmp_path, field, ch):
    """The loader and the linter must not disagree about a file.

    A differential probe over 24 malformed-file shapes showed exact parity
    before this change; the new rule has to land on both sides or CI would pass
    a file the runtime refuses.
    """
    path = _write(tmp_path, _row(**{field: _row()[field] + ch}))

    report = validate_queries(path)
    assert [f.code for f in report.findings] == [f"invisible_char_{field}"]

    with pytest.raises(ValueError, match="invisible character"):
        load_queries(path)


def test_a_wholly_invisible_field_is_rejected():
    """`ZWSP` alone is non-empty and `.strip()`s to itself, so it used to pass."""
    assert ZWSP.strip() != ""
    with pytest.raises(ValueError, match="invisible character"):
        Query(**_row(expected_snippet=ZWSP))


def test_message_names_the_codepoint_and_index_not_just_the_value():
    """The character is invisible; echoing the field back looks identical to a
    correct one, so the message has to name what and where."""
    with pytest.raises(ValueError, match="invisible character") as exc:
        Query(**_row(expected_doc="d1.md" + ZWSP))
    msg = str(exc.value)
    assert "U+200B" in msg
    assert "ZERO WIDTH SPACE" in msg
    assert "index 5" in msg


def test_find_format_char_returns_the_first_occurrence():
    assert find_format_char("clean") is None
    index, label = find_format_char("ab" + BOM + "cd" + ZWSP)
    assert index == 2
    assert label.startswith("U+FEFF")


# ----------------------------------------------------------------------
# Deliberate non-coverage
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mark", ["‎", "‏"])
def test_question_still_accepts_directional_marks(mark):
    """`question` is only embedded, never matched, and RTL text legitimately
    carries LRM/RLM. Scoping the rule away from it is deliberate."""
    Query(**_row(question="ما هي مدة الاسترجاع؟" + mark))


def test_snippet_still_accepts_an_embedded_newline():
    """A newline is category Cc, and a snippet spanning a line break is normal.
    This is why the rule is Cf-only rather than "all invisible categories"."""
    Query(**_row(expected_snippet="refund window\nis thirty days"))


def test_ordinary_unicode_text_is_unaffected():
    for value in ["café.md", "文档.md", "doc-2026_v1.md", "a/b/c.md#3"]:
        Query(**_row(expected_doc=value))


def test_existing_empty_and_whitespace_rules_still_fire():
    with pytest.raises(ValueError, match="non-empty"):
        Query(**_row(expected_doc=""))
    with pytest.raises(ValueError, match="non-empty"):
        Query(**_row(expected_doc="   "))


# ----------------------------------------------------------------------
# End to end: the published metric is what actually moved
# ----------------------------------------------------------------------


def _run_metric(row):
    corpus = [
        Document(filename="d1.md", text="The refund window is thirty days from delivery. " * 6),
        Document(filename="d2.md", text="Shipping takes three to five business days. " * 6),
    ]
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=10),
        corpus,
        [Query(**row)],
        HashEmbedder(),
        ks=(1, 3),
        dataset_version="v1",
    )
    return run.recall_at_k[3], run.snippet_hit_at_k[3]


def test_control_row_scores_a_genuine_one():
    """The baseline the corrupted rows were measured against."""
    assert _run_metric(_row()) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("field", "ch"),
    [
        ("expected_doc", ZWSP),
        ("expected_doc", BOM),
        ("expected_snippet", ZWSP),
        ("expected_snippet", BOM),
        ("expected_snippet", SOFT_HYPHEN),
    ],
)
def test_the_corrupting_rows_can_no_longer_reach_the_metric(field, ch):
    """Before #162 each of these produced a confident 0.000 on one axis while
    the other axis kept reading 1.000 — which is what made the result look like
    a real measurement rather than a data defect."""
    with pytest.raises(ValueError, match="invisible character"):
        _run_metric(_row(**{field: _row()[field] + ch}))
