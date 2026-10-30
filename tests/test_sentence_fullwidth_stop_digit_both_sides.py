"""`．` is a decimal point only with a digit on BOTH sides (#326).

#324 (#321) guarded the unspaced `．` with `(?!\\d)`: it ended a sentence
only when no digit followed. A decimal point has a digit on both sides, so
that guard also swallowed every sentence that opens with a number. Measured
on `main` 357f43a: `会議は終わった．２０２６年に再開する．次に進んだ．` was 2
sentences (3 with `。` or with a space after `．`) and
`会議は終わった．5章を読む．` was 1.

The digit class is ASCII and fullwidth digits, not Unicode `\\d`: `．` is a
decimal point in CJK text, which writes its numbers in those two scripts.
The table below checks every placement of a digit around `．` against the
spaced form (branch (a) of `_SENTENCE_RE`, which has no guard): the two must
agree everywhere except the decimal rows.
"""

from __future__ import annotations

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies.semantic import (
    SemanticBoundaryStrategy,
    _split_sentences_with_offsets,
)

# One digit per script class; the boundary rows sit just outside each range.
DECIMAL_DIGITS = {"ascii": "5", "fullwidth": "５"}
OTHER_DIGITS = {"arabic-indic": "٥", "devanagari": "५", "math-bold": "𝟓"}
NEIGHBOURS = {"slash": "/", "colon": ":", "fw-slash": "／", "fw-colon": "："}
LETTER = "た"


def _sentences(text: str) -> list[str]:
    out = _split_sentences_with_offsets(text)
    for sentence, start in out:  # the offset contract stays exact
        assert text[start : start + len(sentence)] == sentence
    return [s for s, _ in out]


def _cases() -> list[tuple[str, str, str, bool]]:
    """(label, char before `．`, char after `．`, is a decimal point)."""
    rows: list[tuple[str, str, str, bool]] = []
    for name, d in DECIMAL_DIGITS.items():
        rows.append((f"{name}-before-only", d, LETTER, False))
        rows.append((f"{name}-after-only", LETTER, d, False))
        rows.append((f"{name}-both", d, d, True))
    rows.append(("ascii-then-fullwidth", "5", "５", True))
    rows.append(("fullwidth-then-ascii", "５", "5", True))
    for name, d in {**OTHER_DIGITS, **NEIGHBOURS}.items():
        rows.append((f"{name}-before-only", d, LETTER, False))
        rows.append((f"{name}-after-only", LETTER, d, False))
        rows.append((f"{name}-both", d, d, False))
    rows.append(("neither", LETTER, LETTER, False))
    return rows


@pytest.mark.parametrize(("label", "before", "after", "decimal"), _cases())
def test_unspaced_stop_splits_like_the_spaced_one_unless_it_is_a_decimal(
    label: str, before: str, after: str, decimal: bool
) -> None:
    unspaced = f"会議は{before}．{after}章を読む．"
    if decimal:
        assert _sentences(unspaced) == [unspaced], label
        return
    spaced = f"会議は{before}． {after}章を読む．"
    assert _sentences(spaced) == [f"会議は{before}．", f"{after}章を読む．"], label
    assert _sentences(unspaced) == _sentences(spaced), label


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        (
            "fullwidth-year",
            "会議は終わった．２０２６年に再開する．次に進んだ．",
            ["会議は終わった．", "２０２６年に再開する．", "次に進んだ．"],
        ),
        ("ascii-chapter", "会議は終わった．5章を読む．", ["会議は終わった．", "5章を読む．"]),
        ("fullwidth-chapter", "会議は終わった．３章を読む．", ["会議は終わった．", "３章を読む．"]),
        # #324's rows still hold: the digit-before split and the decimals.
        ("digit-before", "人数は２．次に進む．", ["人数は２．", "次に進む．"]),
        (
            "decimal-mid-sentence",
            "値は１．５です．次に進んだ．",
            ["値は１．５です．", "次に進んだ．"],
        ),
        ("version", "版は２．０．１です．", ["版は２．０．１です．"]),
    ],
)
def test_a_sentence_that_opens_with_a_number_is_its_own(
    label: str, text: str, expected: list[str]
) -> None:
    assert _sentences(text) == expected, label


def test_the_numbered_sentence_is_not_size_capped_mid_word() -> None:
    # The issue's measured harm: main merged the year sentence into the one
    # before it and cut the result at 20 chars, `会議は終わった．２０２６年に再`.
    strategy = SemanticBoundaryStrategy(
        HashEmbedder(), max_chunk_chars=20, min_chunk_chars=0, distance_threshold=0.0
    )
    chunks = strategy.chunk("会議は終わった．２０２６年に再開する．次に進んだ．")
    assert [c.text for c in chunks] == [
        "会議は終わった．",
        "２０２６年に再開する．",
        "次に進んだ．",
    ]
    assert not any(c.metadata.get("size_capped") for c in chunks)
