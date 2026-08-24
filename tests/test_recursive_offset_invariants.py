"""`RecursiveStrategy` offset/coverage invariants (#164).

#164 asked whether the `no-redef` in `strategies/recursive.py` was masking a
logic slip — three `out: list[tuple[str, int]] = []` annotations in one
function, two of them in byte-identical branches. "A rebound name inside one
function is the shape that hides a genuine logic slip", and the honest way to
answer that is to *measure*, not to read the branches and conclude.

It was masking nothing. This is the table that established it, run against the
code before the deduplication:

    text              chunk_chars   n   offsets ok   concat==text   oversize   empty
    paras                       5  22       True           True           0       0
    paras                      17   8       True           True           0       0
    ...
    unicode                    40   7       True           True           0       0
    unicode                   800   1       True           True           0       0

    0 of 32 rows violate an invariant

It ships as a test because a report that something is fine is only worth
anything if it is backed by something that keeps being true — and because the
offset round-trip is the property this strategy actually has to hold. A chunk
whose `start_offset`/`end_offset` do not address its own text is a citation
pointing at the wrong span, which is the failure mode the whole repo exists to
measure.
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.recursive import DEFAULT_SEPARATORS, RecursiveStrategy

TEXTS = {
    "paras": "Alpha beta gamma.\n\nDelta epsilon zeta eta.\n\nTheta iota kappa lambda mu nu.",
    "one-long-word": "x" * 250,
    "no-separators": "abcdefghij" * 30,
    "sentences": "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 4,
    "mixed": "A\n\nB\nC. D E " * 20,
    "trailing-separator": "aaa\n\nbbb\n\n",
    "leading-separator": "\n\naaa\n\nbbb",
    "non-ascii": "héllo wörld. " * 20,
}
SIZES = (5, 17, 40, 800)

CASES = [pytest.param(name, size, id=f"{name}-k{size}") for name in TEXTS for size in SIZES]


@pytest.mark.parametrize(("name", "size"), CASES)
def test_offsets_address_their_own_text(name: str, size: int) -> None:
    """`text[c.start_offset:c.end_offset] == c.text` for every chunk.

    The single most important property here: an offset that doesn't address its
    own chunk is a citation pointing at the wrong span.
    """
    text = TEXTS[name]
    for chunk in RecursiveStrategy(chunk_chars=size).chunk(text):
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text
        assert chunk.end_offset - chunk.start_offset == len(chunk.text)


@pytest.mark.parametrize(("name", "size"), CASES)
def test_chunks_concatenate_back_to_the_input(name: str, size: int) -> None:
    """Recursive splitting is a partition, not a filter — nothing is dropped and
    nothing is duplicated. Concatenation is the strongest single statement of
    that, and it is what the greedy merge could plausibly get wrong."""
    text = TEXTS[name]
    chunks = RecursiveStrategy(chunk_chars=size).chunk(text)
    assert "".join(c.text for c in chunks) == text


@pytest.mark.parametrize(("name", "size"), CASES)
def test_no_chunk_exceeds_the_budget(name: str, size: int) -> None:
    """`chunk_chars` is a budget, and the brute-force fallback exists precisely
    so it holds even for text with no separator in it."""
    text = TEXTS[name]
    for chunk in RecursiveStrategy(chunk_chars=size).chunk(text):
        assert len(chunk.text) <= size, f"{len(chunk.text)} > {size}: {chunk.text!r}"


@pytest.mark.parametrize(("name", "size"), CASES)
def test_no_empty_chunk_is_emitted(name: str, size: int) -> None:
    """An empty chunk carries no text but still occupies a retrieval slot and a
    row in every results table."""
    text = TEXTS[name]
    for chunk in RecursiveStrategy(chunk_chars=size).chunk(text):
        assert chunk.text != ""


@pytest.mark.parametrize(("name", "size"), CASES)
def test_chunks_are_in_ascending_offset_order(name: str, size: int) -> None:
    text = TEXTS[name]
    offsets = [c.start_offset for c in RecursiveStrategy(chunk_chars=size).chunk(text)]
    assert offsets == sorted(offsets)


# ----------------------------------------------------------------------
# The two branches that were byte-identical
# ----------------------------------------------------------------------


def test_the_two_brute_force_branches_agree() -> None:
    """`if not separators:` and `if sep == "":` were duplicated inline loops.

    They agreed, and now they share one helper — but the property is worth
    pinning rather than trusting to the refactor: reaching the fallback via an
    exhausted separator list and via an explicit empty separator must produce
    the identical split.
    """
    text = "abcdefghij" * 12
    via_empty_sep = RecursiveStrategy(chunk_chars=7, separators=("",)).chunk(text)
    via_exhaustion = RecursiveStrategy(chunk_chars=7, separators=("☃",)).chunk(text)
    assert [(c.text, c.start_offset) for c in via_empty_sep] == [
        (c.text, c.start_offset) for c in via_exhaustion
    ]
    assert len(via_empty_sep) > 1


def test_the_default_separator_hierarchy_still_ends_in_the_fallback() -> None:
    """The empty separator at the end of `DEFAULT_SEPARATORS` is what guarantees
    termination for text containing none of the earlier separators."""
    assert DEFAULT_SEPARATORS[-1] == ""
    long_unbroken = "z" * 500
    chunks = RecursiveStrategy(chunk_chars=64).chunk(long_unbroken)
    assert len(chunks) == 8
    assert all(len(c.text) <= 64 for c in chunks)


def test_a_separator_list_without_the_empty_fallback_still_terminates() -> None:
    """`__post_init__` only requires `separators` to be non-empty, so an operator
    can drop the fallback. The `not separators` branch is what catches that."""
    chunks = RecursiveStrategy(chunk_chars=10, separators=("\n\n",)).chunk("q" * 95)
    assert "".join(c.text for c in chunks) == "q" * 95
    assert all(len(c.text) <= 10 for c in chunks)
