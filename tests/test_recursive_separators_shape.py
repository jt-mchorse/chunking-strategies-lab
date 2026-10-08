"""`RecursiveStrategy.separators` is checked for shape and element type (#236).

`separators=". "` was accepted and split into the characters "." and " ": the
chunks came out at the wrong boundaries while still satisfying every chunk
invariant, so the mistake was invisible downstream. Measured before the fix:

    separators=". "        -> ['Alpha beta.', ' Gamma delta.', ' Epsilon zeta.']
    separators=(". ", "")  -> ['Alpha beta. ', 'Gamma delta. ', 'Epsilon zeta.']
"""

from __future__ import annotations

import pytest

from chunking_lab.strategies.recursive import DEFAULT_SEPARATORS, RecursiveStrategy

TEXT = "Alpha beta. Gamma delta. Epsilon zeta."


@pytest.mark.parametrize(
    "bad",
    [". ", b". ", bytearray(b". "), {". ", ""}, None, 5],
    ids=["str", "bytes", "bytearray", "set", "None", "int"],
)
def test_a_non_sequence_or_a_bare_string_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="separators must be a tuple or list of str"):
        RecursiveStrategy(chunk_chars=14, separators=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [(". ", None), ("x", b"y"), (". ", 3), ([". "],)],
    ids=["None", "bytes", "int", "nested-list"],
)
def test_every_element_must_be_a_str(bad: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match=r"separators\[\d\] must be a str"):
        RecursiveStrategy(chunk_chars=14, separators=bad)  # type: ignore[arg-type]


def test_empty_is_still_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RecursiveStrategy(chunk_chars=14, separators=())


def test_a_list_is_accepted_and_stored_as_a_tuple() -> None:
    seps = [". ", ""]
    strategy = RecursiveStrategy(chunk_chars=14, separators=seps)  # type: ignore[arg-type]
    assert strategy.separators == (". ", "")
    assert isinstance(strategy.separators, tuple)
    seps.insert(0, " ")  # the caller's list no longer reaches the strategy
    assert strategy.separators == (". ", "")
    assert [c.text for c in strategy.chunk(TEXT)] == [
        "Alpha beta. ",
        "Gamma delta. ",
        "Epsilon zeta.",
    ]


def test_the_empty_separator_stays_legal_and_the_default_is_unchanged() -> None:
    assert RecursiveStrategy(chunk_chars=5, separators=("",)).separators == ("",)
    assert RecursiveStrategy().separators == DEFAULT_SEPARATORS
