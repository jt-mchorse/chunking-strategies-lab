"""Cosine distance stays in [0, 2], so threshold 0.0 splits identical sentences (#277).

`1.0 - dot / (na * nb)` rounded to -2.220446049250313e-16 for some identical
vectors and to 0.0 for others. Measured on main with `HashEmbedder`,
`chunk(f"{s} {s}")` at `distance_threshold=0.0, min_chunk_chars=0`: 7 of 12
"<word> repeats here." sentences gave 1 chunk, 5 gave 2.
"""

from __future__ import annotations

import math

import pytest

from chunking_lab.embedder import HashEmbedder
from chunking_lab.strategies.semantic import SemanticBoundaryStrategy, _cosine_distance

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
]  # fmt: skip


def _unclamped(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1.0 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_the_fixture_has_identical_pairs_that_round_below_zero() -> None:
    # Anti-vacuity: without these the arms below would pass on main too.
    e = HashEmbedder()
    below = [w for w in WORDS if _unclamped(*[e.embed(f"{w} repeats here.")] * 2) < 0.0]
    assert len(below) >= 3


@pytest.mark.parametrize("word", WORDS)
def test_identical_sentences_are_distance_exactly_zero(word: str) -> None:
    v = HashEmbedder().embed(f"{word} repeats here.")
    assert _cosine_distance(v, v) == 0.0


@pytest.mark.parametrize("word", WORDS)
def test_a_repeated_sentence_splits_at_threshold_zero(word: str) -> None:
    s = f"{word} repeats here."
    strategy = SemanticBoundaryStrategy(HashEmbedder(), distance_threshold=0.0, min_chunk_chars=0)
    assert len(strategy.chunk(f"{s} {s}", source_doc_id="d")) == 2


def test_the_upper_bound_and_the_nan_guard_hold() -> None:
    assert _cosine_distance([1.0, 0.0], [-1.0, 0.0]) == 2.0
    assert _cosine_distance([1.0, 1e-17], [-1.0, 0.0]) <= 2.0
    with pytest.raises(ValueError, match="non-finite cosine distance"):
        _cosine_distance([float("nan"), 0.0], [1.0, 0.0])


def test_ordinary_distances_are_unchanged() -> None:
    e = HashEmbedder()
    pairs = [
        (e.embed(f"{a} topic."), e.embed(f"{b} subject."))
        for a, b in zip(WORDS, WORDS[1:], strict=False)
    ]
    for a, b in pairs:
        raw = _unclamped(a, b)
        assert 0.0 < raw < 2.0
        assert _cosine_distance(a, b) == raw
