"""Non-finite-component guards on both cosine helpers (#66).

Both `strategies.semantic._cosine_distance` and `metrics._cosine` guard the
zero-norm degenerate input but previously missed a non-finite (NaN/±Inf)
embedding component, which silently corrupted results: a suppressed topic
boundary (semantic) or a NaN-poisoned `scored.sort()` ranking (metrics).
`Embedder` is a BYO Protocol, so a poisoned vector is reachable. Sibling of
llm-cost-optimizer #88 and rag-production-kit #82.
"""

from __future__ import annotations

import math

import pytest

from chunking_lab.metrics import _cosine
from chunking_lab.strategies.semantic import _cosine_distance


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_cosine_distance_rejects_non_finite_component(bad: float):
    with pytest.raises(ValueError, match="non-finite cosine distance"):
        _cosine_distance([0.1, bad, 0.3], [0.2, 0.4, 0.6])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_cosine_distance_rejects_non_finite_in_second_vector(bad: float):
    with pytest.raises(ValueError, match="non-finite cosine distance"):
        _cosine_distance([0.2, 0.4, 0.6], [0.1, bad, 0.3])


def test_cosine_distance_finite_inputs_unchanged():
    # Identical unit-ish vectors → distance ~0; orthogonal-ish → ~1.
    assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_distance_zero_norm_still_returns_fallback():
    # Regression guard: the documented zero-norm fallback (1.0) is preserved.
    assert _cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_cosine_similarity_rejects_non_finite_component(bad: float):
    with pytest.raises(ValueError, match="non-finite cosine similarity"):
        _cosine([0.1, bad, 0.3], [0.2, 0.4, 0.6])


def test_cosine_similarity_finite_inputs_unchanged():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_norm_still_returns_fallback():
    # Regression guard: the documented zero-norm fallback (0.0) is preserved.
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
