"""A one-shot iterator scores what a list scores (#279).

Since #262 `evaluate_strategy` validates `corpus` before chunking it, and both
loop over it. Measured on main with the pinned corpus/queries and
`FixedSizeStrategy(chunk_chars=600)`:

    list(docs)          -> recall {1: 0.167, 3: 0.417, 5: 0.583}
    (d for d in docs)   -> recall {1: 0.0, 3: 0.0, 5: 0.0}, 0 chunks, no error

A generator `queries` and `ks` are spent the same way by their validators.
"""

from __future__ import annotations

from typing import Any

import pytest

from chunking_lab import load_corpus, load_queries
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import evaluate_strategy
from chunking_lab.strategies import FixedSizeStrategy


def _fingerprint(run: Any) -> dict[str, Any]:
    j = run.to_json()
    return {k: v for k, v in j.items() if "wall" not in k and "time" not in k}


@pytest.fixture(scope="module")
def data() -> tuple[list, list]:
    return list(load_corpus()), list(load_queries())


def _run(corpus: Any, queries: Any, ks: Any = (1, 3, 5)) -> Any:
    return evaluate_strategy(
        FixedSizeStrategy(chunk_chars=600), corpus, queries, HashEmbedder(), ks=ks
    )


def test_a_generator_corpus_scores_what_the_list_scores(data: tuple[list, list]) -> None:
    corpus, queries = data
    expected = _fingerprint(_run(corpus, queries))
    assert expected["recall_at_k"]["5"] > 0  # anti-vacuity: the list run finds hits
    assert _fingerprint(_run((d for d in corpus), queries)) == expected


def test_generator_queries_score_what_the_list_scores(data: tuple[list, list]) -> None:
    corpus, queries = data
    expected = _fingerprint(_run(corpus, queries))
    assert _fingerprint(_run(corpus, (q for q in queries))) == expected


def test_a_generator_ks_scores_what_the_tuple_scores(data: tuple[list, list]) -> None:
    corpus, queries = data
    expected = _fingerprint(_run(corpus, queries, ks=(1, 3, 5)))
    assert _fingerprint(_run(corpus, queries, ks=(k for k in (1, 3, 5)))) == expected


def test_a_bare_string_is_still_refused(data: tuple[list, list]) -> None:
    corpus, queries = data
    with pytest.raises((TypeError, ValueError)):
        _run(corpus, queries, ks="135")
