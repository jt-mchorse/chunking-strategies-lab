"""`evaluate_strategy` refuses a query list with a repeated id (#250).

`load_queries` refuses a duplicate id and `python -m chunking_lab.validate`
reports it as `duplicate_id`, but the library entry point never checked, so a
caller building `Query` objects directly could publish rates over an inflated
count. #192 moved the *emptiness* rule to the metric boundary for exactly this
reason ("checked only inside `load_queries`, three modules away"); uniqueness
is the other population rule that loader enforces.

Measured on the unguarded code, pinned substrate, `FixedSizeStrategy(600, 80)`,
`HashEmbedder`: the twelve queries plus the one recall@5 miss repeated three
times published `n_queries=15` and recall@5 `0.733` (twelve distinct: `0.917`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import Document, load_corpus  # noqa: E402
from chunking_lab.embedder import HashEmbedder  # noqa: E402
from chunking_lab.metrics import evaluate_strategy, validate_queries  # noqa: E402
from chunking_lab.queries import Query, load_queries  # noqa: E402
from chunking_lab.strategies import Chunk, FixedSizeStrategy  # noqa: E402

_CORPUS = [
    Document(filename="alpha.md", text="alpha beta gamma delta epsilon " * 40),
    Document(filename="zeta.md", text="zeta eta theta iota kappa " * 40),
]


def _q(qid: str, doc: str = "alpha.md", snippet: str = "alpha beta") -> Query:
    return Query(id=qid, question=f"question {qid}", expected_doc=doc, expected_snippet=snippet)


def test_a_repeated_id_is_refused() -> None:
    with pytest.raises(ValueError, match="query ids must be unique"):
        evaluate_strategy(
            FixedSizeStrategy(), _CORPUS, [_q("q1"), _q("q2"), _q("q1")], HashEmbedder()
        )


def test_the_message_names_every_repeated_id_once() -> None:
    queries = [_q("q1"), _q("q2"), _q("q1"), _q("q2"), _q("q1"), _q("q3")]
    with pytest.raises(ValueError, match="query ids must be unique") as exc:
        validate_queries(queries)
    assert "['q1', 'q2']" in str(exc.value)


def test_same_id_different_content_is_refused_too() -> None:
    """The rule is on the key, not on the record: two different queries sharing
    an id make the per_query rows indistinguishable by query_id."""
    with pytest.raises(ValueError, match="repeated: \\['q1'\\]"):
        validate_queries([_q("q1"), _q("q1", doc="zeta.md", snippet="zeta eta")])


def test_the_guard_fires_before_any_chunking() -> None:
    calls: list[str] = []

    class _Watching(FixedSizeStrategy):
        def chunk(self, text: str, *, source_doc_id: str = "doc") -> list[Chunk]:
            calls.append(source_doc_id)
            return super().chunk(text, source_doc_id=source_doc_id)

    with pytest.raises(ValueError, match="query ids must be unique"):
        evaluate_strategy(_Watching(), _CORPUS, [_q("q1"), _q("q1")], HashEmbedder())
    assert calls == []


def test_distinct_ids_with_identical_questions_still_evaluate() -> None:
    """Rejects the over-broad neighbour that dedupes on content: two queries may
    legitimately ask the same thing of different documents."""
    queries = [
        Query(id="a", question="same", expected_doc="alpha.md", expected_snippet="alpha"),
        Query(id="b", question="same", expected_doc="zeta.md", expected_snippet="zeta"),
    ]
    run = evaluate_strategy(FixedSizeStrategy(), _CORPUS, queries, HashEmbedder(), ks=(1, 2))
    assert run.n_queries == 2
    assert [r.query_id for r in run.per_query] == ["a", "b"]


def test_the_issue_repro_on_the_pinned_substrate_is_refused() -> None:
    corpus, queries, embedder = load_corpus(), load_queries(), HashEmbedder()
    strategy = FixedSizeStrategy(chunk_chars=600, overlap_chars=80)
    clean = evaluate_strategy(strategy, corpus, queries, embedder)
    assert clean.n_queries == len({q.id for q in queries}) == 12
    hit_ids = {
        r.query_id
        for r in clean.per_query
        if r.expected_doc in r.retrieved_doc_ids_in_rank_order[:5]
    }
    misses = [q for q in queries if q.id not in hit_ids]
    assert misses, "the repro needs at least one recall@5 miss to repeat"
    with pytest.raises(ValueError, match="query ids must be unique"):
        evaluate_strategy(strategy, corpus, queries + misses * 3, embedder)
