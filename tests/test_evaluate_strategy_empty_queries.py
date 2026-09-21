"""`evaluate_strategy` guards both of its populations, not one (#192).

`validate_ks` has guarded `ks` since #28, for a stated reason: "an empty `ks`
silently produces an empty `recall_at_k` dict". That reasoning covers `queries`
too, and the consequence there is worse — an empty `queries` produces a
*populated* map, full of the floor of the metric's range::

    evaluate_strategy(FixedSizeStrategy(), corpus, [], HashEmbedder(), ks=(1, 3, 5))

    n_queries      = 0
    n_chunks_total = 3
    recall_at_k    = {1: 0.0, 3: 0.0, 5: 0.0}     <- measured on the unguarded code
    snippet_hit@k  = {1: 0.0, 3: 0.0, 5: 0.0}

Exit 0, and `RetrievalRun.from_json` accepts it. `0.0` is not a neutral
placeholder on these maps: `_render_summary`'s own comments argue twice (#76,
#160) that a zero there is read as "the strategy failed" — "the table said it
scored zero on everything, and a reader concludes the strategy failed".

Two things these arms are careful about:

1. **Where the guard sits, not just that it raises.** An arm asserting only the
   exception type passes for a guard placed *after* the per-query loop, where
   `n == 0` has already been baked into the maps. `test_the_guard_fires_before_any_
   measurement_work` watches whether the strategy was ever asked to chunk.
2. **`corpus` must keep working.** An empty corpus with real queries yields a
   *truthful* `0.0` — the queries ran and retrieved nothing. That is a
   measurement, not a fabrication, and it is what rejects the over-broad
   neighbour that sweeps both arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import Document  # noqa: E402
from chunking_lab.embedder import HashEmbedder  # noqa: E402
from chunking_lab.metrics import (  # noqa: E402
    RetrievalRun,
    evaluate_strategy,
    validate_queries,
)
from chunking_lab.queries import Query, load_queries  # noqa: E402
from chunking_lab.strategies import Chunk, FixedSizeStrategy  # noqa: E402

_CORPUS = [
    Document(filename="alpha.md", text="alpha beta gamma delta epsilon " * 40),
    Document(filename="zeta.md", text="zeta eta theta iota kappa " * 40),
]
_QUERIES = [
    Query(
        id="q1",
        question="alpha beta",
        expected_doc="alpha.md",
        expected_snippet="alpha beta",
    )
]


# ----------------------------------------------------------------------
# The guard
# ----------------------------------------------------------------------


@pytest.mark.parametrize("empty", [[], (), list(), tuple()])
def test_evaluate_strategy_refuses_an_empty_query_set(empty: object) -> None:
    with pytest.raises(ValueError, match="queries must be non-empty"):
        evaluate_strategy(FixedSizeStrategy(), _CORPUS, empty, HashEmbedder(), ks=(1, 3, 5))  # type: ignore[arg-type]


def test_the_message_names_the_value_that_used_to_ship() -> None:
    """The defect is the published number, not the absence of an exception.

    Pinned as a message assertion because the fabricated maps cannot be
    asserted once the guard exists — the next reader needs the record of what
    was being refused.
    """
    with pytest.raises(ValueError, match="queries must be non-empty") as excinfo:
        evaluate_strategy(FixedSizeStrategy(), _CORPUS, [], HashEmbedder(), ks=(1, 3, 5))
    message = str(excinfo.value)
    assert "recall@k = 0.0" in message
    assert "snippet-hit@k = 0.0" in message
    # And it says which of the two readings is wrong, since both are "zero".
    assert "retrieved nothing" in message
    assert "nothing was measured" in message


def test_the_guard_fires_before_any_measurement_work() -> None:
    """Placement, not just presence.

    A guard added *after* the per-query loop would raise the same exception
    type with `n == 0` already baked into the maps, and would have done the
    chunking and embedding work first. This arm is what separates the two: the
    strategy must never be asked to chunk.
    """
    calls: list[str] = []

    class _SpyStrategy(FixedSizeStrategy):
        def chunk(self, text: str, *, source_doc_id: str = "") -> list[Chunk]:
            calls.append(source_doc_id)
            return super().chunk(text, source_doc_id=source_doc_id)

    with pytest.raises(ValueError, match="queries must be non-empty"):
        evaluate_strategy(_SpyStrategy(), _CORPUS, [], HashEmbedder(), ks=(1, 3, 5))
    assert calls == []

    # ...and the spy really does record work on the path that is allowed to run,
    # so the empty list above means "nothing happened", not "the spy is broken".
    evaluate_strategy(_SpyStrategy(), _CORPUS, _QUERIES, HashEmbedder(), ks=(1,))
    assert calls == ["alpha.md", "zeta.md"]


def test_validate_queries_accepts_a_non_empty_set() -> None:
    validate_queries(_QUERIES)
    validate_queries(tuple(_QUERIES))


# ----------------------------------------------------------------------
# What must keep working
# ----------------------------------------------------------------------


def test_an_empty_corpus_is_not_guarded_because_its_zero_is_truthful() -> None:
    """The arm that rejects the over-broad neighbour.

    An empty corpus with real queries is a legitimate measurement: the queries
    ran, nothing was retrieved, and `recall@k = 0.0` is the honest answer. The
    distinction is which population the denominator counts — `n` is the query
    count, and it is only that one which can be zero while the map is populated.
    """
    run = evaluate_strategy(FixedSizeStrategy(), [], _QUERIES, HashEmbedder(), ks=(1, 3))

    assert run.n_queries == 1
    assert run.n_chunks_total == 0
    assert run.recall_at_k == {1: 0.0, 3: 0.0}
    assert run.snippet_hit_at_k == {1: 0.0, 3: 0.0}
    assert len(run.per_query) == 1
    # The per-query record is what makes this zero readable as a measurement:
    # one query ran and retrieved nothing.
    assert run.per_query[0].retrieved_doc_ids_in_rank_order == ()


def test_an_ordinary_run_is_unchanged() -> None:
    run = evaluate_strategy(FixedSizeStrategy(), _CORPUS, _QUERIES, HashEmbedder(), ks=(1, 3))
    assert run.n_queries == 1
    assert set(run.recall_at_k) == {1, 3}
    assert all(0.0 <= v <= 1.0 for v in run.recall_at_k.values())


def test_the_cli_still_fails_through_the_loaders_own_message(tmp_path: Path) -> None:
    """`run_matrix` reaches `evaluate_strategy` only via `load_queries`.

    That loader already refuses this input, so the operator must keep getting
    one error naming the file — not a second one from the metric boundary
    restating it.
    """
    empty = tmp_path / "queries.jsonl"
    empty.write_text("\n  \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="queries file is empty"):
        load_queries(empty)


def test_the_committed_canonical_results_still_round_trip() -> None:
    """The guard must not reject the artifacts actually in the repo."""
    canonical = sorted((_REPO_ROOT / "results").glob("canonical__*.json"))
    assert canonical, "expected committed canonical__*.json files under results/"
    for path in canonical:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = RetrievalRun.from_json(payload)
        assert run.n_queries > 0, f"{path.name} has no queries"
        assert run.to_json() == payload, path.name
