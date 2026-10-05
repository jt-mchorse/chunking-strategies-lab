"""`n_queries` is the length of `per_query`, on both paths (#204, D-020).

`RetrievalRun.__post_init__` and `from_json` each validated `n_queries` (a
non-bool, non-negative int) and `per_query` (a sequence of `QueryResult`), and
neither asked whether the two agree::

    RetrievalRun(..., n_queries=1, per_query=<two rows>)   # accepted
    RetrievalRun.from_json(run.to_json())                   # accepted

`n_queries` is the denominator behind every rate this repo publishes and
`per_query` is the evidence for it; `results/canonical__*.json` is the provenance
for every published number (#198, D-017).

**Measured before choosing the rule.** All five committed canonical files, and
all fifteen versions of them in git history, have `n_queries == len(per_query)`,
and `to_json` has written `per_query` since the first commit. So there is no
older or documented shape that the stricter rule would refuse -- equality, not
`>=`, and on both paths.

The rest of this suite used to build runs as `n_queries=N, per_query=[]` for
brevity; those fixtures now size their rows with `tests/_query_rows.py`.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from chunking_lab import metrics as metrics_module
from chunking_lab.corpus import Document
from chunking_lab.embedder import HashEmbedder
from chunking_lab.metrics import RetrievalRun, evaluate_strategy
from chunking_lab.queries import Query
from chunking_lab.strategies import FixedSizeStrategy
from tests._query_rows import query_results, query_rows

_ROOT = Path(__file__).resolve().parents[1]
_MISMATCH = r"n_queries is \d+ but per_query has \d+ rows"


def _run(**overrides: Any) -> RetrievalRun:
    kwargs: dict[str, Any] = {
        "strategy_name": "fixed-size",
        "embedder_model": "hash-256",
        "dataset_version": "v1",
        "n_queries": 2,
        "n_chunks_total": 3,
        # 0.0: `query_results` rows retrieve nothing, and since D-021 (#218) a
        # rate must be what the rows give. Every arm here varies the count and
        # the row count, so a rate the rows produce at ANY size keeps the
        # rate rule out of the way of the one under test.
        "recall_at_k": {5: 0.0},
        "snippet_hit_at_k": {5: 0.0},
        "per_query": query_results(2),
        "wall_clock_ms": 1.0,
        "notes": [],
    }
    kwargs.update(overrides)
    return RetrievalRun(**kwargs)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = _run().to_json()
    payload.update(overrides)
    return payload


# ----------------------------------------------------------------------
# The write path
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_queries", "n_rows"),
    [(1, 2), (2, 1), (2, 0), (0, 1), (12, 11)],
    ids=["issue-repro-more-rows", "fewer-rows", "no-rows", "zero-count-with-a-row", "one-short"],
)
def test_the_constructor_refuses_a_count_the_rows_disagree_with(
    n_queries: int, n_rows: int
) -> None:
    """`fewer-rows` and `one-short` are the `>=` neighbour's accept set: a rule
    of `n_queries >= len(per_query)` would pass both, and nothing produces
    either shape."""
    with pytest.raises(ValueError, match=_MISMATCH):
        _run(n_queries=n_queries, per_query=query_results(n_rows))


@pytest.mark.parametrize("n", [0, 1, 2, 12])
def test_an_agreeing_run_constructs_and_round_trips(n: int) -> None:
    run = _run(n_queries=n, per_query=query_results(n))
    assert RetrievalRun.from_json(run.to_json()) == run


def test_the_count_rules_run_before_the_agreement_rule() -> None:
    """Order is load-bearing. `"2" != 2`, so an agreement check that ran first
    would name the wrong defect for a string count; and `True == 1`, so it
    would *pass* a bool count over one row and leave the refusal to luck."""
    with pytest.raises(ValueError, match="n_queries") as excinfo:
        _run(n_queries="2", per_query=query_results(2))
    assert not re.search(_MISMATCH, str(excinfo.value))
    with pytest.raises(ValueError, match="n_queries") as excinfo:
        _run(n_queries=True, per_query=query_results(1))
    assert not re.search(_MISMATCH, str(excinfo.value))


def test_the_container_rule_runs_before_the_agreement_rule() -> None:
    """`len("ab") == 2`: without `_validate_per_query` in front, a string
    `per_query` would satisfy `n_queries=2`."""
    with pytest.raises(ValueError, match="per_query") as excinfo:
        _run(n_queries=2, per_query="ab")
    assert not re.search(_MISMATCH, str(excinfo.value))


# ----------------------------------------------------------------------
# The read path
# ----------------------------------------------------------------------


def test_from_json_refuses_the_issues_payload() -> None:
    with pytest.raises(ValueError, match=_MISMATCH):
        RetrievalRun.from_json(_payload(n_queries=1))


def test_an_absent_per_query_loads_only_as_a_zero_query_run() -> None:
    """`from_json` defaults an absent `per_query` to `()`. No writer in this
    repo's history ever omitted it, so the default now means exactly what it
    says: no rows, therefore no queries."""
    payload = _payload()
    del payload["per_query"]
    with pytest.raises(ValueError, match="n_queries is 2 but per_query has 0 rows"):
        RetrievalRun.from_json(payload)
    payload["n_queries"] = 0
    assert RetrievalRun.from_json(payload).per_query == ()


@pytest.mark.parametrize(
    "path", sorted((_ROOT / "results").glob("canonical__*.json")), ids=lambda p: p.name
)
def test_a_hand_edited_canonical_file_is_refused(path: Path) -> None:
    """The provenance files themselves: drop one row and the file no longer
    loads. The unmodified file is the control and must still load."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert RetrievalRun.from_json(raw).n_queries == len(raw["per_query"])
    raw["per_query"] = raw["per_query"][:-1]
    with pytest.raises(ValueError, match=_MISMATCH):
        RetrievalRun.from_json(raw)


def test_the_corpus_of_canonical_files_is_not_empty() -> None:
    """The parametrised arm above over an empty glob is five green nothings."""
    assert len(sorted((_ROOT / "results").glob("canonical__*.json"))) == 5


# ----------------------------------------------------------------------
# One definition, reached by both paths
# ----------------------------------------------------------------------


def test_from_json_reaches_the_shared_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the orchestrator, not the helper: a spy on the module-level
    function sees the read path call it. Removing the call from `__post_init__`
    turns this red even though every direct-call arm would still pass."""
    payload = _payload()  # built before the spy, so its own construction is not counted
    seen: list[tuple[int, int]] = []
    real = metrics_module._validate_query_count

    def spy(n_queries: int, per_query: Any) -> None:
        seen.append((n_queries, len(per_query)))
        real(n_queries, per_query)

    monkeypatch.setattr(metrics_module, "_validate_query_count", spy)
    RetrievalRun.from_json(payload)
    assert seen == [(2, 2)]


def test_there_is_exactly_one_call_site() -> None:
    """A second copy on the read path is what produced #180, #181 and #182, and
    `from_json` already builds through `cls(...)`. Counted by AST, so a comment
    or docstring naming the function does not count."""
    tree = ast.parse((_ROOT / "chunking_lab" / "metrics.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_query_count"
    ]
    assert len(calls) == 1


# ----------------------------------------------------------------------
# The in-package producer still satisfies it
# ----------------------------------------------------------------------


def test_evaluate_strategy_builds_an_agreeing_run() -> None:
    corpus = [
        Document(filename="a.md", text="Apples grow on apple trees in orchards."),
        Document(filename="b.md", text="Bears hibernate through the winter months."),
    ]
    queries = [
        Query(
            id="q1",
            question="Where do apples grow?",
            expected_doc="a.md",
            expected_snippet="apple trees",
        ),
        Query(
            id="q2", question="What do bears do?", expected_doc="b.md", expected_snippet="hibernate"
        ),
        Query(id="q3", question="Winter?", expected_doc="b.md", expected_snippet="winter"),
    ]
    run = evaluate_strategy(
        strategy=FixedSizeStrategy(),
        corpus=corpus,
        queries=queries,
        embedder=HashEmbedder(),
    )
    assert run.n_queries == len(run.per_query) == 3


def test_the_fixture_helper_builds_rows_the_reader_accepts() -> None:
    """The rows `tests/_query_rows.py` supplies to the rest of the suite are
    real `QueryResult`s, not a shape that only happens to have a length."""
    assert len(query_rows(3)) == 3
    assert [q.query_id for q in query_results(3)] == ["q0", "q1", "q2"]
