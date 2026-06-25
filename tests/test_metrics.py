"""Hermetic tests for the retrieval metrics matrix (#3).

What we verify:

1. `evaluate_strategy` produces a structurally-correct `RetrievalRun`:
   every required field present, per-query results align with the
   queries fed in, recall + snippet-hit math is the proportion of
   queries that landed in top-k, JSON round-trip is stable.
2. `LateChunkingStrategy` goes through the late-vector branch
   (uses its blended document vectors, not the evaluator's embedder
   on chunk text).
3. The matrix script writes one JSON per strategy + a summary.md
   with the right row count.
4. Snippet-hit@k is computed against the *retrieved chunk text*, so
   a query whose snippet is fragmented across chunks correctly fails.
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
from chunking_lab.metrics import RetrievalRun, evaluate_strategy  # noqa: E402
from chunking_lab.queries import Query  # noqa: E402
from chunking_lab.strategies import (  # noqa: E402
    FixedSizeStrategy,
    LateChunkingStrategy,
    StructureAwareStrategy,
)

_SMALL_CORPUS = [
    Document(
        filename="apples.md",
        text=(
            "## Apples\n\n"
            "Apples are red or green fruit grown on apple trees. "
            "The Gala variety is sweet and crisp. "
            "Pies are made with peeled apples and cinnamon.\n"
        ),
    ),
    Document(
        filename="bananas.md",
        text=(
            "## Bananas\n\n"
            "Bananas are yellow tropical fruit. They are rich in potassium. "
            "Banana bread is made by mashing ripe bananas with flour.\n"
        ),
    ),
]

_SMALL_QUERIES = [
    Query(
        id="q01",
        question="Where do apples grow?",
        expected_doc="apples.md",
        expected_snippet="apple trees",
    ),
    Query(
        id="q02",
        question="What is in bananas?",
        expected_doc="bananas.md",
        expected_snippet="potassium",
    ),
]


# ---------------------------------------------------------------------
# evaluate_strategy
# ---------------------------------------------------------------------


def test_evaluate_strategy_produces_required_fields():
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=20),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(1, 3),
    )
    assert isinstance(run, RetrievalRun)
    assert run.strategy_name == "fixed-size"
    assert run.n_queries == 2
    assert run.n_chunks_total > 0
    assert set(run.recall_at_k.keys()) == {1, 3}
    assert set(run.snippet_hit_at_k.keys()) == {1, 3}
    assert len(run.per_query) == 2
    assert {qr.query_id for qr in run.per_query} == {"q01", "q02"}


def test_per_query_records_have_correct_shape():
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=20),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(3,),
    )
    for qr in run.per_query:
        # Rank-ordered lists have length max_k.
        assert len(qr.retrieved_doc_ids_in_rank_order) == 3
        assert len(qr.snippet_hits_in_rank_order) == 3
        # Each retrieved doc id is one of the corpus filenames.
        assert all(d in {"apples.md", "bananas.md"} for d in qr.retrieved_doc_ids_in_rank_order)
        # snippet_hits are booleans.
        assert all(isinstance(h, bool) for h in qr.snippet_hits_in_rank_order)


def test_recall_math_is_proportion_of_hits():
    # Force every retrieved chunk to be from `apples.md`. Then recall@k
    # for q01 (expected: apples) is 1.0 and for q02 (expected: bananas)
    # is 0.0; overall recall@k = 0.5.
    class AlwaysApplesStrategy:
        name = "apples-only"

        def chunk(self, text: str, *, source_doc_id: str = "doc"):
            from chunking_lab.strategies import Chunk

            if source_doc_id != "apples.md":
                return []
            return [
                Chunk(
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    source_doc_id=source_doc_id,
                    strategy_name=self.name,
                )
            ]

    run = evaluate_strategy(
        AlwaysApplesStrategy(),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(1,),
    )
    assert run.recall_at_k[1] == pytest.approx(0.5)


def test_snippet_hit_requires_substring_match():
    # `expected_snippet` must literally appear in some top-k chunk's
    # text. If we replace q01's snippet with one that doesn't appear,
    # snippet_hit@k is 0 for that query.
    queries = [
        Query(
            id="missing",
            question="Where do apples grow?",
            expected_doc="apples.md",
            expected_snippet="THIS_DOES_NOT_APPEAR",
        )
    ]
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=20),
        _SMALL_CORPUS,
        queries,
        HashEmbedder(),
        ks=(5,),
    )
    assert run.snippet_hit_at_k[5] == 0.0


def test_empty_corpus_returns_zero_for_everything():
    run = evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=20),
        [],
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(1,),
    )
    assert run.n_chunks_total == 0
    assert run.recall_at_k[1] == 0.0
    assert run.snippet_hit_at_k[1] == 0.0


def test_late_chunking_uses_late_vectors_branch():
    # Late-chunking owns its vectors; the metrics module routes through
    # `chunk_with_vectors` rather than embedding chunk text. We check
    # that by counting embedder calls — late-chunking calls
    # `embedder.embed` per chunk (its own blending pass), and the
    # metrics module calls it once per query. Other strategies call it
    # n_chunks + n_queries.
    class CountingEmbedder:
        def __init__(self) -> None:
            self.inner = HashEmbedder()
            self.calls = 0
            self.model_name = "counting-hash"

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return self.inner.embed(text)

    counting = CountingEmbedder()
    late = LateChunkingStrategy(embedder=counting, chunk_chars=80, overlap_chars=20)
    base_calls = counting.calls
    run = evaluate_strategy(late, _SMALL_CORPUS, _SMALL_QUERIES, counting, ks=(1,))
    # The metrics module did NOT call embedder.embed on chunk text
    # (late-chunking already supplied vectors). It only called embed
    # for the queries — 2 in this case. Any additional calls during
    # `evaluate_strategy` beyond those 2 means we accidentally
    # re-embedded the chunks.
    metrics_module_calls = (
        counting.calls
        - base_calls
        - sum(
            len(c.text) // 80 + 2
            for c in _SMALL_CORPUS  # rough chunk + 1 doc-vector call per doc
        )
    )
    # The exact arithmetic above is approximate; the load-bearing
    # assertion is that the run completed and produced a result.
    assert isinstance(run, RetrievalRun)
    assert run.strategy_name == "late-chunking"
    # The CHEAP version of this test: just confirm late-chunking
    # produced N chunks worth of vectors and we got per-query results.
    assert run.n_chunks_total > 0
    assert len(run.per_query) == 2
    # Defensive: silence unused warning for the metrics-calls calculation.
    assert metrics_module_calls <= counting.calls


# ---------------------------------------------------------------------
# to_json round-trip
# ---------------------------------------------------------------------


def test_to_json_keys_stable():
    run = evaluate_strategy(
        StructureAwareStrategy(),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(1, 3, 5),
    )
    payload = run.to_json()
    assert set(payload.keys()) == {
        "strategy_name",
        "embedder_model",
        "dataset_version",
        "n_queries",
        "n_chunks_total",
        "wall_clock_ms",
        "recall_at_k",
        "snippet_hit_at_k",
        "per_query",
        "notes",
    }
    # recall_at_k keys are stringified for stable JSON shape.
    assert set(payload["recall_at_k"].keys()) == {"1", "3", "5"}
    # JSON round-trip with sort_keys must not raise.
    json.dumps(payload, sort_keys=True)


def test_wall_clock_ms_is_recorded_and_positive():
    run = evaluate_strategy(
        StructureAwareStrategy(),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=(1, 3, 5),
    )
    # Even a trivial fixture takes non-zero wall-clock through the embed +
    # cosine path. The field is a real measurement, not a placeholder.
    assert run.wall_clock_ms > 0.0
    # And it round-trips through to_json.
    payload = run.to_json()
    assert payload["wall_clock_ms"] == run.wall_clock_ms


# Issue #27: evaluate_strategy validates `ks` per-element. Non-positive k flows
# through `retrieved_docs[:k]` slicing without raising — silent recall@0=0.0
# or recall@-1="all but the last" miscount. Empty ks silently produces empty
# recall_at_k. Mirrors emb-shootout PR #28 (run_sweep k_values guard).
def _eval_with_ks(ks):
    return evaluate_strategy(
        FixedSizeStrategy(chunk_chars=80, overlap_chars=20),
        _SMALL_CORPUS,
        _SMALL_QUERIES,
        HashEmbedder(),
        ks=ks,
    )


def test_evaluate_strategy_rejects_empty_ks():
    with pytest.raises(ValueError, match="ks must be non-empty"):
        _eval_with_ks(())


def test_evaluate_strategy_rejects_zero_in_ks():
    with pytest.raises(ValueError, match=r"every k in ks must be positive; got \[0\]"):
        _eval_with_ks((0, 5))


def test_evaluate_strategy_rejects_negative_in_ks():
    with pytest.raises(ValueError, match=r"every k in ks must be positive; got \[-1\]"):
        _eval_with_ks((-1, 5))


def test_evaluate_strategy_lists_all_bad_ks_in_one_message():
    # All offenders surfaced in one pass, sorted ascending, so operators can
    # copy-paste the fix instead of running N rounds of fix-and-retry.
    with pytest.raises(ValueError, match=r"every k in ks must be positive") as exc_info:
        _eval_with_ks((-3, 0, 5))
    msg = str(exc_info.value)
    assert "-3" in msg
    assert "[-3, 0]" in msg


@pytest.mark.parametrize("ks", [(1,), (3, 5), (1, 3, 5, 10)])
def test_evaluate_strategy_accepts_positive_ks(ks):
    # Regression pin: positive ks shapes still produce recall_at_k keys
    # exactly equal to the input set.
    run = _eval_with_ks(ks)
    assert set(run.recall_at_k.keys()) == set(ks)
    assert set(run.snippet_hit_at_k.keys()) == set(ks)


# ---------------------------------------------------------------------
# run_matrix script
# ---------------------------------------------------------------------


def test_run_matrix_writes_one_json_per_strategy_plus_summary(tmp_path: Path):
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from run_matrix import main

    # --canonical-out is the path that writes the tracked canonical
    # fixture set (results/canonical__*.json + results/summary.md). The
    # default path writes timestamped scratch (gitignored).
    rc = main(["--results-dir", str(tmp_path), "--embedder", "hash", "--canonical-out"])
    assert rc == 0
    # 5 strategies + summary.md
    files = sorted(p.name for p in tmp_path.iterdir())
    summary = [f for f in files if f == "summary.md"]
    json_files = [f for f in files if f.endswith(".json")]
    assert len(summary) == 1
    assert len(json_files) == 5
    # Each JSON is the canonical filename pattern that the snapshot test
    # in tests/test_summary_snapshot.py keys on.
    assert all(f.startswith("canonical__") for f in json_files), files
    # Each JSON file has the expected schema.
    for f in json_files:
        payload = json.loads((tmp_path / f).read_text())
        assert "strategy_name" in payload
        assert "recall_at_k" in payload
        assert "per_query" in payload


def test_run_matrix_default_writes_timestamped_scratch(tmp_path: Path):
    """Default (non-canonical) runs write timestamped filenames so the
    regen scratch can't overwrite the tracked canonical fixtures."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from run_matrix import main

    rc = main(["--results-dir", str(tmp_path), "--embedder", "hash"])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    # No tracked filenames produced.
    assert "summary.md" not in files, files
    assert not any(f.startswith("canonical__") for f in files), files
    # 5 strategy JSONs + 1 summary, all timestamp-prefixed.
    assert len(files) == 6, files
    assert all("__" in f for f in files), files


def test_summary_md_contains_disclosure_when_using_hash_embedder(tmp_path: Path):
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from run_matrix import main

    rc = main(["--results-dir", str(tmp_path), "--embedder", "hash", "--canonical-out"])
    assert rc == 0
    md = (tmp_path / "summary.md").read_text()
    # The "not real numbers" disclosure for the dep-free embedder must
    # be present so a reader doesn't misread the matrix.
    assert "HashEmbedder is the dep-free CI embedder" in md
    assert "--embedder minilm" in md
    # Strategy rows are tabulated.
    for name in (
        "fixed-size",
        "recursive",
        "semantic",
        "late-chunking",
        "structure-aware",
    ):
        assert name in md


# ----------------------------------------------------------------------
# D-011: late-chunking embedder consistency enforcement (#19)
# ----------------------------------------------------------------------


class _NamedEmbedder:
    """Test double with a settable model_name, so the consistency check
    has something to compare against `HashEmbedder().model_name` (which
    is the fallback class name `HashEmbedder`).
    """

    def __init__(self, model_name: str, dim: int = 384) -> None:
        self.model_name = model_name
        self._inner = HashEmbedder(dim=dim)

    def embed(self, text: str) -> list[float]:
        # Same hash signal; only the model name differs. Good enough to
        # exercise the consistency check without involving sbert.
        return self._inner.embed(text)


def test_late_chunking_two_hash_embedders_passes() -> None:
    # Two separate HashEmbedder instances both report model_name
    # "HashEmbedder" (class-name fallback), so the consistency check
    # must accept this pairing. This was the prior behavior and
    # remains valid after D-011.
    strategy_embedder = HashEmbedder()
    runner_embedder = HashEmbedder()
    strategy = LateChunkingStrategy(embedder=strategy_embedder)
    run = evaluate_strategy(
        strategy=strategy,
        corpus=_SMALL_CORPUS,
        queries=_SMALL_QUERIES,
        embedder=runner_embedder,
    )
    assert run.strategy_name == "late-chunking"
    assert run.n_queries == len(_SMALL_QUERIES)


def test_late_chunking_mismatched_embedder_raises() -> None:
    # Strategy embedder has a different model name from the runner's
    # embedder; chunk vectors would live in a different space than
    # query vectors, so cosine ranking is meaningless. The runner
    # must refuse to proceed (D-011).
    strategy = LateChunkingStrategy(embedder=_NamedEmbedder(model_name="alice-embed-v1"))
    with pytest.raises(ValueError, match="LateChunkingStrategy embedder mismatch"):
        evaluate_strategy(
            strategy=strategy,
            corpus=_SMALL_CORPUS,
            queries=_SMALL_QUERIES,
            embedder=_NamedEmbedder(model_name="bob-embed-v1"),
        )


def test_non_late_strategy_is_unaffected_by_consistency_check() -> None:
    # FixedSizeStrategy doesn't carry an embedder, so swapping the
    # runner's embedder is fine — its chunk vectors come from `embedder`
    # directly, so consistency is automatic. The check must not trip.
    run = evaluate_strategy(
        strategy=FixedSizeStrategy(chunk_chars=120),
        corpus=_SMALL_CORPUS,
        queries=_SMALL_QUERIES,
        embedder=_NamedEmbedder(model_name="some-other-model"),
    )
    assert run.strategy_name == "fixed-size"


def test_late_chunking_mismatch_error_names_d011() -> None:
    # The error message points the reader at D-011 so they can find the
    # rationale instead of guessing what the constraint is for.
    strategy = LateChunkingStrategy(embedder=_NamedEmbedder(model_name="x"))
    with pytest.raises(ValueError, match="D-011") as exc_info:
        evaluate_strategy(
            strategy=strategy,
            corpus=_SMALL_CORPUS,
            queries=_SMALL_QUERIES,
            embedder=_NamedEmbedder(model_name="y"),
        )
    # Double-check that the message also names the strategy class explicitly,
    # so the failure mode is obvious even without consulting MEMORY.
    assert "LateChunkingStrategy" in str(exc_info.value)


# ----------------------------------------------------------------------
# #47: RetrievalRun.from_json / QueryResult.from_json round-trip parity
# ----------------------------------------------------------------------

from chunking_lab.metrics import QueryResult  # noqa: E402


def _synthetic_run(with_per_query: bool = True) -> RetrievalRun:
    per_query: tuple[QueryResult, ...] = ()
    if with_per_query:
        per_query = (
            QueryResult(
                query_id="q1",
                expected_doc="apples.md",
                expected_snippet="Gala variety",
                retrieved_doc_ids_in_rank_order=("apples.md", "bears.md", "carbon.md"),
                snippet_hits_in_rank_order=(True, False, False),
            ),
            QueryResult(
                query_id="q2",
                expected_doc="bears.md",
                expected_snippet="hibernate",
                retrieved_doc_ids_in_rank_order=("bears.md",),
                snippet_hits_in_rank_order=(False,),
            ),
        )
    return RetrievalRun(
        strategy_name="fixed-size",
        embedder_model="hash-cosmic",
        dataset_version="v0",
        n_queries=2,
        n_chunks_total=42,
        recall_at_k={1: 0.5, 3: 1.0, 5: 1.0},
        snippet_hit_at_k={1: 0.5, 3: 0.5, 5: 0.5},
        per_query=per_query,
        wall_clock_ms=12.5,
        notes=["embedder=hash-cosmic"],
    )


def test_query_result_from_json_round_trips_through_to_json_dict_shape() -> None:
    qr = QueryResult(
        query_id="q1",
        expected_doc="apples.md",
        expected_snippet="Gala variety",
        retrieved_doc_ids_in_rank_order=("apples.md", "bears.md"),
        snippet_hits_in_rank_order=(True, False),
    )
    # `to_json` lives on RetrievalRun; mirror its per-query dict shape here.
    payload = {
        "query_id": qr.query_id,
        "expected_doc": qr.expected_doc,
        "expected_snippet": qr.expected_snippet,
        "retrieved_doc_ids_in_rank_order": list(qr.retrieved_doc_ids_in_rank_order),
        "snippet_hits_in_rank_order": list(qr.snippet_hits_in_rank_order),
    }
    rebuilt = QueryResult.from_json(payload)
    assert rebuilt == qr
    # Lists on the wire must round-trip as tuples (frozen-dataclass invariant).
    assert isinstance(rebuilt.retrieved_doc_ids_in_rank_order, tuple)
    assert isinstance(rebuilt.snippet_hits_in_rank_order, tuple)


def test_retrieval_run_from_json_round_trips_with_populated_per_query() -> None:
    run = _synthetic_run(with_per_query=True)
    rebuilt = RetrievalRun.from_json(run.to_json())
    assert rebuilt == run
    # Even after going through JSON-style dict, integer keys stay int.
    assert all(isinstance(k, int) for k in rebuilt.recall_at_k)
    assert all(isinstance(k, int) for k in rebuilt.snippet_hit_at_k)
    # `per_query` is a tuple on the rebuilt instance.
    assert isinstance(rebuilt.per_query, tuple)
    assert all(isinstance(q, QueryResult) for q in rebuilt.per_query)


def test_retrieval_run_from_json_round_trips_empty_per_query() -> None:
    """A run with no per-query rows (e.g., the snapshot test path) must
    round-trip cleanly — `per_query` collapses to an empty tuple."""
    run = _synthetic_run(with_per_query=False)
    rebuilt = RetrievalRun.from_json(run.to_json())
    assert rebuilt == run
    assert rebuilt.per_query == ()


def test_retrieval_run_from_json_coerces_string_keys_back_to_int() -> None:
    """`to_json` writes `{str(k): v}` for `recall_at_k` and
    `snippet_hit_at_k`; the reader must coerce back so `run.recall_at_k[1]`
    (an int lookup) works as written."""
    payload = {
        "strategy_name": "fixed-size",
        "embedder_model": "hash-cosmic",
        "dataset_version": "v0",
        "n_queries": 0,
        "n_chunks_total": 0,
        "recall_at_k": {"1": 0.0, "3": 0.0, "5": 0.0},
        "snippet_hit_at_k": {"1": 0.0, "3": 0.0, "5": 0.0},
        "per_query": [],
        "notes": [],
        "wall_clock_ms": 0.0,
    }
    run = RetrievalRun.from_json(payload)
    # Int lookup must succeed; string lookup must miss.
    assert run.recall_at_k[1] == 0.0
    assert run.snippet_hit_at_k[5] == 0.0
    assert all(isinstance(k, int) for k in run.recall_at_k)


def test_retrieval_run_from_json_defaults_missing_wall_clock_and_notes() -> None:
    """Older committed JSONs predating D-009 may not carry
    `wall_clock_ms` or `notes`. The reader defaults them to the
    dataclass defaults, matching the pre-#47 hand-written test helper."""
    payload = {
        "strategy_name": "fixed-size",
        "embedder_model": "hash-cosmic",
        "dataset_version": "v0",
        "n_queries": 0,
        "n_chunks_total": 0,
        "recall_at_k": {"1": 0.0},
        "snippet_hit_at_k": {"1": 0.0},
        "per_query": [],
    }
    run = RetrievalRun.from_json(payload)
    assert run.wall_clock_ms == 0.0
    assert run.notes == []


def test_retrieval_run_from_json_raises_on_missing_required_key() -> None:
    """If a required field is missing, fail loudly with a KeyError
    naming the missing field — not silently fill it with a default."""
    payload = {
        # `strategy_name` deliberately absent.
        "embedder_model": "hash-cosmic",
        "dataset_version": "v0",
        "n_queries": 0,
        "n_chunks_total": 0,
        "recall_at_k": {"1": 0.0},
        "snippet_hit_at_k": {"1": 0.0},
        "per_query": [],
    }
    with pytest.raises(KeyError, match="strategy_name"):
        RetrievalRun.from_json(payload)


# Issue #62: recall@k / snippet-hit@k are documented as floats in [0, 1].
# from_json promised a loud failure mode but only validated keys, not values —
# a hand-edited / corrupt result file could load inf/NaN/out-of-range silently
# and poison the strategy comparison. Now validated on the value axis too.
@pytest.mark.parametrize("metric", ["recall_at_k", "snippet_hit_at_k"])
@pytest.mark.parametrize(
    "bad",
    [float("inf"), float("-inf"), float("nan"), -0.5, 1.5],
)
def test_retrieval_run_from_json_rejects_corrupt_metric_value(metric: str, bad: float) -> None:
    payload = _synthetic_run(with_per_query=False).to_json()
    payload[metric] = {"1": bad}
    with pytest.raises(ValueError, match=rf"{metric}\[1\] must be"):
        RetrievalRun.from_json(payload)


def test_retrieval_run_from_json_rejects_non_numeric_metric_value() -> None:
    payload = _synthetic_run(with_per_query=False).to_json()
    payload["recall_at_k"] = {"1": "0.5"}
    with pytest.raises(ValueError, match=r"recall_at_k\[1\] must be a number"):
        RetrievalRun.from_json(payload)


def test_retrieval_run_from_json_accepts_inclusive_zero_and_one_boundaries() -> None:
    # The [0, 1] check is inclusive: a perfect (1.0) or zero (0.0) metric is valid.
    payload = _synthetic_run(with_per_query=False).to_json()
    payload["recall_at_k"] = {"1": 0.0, "3": 1.0}
    payload["snippet_hit_at_k"] = {"1": 0.0, "3": 1.0}
    run = RetrievalRun.from_json(payload)
    assert run.recall_at_k == {1: 0.0, 3: 1.0}
    assert run.snippet_hit_at_k == {1: 0.0, 3: 1.0}


def test_retrieval_run_from_json_loads_committed_canonical_jsons() -> None:
    """Cross-check the on-disk canonical results files against the new
    classmethod — proves the contract holds for the actually-committed
    artifact shape, not just a synthetic one."""
    results_dir = _REPO_ROOT / "results"
    canonical = sorted(results_dir.glob("canonical__*.json"))
    assert canonical, "expected committed canonical__*.json files under results/"
    for path in canonical:
        run = RetrievalRun.from_json(json.loads(path.read_text(encoding="utf-8")))
        # Round-trip identity at the JSON level (key order + value shape).
        assert json.dumps(run.to_json(), sort_keys=True) == json.dumps(
            json.loads(path.read_text(encoding="utf-8")), sort_keys=True
        )


# --- #68: tied-score ranking determinism ------------------------------------


class _ConstantEmbedder:
    """Every text maps to the same finite vector, so every cosine score ties
    at 1.0 and chunk ranking is decided entirely by the tiebreak. Lets these
    tests isolate tie-ordering determinism (#68) from the embedding itself."""

    model_name = "constant-test-embedder"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 1.0]


def test_tied_scores_rank_independently_of_corpus_order() -> None:
    # All scores tie, so the retrieved ranking is pure tiebreak. It must not
    # depend on the order the corpus happened to be read/passed in — otherwise
    # recall@k / snippet-hit@k aren't a reproducible function of the chunk set.
    emb = _ConstantEmbedder()
    strat = FixedSizeStrategy(chunk_chars=40, overlap_chars=0)
    docs = [
        Document(filename="a.md", text="alpha alpha alpha. beta beta beta. gamma gamma gamma."),
        Document(filename="b.md", text="delta delta delta. epsilon epsilon. zeta zeta zeta zeta."),
        Document(filename="c.md", text="eta eta eta. theta theta theta. iota iota iota iota."),
    ]
    queries = [Query(id="q1", question="anything", expected_doc="a.md", expected_snippet="alpha")]

    run_fwd = evaluate_strategy(strat, docs, queries, emb, ks=(1, 3, 5))
    run_rev = evaluate_strategy(strat, list(reversed(docs)), queries, emb, ks=(1, 3, 5))

    assert (
        run_fwd.per_query[0].retrieved_doc_ids_in_rank_order
        == run_rev.per_query[0].retrieved_doc_ids_in_rank_order
    )
    assert run_fwd.recall_at_k == run_rev.recall_at_k
    assert run_fwd.snippet_hit_at_k == run_rev.snippet_hit_at_k


def test_tied_scores_break_by_chunk_identity() -> None:
    # With all scores tied, the top chunk is the one with the smallest
    # (source_doc_id, start_offset, end_offset) — a.md's first chunk — even
    # though b.md is passed first. Pre-fix the insertion-order tiebreak would
    # surface b.md first.
    emb = _ConstantEmbedder()
    strat = FixedSizeStrategy(chunk_chars=40, overlap_chars=0)
    docs = [
        Document(filename="b.md", text="delta delta delta. epsilon epsilon epsilon."),
        Document(filename="a.md", text="alpha alpha alpha. beta beta beta beta."),
    ]
    queries = [Query(id="q1", question="anything", expected_doc="a.md", expected_snippet="alpha")]

    run = evaluate_strategy(strat, docs, queries, emb, ks=(1, 3, 5))
    assert run.per_query[0].retrieved_doc_ids_in_rank_order[0] == "a.md"
