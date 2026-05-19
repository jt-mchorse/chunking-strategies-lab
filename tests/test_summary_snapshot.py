"""Snapshot test for `results/summary.md`.

`scripts/run_matrix.py` writes 5 strategy JSONs and a single `summary.md`
markdown table from the in-memory `RetrievalRun` list. The committed JSONs
carry `wall_clock_ms` baked in, so feeding them back through the renderer
produces a deterministic markdown — but no existing test enforces the
committed `summary.md` matches what `_render_summary` would produce from
the JSONs today.

This module is the missing piece. Pattern parallels the snapshot tests
landed today in `llm-cost-optimizer` (docs/savings.{json,md} + README),
`prompt-regression-suite` (docs/regression_demo.html), and
`rag-production-kit` (README rewriter table).

When the snapshot fails, the regen path is:

    python scripts/run_matrix.py

…then `git diff results/summary.md` before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.metrics import RetrievalRun  # noqa: E402
from scripts.run_matrix import _render_summary  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"
SUMMARY_MD = RESULTS_DIR / "summary.md"

# Strategy order in `_build_strategies` in `scripts/run_matrix.py`. The
# renderer iterates in input order, so the snapshot needs to feed the
# JSONs back in the same order they were originally produced.
STRATEGY_ORDER = (
    "fixed-size",
    "recursive",
    "semantic",
    "late-chunking",
    "structure-aware",
)

REGEN_HINT = (
    "Regenerate the summary:\n"
    "  python scripts/run_matrix.py\n"
    "Then inspect with `git diff results/summary.md` before committing."
)


def _load_run_from_json(path: Path) -> RetrievalRun:
    """Reconstruct a `RetrievalRun` from a committed result JSON.

    The renderer (`_render_summary`) doesn't use `per_query`, so we pass
    an empty tuple for it — recreating the full `QueryResult` list isn't
    needed for the snapshot.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RetrievalRun(
        strategy_name=payload["strategy_name"],
        embedder_model=payload["embedder_model"],
        dataset_version=payload["dataset_version"],
        n_queries=payload["n_queries"],
        n_chunks_total=payload["n_chunks_total"],
        recall_at_k={int(k): v for k, v in payload["recall_at_k"].items()},
        snippet_hit_at_k={int(k): v for k, v in payload["snippet_hit_at_k"].items()},
        per_query=(),
        wall_clock_ms=payload.get("wall_clock_ms", 0.0),
        notes=list(payload.get("notes", [])),
    )


def _committed_run_jsons() -> list[Path]:
    """All committed `results/*.json` files, sorted by glob — the script
    writes one per strategy with a timestamp prefix, so sorting is stable
    within a single matrix run.
    """
    return sorted(RESULTS_DIR.glob("*.json"))


def _runs_in_strategy_order() -> list[RetrievalRun]:
    by_name = {p.stem.split("__", 1)[1]: p for p in _committed_run_jsons()}
    missing = set(STRATEGY_ORDER) - by_name.keys()
    assert not missing, (
        f"Missing committed result JSONs for strategies: {sorted(missing)}.\n"
        f"Found: {sorted(by_name)}.\n{REGEN_HINT}"
    )
    return [_load_run_from_json(by_name[name]) for name in STRATEGY_ORDER]


def test_committed_summary_md_matches_render_from_committed_results() -> None:
    """`_render_summary` over the committed JSONs must equal `results/summary.md`."""
    runs = _runs_in_strategy_order()
    rendered = _render_summary(runs, runs[0].embedder_model)
    committed = SUMMARY_MD.read_text(encoding="utf-8")
    assert rendered == committed, (
        f"results/summary.md is out of sync with `_render_summary` over the "
        f"committed result JSONs.\n{REGEN_HINT}"
    )


def test_committed_summary_md_strategy_set_is_complete() -> None:
    """Every strategy in `STRATEGY_ORDER` must have a committed JSON, and
    no extras. Guards against silently adding or dropping a strategy.
    """
    found_names = {p.stem.split("__", 1)[1] for p in _committed_run_jsons()}
    expected_names = set(STRATEGY_ORDER)
    extra = found_names - expected_names
    missing = expected_names - found_names
    assert not missing, (
        f"Committed strategy JSONs are missing {sorted(missing)}.\n"
        f"Found: {sorted(found_names)}.\n"
        f"Expected: {sorted(expected_names)}.\n"
        "If the strategy lineup changed intentionally, update STRATEGY_ORDER "
        f"in this file and run_matrix._build_strategies together.\n{REGEN_HINT}"
    )
    assert not extra, (
        f"Committed strategy JSONs include unexpected names {sorted(extra)}.\n"
        f"Found: {sorted(found_names)}.\n"
        f"Expected: {sorted(expected_names)}.\n"
        "If the strategy lineup changed intentionally, update STRATEGY_ORDER "
        f"in this file and run_matrix._build_strategies together.\n{REGEN_HINT}"
    )


@pytest.mark.parametrize("strategy", STRATEGY_ORDER)
def test_each_strategy_json_loads_into_a_retrieval_run(strategy: str) -> None:
    """Every committed strategy JSON must round-trip through the loader.

    Catches schema regressions where a new required field is added to
    `RetrievalRun` but the existing JSONs aren't migrated.
    """
    matches = [p for p in _committed_run_jsons() if p.stem.endswith(f"__{strategy}")]
    assert len(matches) == 1, (
        f"Expected exactly one committed JSON for {strategy!r}; found {len(matches)}."
    )
    run = _load_run_from_json(matches[0])
    assert run.strategy_name == strategy
    # `recall_at_k` and `snippet_hit_at_k` must carry the k values the
    # renderer expects (1, 3, 5). If a future bench drops one, the
    # summary renderer's `.get(k, 0)` would silently emit 0 for it —
    # this assertion makes the desync loud at load time.
    for k in (1, 3, 5):
        assert k in run.recall_at_k, f"{strategy}: recall@{k} missing"
        assert k in run.snippet_hit_at_k, f"{strategy}: snippet-hit@{k} missing"
