"""Freshness guard for the tracked canonical result fixtures (#112).

`tests/test_summary_snapshot.py` locks `summary.md` against a re-render of
the *committed* JSONs — an internal-consistency check. It never runs the
strategies, so a canonical fixture that has silently drifted from the
current chunker code (because a behavior fix landed without a
`run_matrix.py --canonical-out` regen) still passes.

That is exactly what happened to the `semantic` fixture (#112): a
semantic-chunker change moved `n_chunks_total` 84 -> 86 and shifted its
recall/snippet-hit metrics, but the tracked fixture kept the pre-change
numbers and CI stayed green.

This module closes that gap. It re-runs every strategy over the pinned
corpus/queries with the dep-free `HashEmbedder` (the same hermetic path
`run_matrix.py` uses by default) and asserts the committed
`canonical__<strategy>.json` matches the fresh run — comparing everything
`RetrievalRun.to_json()` emits *except* `wall_clock_ms`, which is
wall-clock timing and legitimately varies run to run.

The full matrix runs in ~0.2s, so this is CI-safe. When it fails, the fix
is the documented regen:

    python scripts/run_matrix.py --canonical-out

…then `git diff results/` before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.corpus import load_corpus  # noqa: E402
from chunking_lab.metrics import evaluate_strategy  # noqa: E402
from chunking_lab.queries import load_queries  # noqa: E402
from scripts.run_matrix import _build_embedder, _build_strategies  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"

# Timing is the one field that legitimately changes every run; everything
# else `RetrievalRun.to_json()` emits is a deterministic function of the
# corpus, queries, and strategy code.
_NON_DETERMINISTIC_KEYS = ("wall_clock_ms",)

REGEN_HINT = (
    "The committed canonical fixture has drifted from the current chunker.\n"
    "Regenerate it:\n"
    "  python scripts/run_matrix.py --canonical-out\n"
    "Then inspect with `git diff results/` before committing."
)


def _fresh_runs_by_strategy() -> dict[str, dict]:
    """Run every strategy once over the pinned corpus/queries and return
    each strategy's `to_json()` payload keyed by strategy name.
    """
    embedder = _build_embedder("hash")
    corpus = load_corpus()
    queries = load_queries()
    out: dict[str, dict] = {}
    for strat in _build_strategies(embedder):
        run = evaluate_strategy(strat, corpus, queries, embedder)
        out[run.strategy_name] = run.to_json()
    return out


def _strip_timing(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _NON_DETERMINISTIC_KEYS}


_FRESH = _fresh_runs_by_strategy()


@pytest.mark.parametrize("strategy", sorted(_FRESH))
def test_canonical_fixture_matches_fresh_run(strategy: str) -> None:
    """Each committed `canonical__<strategy>.json` must reproduce from a
    fresh `run_matrix.py` execution (modulo wall-clock timing)."""
    fixture_path = RESULTS_DIR / f"canonical__{strategy}.json"
    assert fixture_path.exists(), f"missing canonical fixture: {fixture_path}"

    committed = json.loads(fixture_path.read_text(encoding="utf-8"))
    fresh = _FRESH[strategy]

    assert _strip_timing(committed) == _strip_timing(fresh), (
        f"canonical fixture for {strategy!r} is stale.\n{REGEN_HINT}"
    )


def test_every_strategy_has_a_canonical_fixture() -> None:
    """Guard against a strategy being added to the runner but not to the
    tracked fixture set (the inverse drift)."""
    fixture_strategies = {p.stem.split("__", 1)[1] for p in RESULTS_DIR.glob("canonical__*.json")}
    assert set(_FRESH) == fixture_strategies, (
        f"runner strategies {sorted(_FRESH)} != canonical fixtures {sorted(fixture_strategies)}.\n"
        f"{REGEN_HINT}"
    )
