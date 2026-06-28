"""Tests for the comparison notebook (#4).

Verifies (a) the notebook file is committed, structurally valid, and has the
three expected charts; (b) the chart-data extraction logic in the first cell
works against the real result JSONs under `results/`.

The plot extras (`[notebook]`) install `nbformat`. If `nbformat` is missing
(operator running base CI without extras), these tests skip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK = _REPO_ROOT / "notebooks" / "comparison.ipynb"

# Build script lives next to the notebook; import it so the lock test below can
# inspect the actual `_LOAD_CELL` source that ships into comparison.ipynb.
_NOTEBOOKS_DIR = _REPO_ROOT / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))


def _stamp_rank(stamp: str) -> tuple[int, str]:
    """Recency key mirroring `_build_notebook._LOAD_CELL`'s `_stamp_rank`.

    Timestamped runs use a digit prefix (`YYYYMMDDThhmmss`); committed fixtures
    use the literal `canonical`. Plain `>` ranks `canonical` above every
    digit-prefixed stamp ('c' > '0'-'9'), silently shadowing a fresh run — the
    bug this guards. Rank `canonical` lowest so any timestamped run wins.
    """
    return (0, "") if stamp == "canonical" else (1, stamp)


def test_notebook_committed():
    assert _NOTEBOOK.exists(), f"comparison notebook missing at {_NOTEBOOK}"


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(_NOTEBOOK, as_version=4)
    # Will raise if structure violates schema.
    nbformat.validate(nb)


def test_notebook_has_at_least_three_charts():
    """Acceptance criterion: at least 3 charts (recall, faithfulness, latency)."""
    nb = nbformat.read(_NOTEBOOK, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    plot_calls = sum(c.source.count("plt.show") for c in code_cells)
    assert plot_calls >= 3, f"notebook has only {plot_calls} plt.show() calls; need ≥3 charts"


def test_notebook_takeaways_section_present():
    nb = nbformat.read(_NOTEBOOK, as_version=4)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "## Takeaways" in md_text, "notebook missing the Takeaways section"
    # Honest-numbers disclosure must appear so a reader doesn't misread the charts.
    assert "HashEmbedder" in md_text
    assert "minilm" in md_text


def test_results_dir_loadable_by_notebook_helper():
    """Smoke-test the loader logic in the notebook's first code cell against the
    committed `results/` directory — the canonical case the notebook renders."""
    results_dir = _REPO_ROOT / "results"
    assert results_dir.is_dir()
    files = sorted(results_dir.glob("*.json"))
    assert len(files) >= 1, "no result JSON files committed under results/"
    latest_by_strategy: dict[str, dict] = {}
    latest_stamp: dict[str, str] = {}
    for p in files:
        stamp = p.name.split("__")[0]
        payload = json.loads(p.read_text(encoding="utf-8"))
        name = payload["strategy_name"]
        if name not in latest_stamp or _stamp_rank(stamp) > _stamp_rank(latest_stamp[name]):
            latest_by_strategy[name] = payload
            latest_stamp[name] = stamp
    # Every loaded run carries every chart-data field the notebook references.
    for payload in latest_by_strategy.values():
        assert "recall_at_k" in payload
        assert "snippet_hit_at_k" in payload
        # New in this PR: wall_clock_ms. .get() in the notebook tolerates 0.0
        # for legacy JSONs, but freshly-committed ones must carry the field.
        assert "wall_clock_ms" in payload
        assert isinstance(payload["wall_clock_ms"], (int, float))
        for k in ("1", "3", "5"):
            assert k in payload["recall_at_k"]
            assert k in payload["snippet_hit_at_k"]


def test_notebook_executed_outputs_exist():
    """The committed notebook ships with executed outputs so reviewers see the
    charts without having to run jupyter locally."""
    nb = nbformat.read(_NOTEBOOK, as_version=4)
    code_cells_with_output = [
        c for c in nb.cells if c.cell_type == "code" and getattr(c, "outputs", [])
    ]
    # At least the three plot cells should have outputs (the figure images).
    assert len(code_cells_with_output) >= 3, (
        f"only {len(code_cells_with_output)} executed code cells with outputs; "
        "did you forget `jupyter nbconvert --execute`?"
    )


def _select_latest(results_dir: Path) -> dict[str, str]:
    """Stamp-selection logic from the notebook's load cell, returning the
    chosen stamp per strategy (enough to assert which file won)."""
    latest_stamp: dict[str, str] = {}
    for p in sorted(results_dir.glob("*.json")):
        stamp = p.name.split("__")[0]
        name = json.loads(p.read_text(encoding="utf-8"))["strategy_name"]
        if name not in latest_stamp or _stamp_rank(stamp) > _stamp_rank(latest_stamp[name]):
            latest_stamp[name] = stamp
    return latest_stamp


def test_load_latest_prefers_timestamped_run_over_canonical(tmp_path: Path):
    """Regression for #78: a fresh timestamped run must beat the committed
    `canonical__*` baseline. The pre-fix `stamp > latest` compared lexically,
    and 'canonical' > any digit prefix, so the stale fixture always won.

    The existing `test_results_dir_loadable_by_notebook_helper` only runs
    against the canonical-only `results/`, so it never exercises this tie.
    """
    (tmp_path / "canonical__fixed-size.json").write_text(
        json.dumps({"strategy_name": "fixed-size", "recall_at_k": {"5": 0.111}}),
        encoding="utf-8",
    )
    (tmp_path / "20260627T120000__fixed-size.json").write_text(
        json.dumps({"strategy_name": "fixed-size", "recall_at_k": {"5": 0.900}}),
        encoding="utf-8",
    )

    chosen = _select_latest(tmp_path)
    assert chosen["fixed-size"] == "20260627T120000", (
        "loader kept the stale canonical fixture instead of the fresh "
        "timestamped run (lexicographic stamp-compare bug, #78)"
    )
    # Sanity: a far-future stamp also beats canonical (not just this date).
    (tmp_path / "99990101T000000__fixed-size.json").write_text(
        json.dumps({"strategy_name": "fixed-size", "recall_at_k": {"5": 0.5}}),
        encoding="utf-8",
    )
    assert _select_latest(tmp_path)["fixed-size"] == "99990101T000000"


def test_build_notebook_load_cell_ships_the_stamp_rank_fix():
    """Lock that the emitted load cell actually carries the rank-based compare.

    Combined with `test_build_notebook_snapshot.py` (notebook == build script),
    this guarantees the shipped `comparison.ipynb` ranks `canonical` lowest and
    can't silently regress to a bare `stamp > latest_stamp[name]` compare.
    """
    import _build_notebook  # noqa: PLC0415

    cell = _build_notebook._LOAD_CELL
    assert "_stamp_rank" in cell, "load cell lost the _stamp_rank recency key (#78)"
    assert "_stamp_rank(stamp) > _stamp_rank(latest_stamp[name])" in cell, (
        "load cell no longer compares via _stamp_rank — a bare lexicographic "
        "`stamp > latest_stamp[name]` would reintroduce the canonical-shadows-"
        "fresh-run bug (#78)"
    )


def _non_default_ks_runs() -> list[dict]:
    """Result JSONs as produced by `run_matrix.py --ks 2,4` — no 1/3/5 keys."""
    return [
        {
            "strategy_name": "fixed-size",
            "n_chunks_total": 10,
            "recall_at_k": {"2": 0.4, "4": 0.6},
            "snippet_hit_at_k": {"2": 0.1, "4": 0.2},
            "wall_clock_ms": 5.0,
            "embedder_model": "HashEmbedder",
            "n_queries": 4,
        },
        {
            "strategy_name": "recursive",
            "n_chunks_total": 12,
            "recall_at_k": {"2": 0.5, "4": 0.7},
            "snippet_hit_at_k": {"2": 0.15, "4": 0.25},
            "wall_clock_ms": 6.0,
            "embedder_model": "HashEmbedder",
            "n_queries": 4,
        },
    ]


def test_chart_cells_handle_non_default_ks():
    """Regression for #82: the recall/snippet chart cells must derive `ks` from
    the loaded runs, not hardcode 1/3/5, so a notebook built from a non-default
    `--ks` (a supported run_matrix.py flag) renders instead of raising KeyError.

    Pre-fix `ks = [1, 3, 5]` + `r["recall_at_k"][str(k)]` raised `KeyError: '1'`
    on a `--ks 2,4` payload. `run_matrix.py:104` already derives ks dynamically;
    this locks the notebook builder to the same behavior.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless: plt.show() is a no-op
    import _build_notebook  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs = _non_default_ks_runs()
    ns: dict = {"runs": runs, "plt": plt, "embedder": "HashEmbedder", "n_queries": 4}
    # Each chart cell must run without KeyError on the --ks 2,4 payload.
    exec(_build_notebook._RECALL_CELL, ns)  # noqa: S102  (defines ks, x, width, strategies)
    exec(_build_notebook._SNIPPET_CELL, ns)  # noqa: S102  (reuses ns from recall cell)
    plt.close("all")
    assert ns["ks"] == [2, 4], f"chart cells did not derive ks from runs: {ns.get('ks')!r}"


def test_load_cell_reports_max_available_k_not_hardcoded_five():
    """Regression for #82: the load cell's per-run print must index the largest
    k actually present (`max(...)`), not a hardcoded "5" that KeyErrors on a
    non-default `--ks`. Also lock that the emitted source dropped the literals.
    """
    import _build_notebook  # noqa: PLC0415

    load = _build_notebook._LOAD_CELL
    recall = _build_notebook._RECALL_CELL
    assert 'recall_at_k"]["5"]' not in load, "load cell still hardcodes recall_at_k['5'] (#82)"
    assert "ks = [1, 3, 5]" not in recall, "recall chart cell still hardcodes ks = [1, 3, 5] (#82)"

    # The kmax derivation the load cell now uses must pick the largest present k
    # and index it without raising, on a payload that omits 5.
    for r in _non_default_ks_runs():
        kmax = max(int(k) for k in r["recall_at_k"])
        assert kmax == 4
        float(r["recall_at_k"][str(kmax)])  # must not raise
        float(r["snippet_hit_at_k"][str(kmax)])
