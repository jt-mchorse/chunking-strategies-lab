"""Tests for the comparison notebook (#4).

Verifies (a) the notebook file is committed, structurally valid, and has the
three expected charts; (b) the chart-data extraction logic in the first cell
works against the real result JSONs under `results/`.

The plot extras (`[notebook]`) install `nbformat`. If `nbformat` is missing
(operator running base CI without extras), these tests skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")


_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK = _REPO_ROOT / "notebooks" / "comparison.ipynb"


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
        if name not in latest_stamp or stamp > latest_stamp[name]:
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
