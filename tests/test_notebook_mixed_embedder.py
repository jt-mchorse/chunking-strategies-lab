"""The notebook's chart titles do not borrow the first run's embedder (#221).

#211 fixed `ks = runs[0]...` because the loader keeps the newest file per
strategy, so one fresh `run_matrix --strategy X` run sits beside four canonical
files. The same cell kept `embedder = runs[0][...]` and `n_queries = runs[0][...]`,
and fixed-size always loads first: one fresh `--embedder minilm` fixed-size run
titled every chart MiniLM over four HashEmbedder bars.

These execute the builder's real cell sources against a temporary `results/`
and read the titles matplotlib actually drew.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from notebooks._build_notebook import (  # noqa: E402
    _LATENCY_CELL,
    _LOAD_CELL,
    _RECALL_CELL,
    _SNIPPET_CELL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _results(tmp_path: Path, *, fresh: dict[str, Any] | None = None) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    for p in (REPO_ROOT / "results").glob("canonical__*.json"):
        shutil.copy(p, results / p.name)
    if fresh is not None:
        base = json.loads((results / "canonical__fixed-size.json").read_text(encoding="utf-8"))
        base.update(fresh)
        (results / "20261002T090000__fixed-size.json").write_text(
            json.dumps(base), encoding="utf-8"
        )
    return tmp_path


def _run_cells(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.chdir(root)
    plt.switch_backend("Agg")
    ns: dict[str, Any] = {}
    titles = []
    exec(_LOAD_CELL, ns)
    for cell in (_RECALL_CELL, _SNIPPET_CELL, _LATENCY_CELL):
        exec(cell, ns)
        titles.append(plt.gcf().axes[0].get_title())
        plt.close("all")
    return titles, capsys.readouterr().out


def test_a_homogeneous_set_titles_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    titles, out = _run_cells(_results(tmp_path), monkeypatch, capsys)
    assert all(t.endswith("embedder=HashEmbedder · n_queries=12") for t in titles), titles
    assert "MIXED" not in out
    assert "WARNING" not in out


def test_a_mixed_embedder_set_says_mixed_in_every_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    titles, out = _run_cells(
        _results(tmp_path, fresh={"embedder_model": MINILM}), monkeypatch, capsys
    )
    assert len(titles) == 3
    for t in titles:
        assert f"embedder=MIXED (HashEmbedder, {MINILM})" in t, t
        assert f"embedder={MINILM} " not in t  # the first run's value, borrowed
    assert "WARNING: these runs are not comparable" in out
    assert f"embedder={MINILM}  n_queries=12" in out  # the per-run line names it


def test_a_mixed_query_count_says_mixed_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh_rows = json.loads(
        (REPO_ROOT / "results" / "canonical__fixed-size.json").read_text(encoding="utf-8")
    )["per_query"][:6]
    # Six of the twelve rows: still a self-consistent record shape for the loader,
    # which reads the JSON directly (#218's note: no RetrievalRun rule reaches it).
    titles, _ = _run_cells(
        _results(tmp_path, fresh={"n_queries": 6, "per_query": fresh_rows}), monkeypatch, capsys
    )
    assert all("n_queries=MIXED (12, 6)" in t for t in titles), titles
