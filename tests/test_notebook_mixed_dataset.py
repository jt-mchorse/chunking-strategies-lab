"""The notebook's MIXED warning covers the dataset version too (#238).

#221 made the load cell and chart titles say `MIXED` when the newest run per
strategy disagreed on `embedder_model` or `n_queries`. `dataset_version` was
not checked. Measured on `main`: one fresh `run_matrix.py --strategy
fixed-size --dataset-version v1-newcorpus` run beside the four canonical `v0`
files printed `Loaded 5 strategy runs · embedder=HashEmbedder · n_queries=12`
with no warning, and every chart compared two corpora as one.
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


def _results(tmp_path: Path, *, fresh: dict[str, Any] | None = None) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    for p in (REPO_ROOT / "results").glob("canonical__*.json"):
        shutil.copy(p, results / p.name)
    if fresh is not None:
        base = json.loads((results / "canonical__fixed-size.json").read_text(encoding="utf-8"))
        base.update(fresh)
        (results / "20261007T090000__fixed-size.json").write_text(
            json.dumps(base), encoding="utf-8"
        )
    return tmp_path


def _run_cells(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.chdir(root)
    plt.switch_backend("Agg")
    ns: dict[str, Any] = {}
    titles = []
    exec(_LOAD_CELL, ns)  # noqa: S102
    for cell in (_RECALL_CELL, _SNIPPET_CELL, _LATENCY_CELL):
        exec(cell, ns)  # noqa: S102
        titles.append(plt.gcf().axes[0].get_title())
        plt.close("all")
    return titles, capsys.readouterr().out


def test_a_mixed_dataset_set_warns_and_says_so_in_every_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    canonical = json.loads(
        (REPO_ROOT / "results" / "canonical__fixed-size.json").read_text(encoding="utf-8")
    )
    base_version = canonical["dataset_version"]
    titles, out = _run_cells(
        _results(tmp_path, fresh={"dataset_version": "v1-newcorpus"}), monkeypatch, capsys
    )
    expected = f"dataset=MIXED ({', '.join(sorted([base_version, 'v1-newcorpus']))})"
    assert "WARNING" in out
    assert expected in out
    assert all(expected in t for t in titles), titles
    assert "dataset=v1-newcorpus" in out  # the per-run line names each run's corpus


def test_a_homogeneous_set_names_no_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    titles, out = _run_cells(_results(tmp_path), monkeypatch, capsys)
    assert "dataset=" not in out
    assert not any("dataset=" in t for t in titles)
    assert "WARNING" not in out
