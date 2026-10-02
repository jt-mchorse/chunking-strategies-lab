"""The comparison notebook gets the two `run_matrix` fixes it missed (#210).

`notebooks/_build_notebook.py` publishes the same run JSONs `run_matrix.py`
renders, and the notebook is committed with executed outputs. Measured on
`main` at `d11fcb3`, by exec-ing the builder's cell sources:

* a fresh `--strategy fixed-size --ks 1,10` run beside four canonical
  `--ks 1,3,5` files -> `_RECALL_CELL` raised `KeyError: '10'`, because `ks`
  came from `runs[0]` alone (#82) while `run_matrix` has taken the union since
  #160;
* a pre-D-009 JSON with no `wall_clock_ms` -> the load cell printed
  `wall=0ms` and the latency chart labelled a bar `0ms`, the flattering
  artefact D-016 forbids.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nbformat")

from notebooks import _build_notebook  # noqa: E402
from scripts import run_matrix  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical_runs() -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((_REPO_ROOT / "results").glob("canonical__*.json"))
    ]


def _mixed_k_runs() -> list[dict[str, Any]]:
    runs = _canonical_runs()
    fresh = dict(runs[0])
    fresh["recall_at_k"] = {"1": 0.5, "10": 0.9}
    fresh["snippet_hit_at_k"] = {"1": 0.2, "10": 0.4}
    return [fresh, *runs[1:]]


def _wall_label():
    """`_wall_label` as the load cell defines it, compiled on its own."""
    tree = ast.parse(_build_notebook._LOAD_CELL)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_wall_label"]
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<load-cell>", "exec"), ns)  # noqa: S102
    return ns["_wall_label"]


def _chart_ns(runs: list[dict[str, Any]]) -> dict[str, Any]:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return {
        "runs": runs,
        "plt": plt,
        "embedder": "HashEmbedder",
        "n_queries": 12,
        "_wall_label": _wall_label(),
    }


# ----------------------------------------------------------------------
# 1. Mixed k sets
# ----------------------------------------------------------------------


def test_the_premise_canonical_runs_share_one_k_set() -> None:
    """Control: the mixed payload below differs from the committed one only in
    the fresh run's k set."""
    assert {tuple(sorted(r["recall_at_k"])) for r in _canonical_runs()} == {("1", "3", "5")}


def test_chart_cells_take_the_union_of_k_across_runs() -> None:
    ns = _chart_ns(_mixed_k_runs())
    exec(_build_notebook._RECALL_CELL, ns)  # noqa: S102
    exec(_build_notebook._SNIPPET_CELL, ns)  # noqa: S102
    ns["plt"].close("all")
    assert ns["ks"] == [1, 3, 5, 10]


def test_a_k_a_run_did_not_measure_draws_no_bar() -> None:
    """Absent, never a fabricated 0: the #160 rule for a missing cell."""
    ns = _chart_ns(_mixed_k_runs())
    exec(_build_notebook._RECALL_CELL, ns)  # noqa: S102
    ax = ns["plt"].gcf().axes[0]
    by_label = {c.get_label(): [b.get_height() for b in c] for c in ax.containers}
    ns["plt"].close("all")
    # The fresh run (index 0) has no recall@3; the others have no recall@10.
    assert math.isnan(by_label["recall@3"][0])
    assert all(math.isnan(h) for h in by_label["recall@10"][1:])
    assert by_label["recall@10"][0] == pytest.approx(0.9)


def test_the_committed_payload_still_charts_one_three_five() -> None:
    ns = _chart_ns(_canonical_runs())
    exec(_build_notebook._RECALL_CELL, ns)  # noqa: S102
    ns["plt"].close("all")
    assert ns["ks"] == [1, 3, 5]


# ----------------------------------------------------------------------
# 2. A defaulted wall-clock is not "0ms"
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ms", [0.0, 0.0004, 0.4, 0.5, 0.51, 19.99, 85.13, 1234.5])
def test_wall_label_is_run_matrixs_cell_plus_a_unit(ms: float) -> None:
    """Parity with `run_matrix._wall_clock_cell`, so the two surfaces cannot
    publish one value two ways."""
    cell = run_matrix._wall_clock_cell(ms)
    expected = cell if cell == run_matrix._ABSENT_CELL else f"{cell}ms"
    assert _wall_label()(ms) == expected


def test_the_load_cell_prints_a_missing_wall_clock_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    shutil.copytree(_REPO_ROOT / "results", results)
    legacy = results / "canonical__structure-aware.json"
    payload = json.loads(legacy.read_text(encoding="utf-8"))
    payload.pop("wall_clock_ms")
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("matplotlib")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(_build_notebook._LOAD_CELL, {})  # noqa: S102
    line = next(ln for ln in out.getvalue().splitlines() if "structure-aware" in ln)
    assert line.endswith("wall=—"), line
    assert "0ms" not in line


def test_the_latency_chart_labels_a_missing_wall_clock_as_absent() -> None:
    runs = _canonical_runs()
    runs[-1] = {k: v for k, v in runs[-1].items() if k != "wall_clock_ms"}
    ns = _chart_ns(runs)
    ns["strategies"] = [r["strategy_name"] for r in runs]
    exec(_build_notebook._LATENCY_CELL, ns)  # noqa: S102
    labels = [t.get_text() for t in ns["plt"].gcf().axes[0].texts]
    ns["plt"].close("all")
    assert labels[-1] == "—"
    assert "0ms" not in labels
    assert all(label.endswith("ms") for label in labels[:-1])
