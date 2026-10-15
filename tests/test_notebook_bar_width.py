"""Each strategy's bars stay inside its own tick group, for any number of k's (#270).

`_RECALL_CELL` centred `len(ks)` bars on each strategy tick with a fixed
`width = 0.25`, and ticks sit 1.0 apart. That fits four k's at most. Measured
on `main` at `70b593b` with a real `run_matrix.py --ks 1,2,3,4,5` run:

    fixed-size      recall@5   x=[0.375,0.625]
    recursive       recall@1   x=[0.375,0.625]

The same interval. recall@5 is drawn last, so recursive's recall@1 was hidden
behind fixed-size's bar, and the chart gave no sign a value was missing. Five
k's reach the chart through one `--ks` run, or through #210's union of a
fresh `--ks 1,10,20` run with the canonical `1,3,5` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nbformat")

from notebooks import _build_notebook  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical_runs() -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((_REPO_ROOT / "results").glob("canonical__*.json"))
    ]


def _runs_at(ks: list[int]) -> list[dict[str, Any]]:
    runs = []
    for r in _canonical_runs():
        r = dict(r)
        r["recall_at_k"] = {str(k): 0.5 for k in ks}
        r["snippet_hit_at_k"] = {str(k): 0.25 for k in ks}
        runs.append(r)
    return runs


def _draw(runs: list[dict[str, Any]], cell: str) -> tuple[dict[str, Any], list[list[tuple]]]:
    """Exec the chart cell(s); return the namespace and, per chart, the bars as
    `(strategy_index, left, right)`."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ns: dict[str, Any] = {
        "runs": runs,
        "plt": plt,
        "embedder": "HashEmbedder",
        "n_queries": 12,
        "dataset_note": "",
    }
    exec(_build_notebook._RECALL_CELL, ns)  # noqa: S102
    charts = [plt.gcf().axes[0]]
    if cell == "snippet":
        exec(_build_notebook._SNIPPET_CELL, ns)  # noqa: S102  (reuses ks, x, width)
        charts = [plt.gcf().axes[0]]
    out = [
        [
            (si, p.get_x(), p.get_x() + p.get_width())
            for c in ax.containers
            for si, p in enumerate(c.patches)
        ]
        for ax in charts
    ]
    plt.close("all")
    return ns, out


@pytest.mark.parametrize("cell", ["recall", "snippet"])
@pytest.mark.parametrize("n_ks", range(1, 9))
def test_no_bar_crosses_into_a_neighbouring_strategys_group(cell: str, n_ks: int) -> None:
    ks = [1, 2, 3, 4, 5, 10, 20, 50][:n_ks]
    _, (bars,) = _draw(_runs_at(ks), cell)
    n_strategies = len(_canonical_runs())
    assert len(bars) == n_ks * n_strategies
    for si in range(n_strategies - 1):
        right_edge = max(r for s, _, r in bars if s == si)
        next_left = min(left for s, left, _ in bars if s == si + 1)
        assert right_edge < next_left, (
            f"{cell} chart, {n_ks} k's: strategy {si}'s bars end at {right_edge:.3f}, "
            f"strategy {si + 1}'s begin at {next_left:.3f}"
        )


def test_the_five_k_case_the_issue_measured() -> None:
    """The exact interval #270 printed is no longer shared."""
    _, (bars,) = _draw(_runs_at([1, 2, 3, 4, 5]), "recall")
    spans = [(round(left, 6), round(right, 6)) for _, left, right in bars]
    assert len(set(spans)) == len(spans), "two bars occupy the same interval"


def test_the_canonical_chart_is_unchanged() -> None:
    """Control: three k's keep the 0.25 width and the positions the committed
    PNG outputs were drawn at, which is why the notebook was not re-executed."""
    ns, (bars,) = _draw(_canonical_runs(), "recall")
    assert ns["ks"] == [1, 3, 5]
    assert ns["width"] == 0.25
    lefts = sorted({round(left, 6) for _, left, _ in bars})
    expected = sorted(
        {
            round(x + (i - 1) * 0.25 - 0.125, 6)
            for x in range(len(ns["strategies"]))
            for i in range(3)
        }
    )
    assert lefts == expected
