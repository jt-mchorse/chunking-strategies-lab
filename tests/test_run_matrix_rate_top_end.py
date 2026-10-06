"""A near-perfect rate is not published as a perfect one (#226).

`_render_no_fabricated_zero` widened a rate only at the zero end, and
`_metric_cell` argued that was the only end that mattered ("`0.000` is the
worst value ... nobody is made to look good"). `1.000` is the best value, and
`.3f` rounds up into it: on main (808bb0e) one miss in 2001 queries,
recall 0.99950025, rendered `1.000` -- byte-identical to a perfect run --
in the summary table and in the per-run stdout line. Same gap as
embedding-model-shootout#178.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_matrix
from scripts.run_matrix import _metric_cell, _render_rate

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (1.0, "1.000"),
        (1999 / 2000, "0.9995"),
        (2000 / 2001, "0.9995"),
        (0.99999, "0.99999"),
        (0.9994, "0.999"),
        (0.5, "0.500"),
        (0.0, "0.000"),
        (0.00025, "0.00025"),  # #198's end still holds
    ],
)
def test_render_rate(value: float, rendered: str) -> None:
    assert _render_rate(value) == rendered


def test_a_perfect_and_a_one_miss_cell_differ() -> None:
    assert _metric_cell({5: 1.0}, 5) != _metric_cell({5: 2000 / 2001}, 5)
    assert _metric_cell({5: 1.0}, 3) == run_matrix._ABSENT_CELL


def test_wall_clock_keeps_the_shared_helper() -> None:
    # A 0.9996 ms run is not "perfect"; the top-end rule is for rates only.
    assert run_matrix._render_no_fabricated_zero(0.9996, places=3) == "1.000"


def test_the_stdout_line_renders_rates_through_the_rate_renderer() -> None:
    # The per-run line in `main` printed the shared helper directly; pin that
    # both of its rate fields go through `_render_rate`.
    src = (_SCRIPTS / "run_matrix.py").read_text(encoding="utf-8")
    line = src[src.index('f"recall@{top_k}=') : src.index("wall_clock={_wall_clock_cell")]
    assert line.count("_render_rate(run.") == 2
    assert "_render_no_fabricated_zero" not in line
