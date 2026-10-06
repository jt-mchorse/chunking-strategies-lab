"""The notebook's load cell renders rates the way `run_matrix` does (#232).

It printed recall and snippet-hit with a bare `.3f`, so 0.9995 read `1.000`
(the #226 collapse) and 0.0004 read `0.000` (the #198 one), though both fixes
had landed in `run_matrix._render_rate`. The cell keeps inline copies of
run_matrix's renderers (`_wall_label`, #210); `_rate_label` is the rate one,
and these arms hold it to the original.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from notebooks import _build_notebook
from scripts.run_matrix import _render_rate

ROOT = Path(__file__).resolve().parent.parent


def _cell_function(name: str) -> Any:
    tree = ast.parse(_build_notebook._LOAD_CELL)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<load-cell>", "exec"), ns)  # noqa: S102
    return ns[name]


@pytest.mark.parametrize(
    "value", [0.0, 1.0, 0.9995, 0.0004, 0.917, 0.5, 0.083, 0.99999999, 1e-9, 0.0005, 0.99949]
)
def test_rate_label_is_run_matrix_render_rate(value: float) -> None:
    assert _cell_function("_rate_label")(value) == _render_rate(value)


def test_the_print_goes_through_rate_label_for_both_rates() -> None:
    cell = _build_notebook._LOAD_CELL
    assert "recall@{kmax}={_rate_label(recall_top)}" in cell
    assert "snippet-hit@{kmax}={_rate_label(snippet_top)}" in cell
    assert ":.3f}  snippet-hit" not in cell


def test_the_committed_output_is_what_the_new_cell_prints(monkeypatch) -> None:
    # The committed notebook was not re-executed for this change; that is only
    # honest if the new cell prints exactly what the old one did on the
    # committed results. Every committed rate is an ordinary value, so it must.
    pytest.importorskip("matplotlib")
    nb = json.loads((ROOT / "notebooks" / "comparison.ipynb").read_text(encoding="utf-8"))
    (cell,) = [
        c for c in nb["cells"] if c["cell_type"] == "code" and "_rate_label" in "".join(c["source"])
    ]
    committed = "".join(
        "".join(o["text"]) for o in cell["outputs"] if o.get("output_type") == "stream"
    )
    monkeypatch.chdir(ROOT / "notebooks")
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile("".join(cell["source"]), "<load-cell>", "exec"), {})  # noqa: S102
    assert out.getvalue() == committed
