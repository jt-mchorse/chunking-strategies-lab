"""A strictly positive recall@k / snippet-hit@k is never published as ``0.000``
(#198).

`_metric_cell`'s docstring says it exists to distinguish "measured zero" from
"not measured". It drew one half of that — absent vs present — and then
published a *present, strictly positive, sub-resolution* value as ``0.000``,
the same cell a genuine ``0.0`` gets. The distinction it names is exactly the
one it lost: a run that found the gold chunk for one query in four thousand
published a row byte-identical to a run that found nothing.

Fourth member of the class D-016 (#196), `embedding-model-shootout#149` and
`vector-search-at-scale#148` belong to, and the one #196 deferred.

**The population was four sites, not the two the issue named.**
`run_matrix.py`'s per-strategy stdout line printed `recall@k` and
`snippet-hit@k` at a bare ``.3f``. The comment fifteen lines above it makes
precisely this argument — "stdout is a publication surface like the summary
table, and a silent ``0.000`` here would be the same fabricated measurement" —
and it was written *about the recall cells*. #196 then applied that reasoning
to the ``wall_clock=`` field on the **next line** and left these two. A fix's
own wording pointing at the sites it missed; the AST arm at the bottom of this
module is what makes the fifth one fail instead of ship.

**Which half of the `ems#149` argument transfers**, stated because the same
shape got opposite answers in three repos this month. The *arithmetic* half
does: a present measurement below half of ``10**-3`` reaches the identical
cell, and only the rendering collapses — the JSONs carry ``0.00025``
throughout. The *extreme-default* half does not. For wall-clock (#196) and for
cost (`vsas#148`) the fabricated zero is the **best** value in its column, so
it flatters a "which is fastest / cheapest" read. Here ``0.000`` is the
**worst** value on both columns, so the collapse *understates*. Nobody is made
to look good; what is lost is the distinction itself.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from chunking_lab.metrics import RetrievalRun

_ROOT = Path(__file__).resolve().parents[1]
_RUN_MATRIX_PATH = _ROOT / "scripts" / "run_matrix.py"


def _load_run_matrix():
    spec = importlib.util.spec_from_file_location(
        "run_matrix_for_metric_zero_tests", _RUN_MATRIX_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_matrix = _load_run_matrix()


def _run(
    *,
    recall: float,
    snippet: float,
    name: str = "fixed-size",
    k: int = 5,
    wall_clock_ms: float = 20.0,
) -> RetrievalRun:
    """A `RetrievalRun` built through the validating read path, not the
    constructor.

    `from_json` is the reachability the issue rests on: `scripts/run_matrix.py
    --results-dir <dir>` walks hand-edited or externally generated result files,
    and every value asserted on below is one the loader actually accepts. The
    canonical 12-query corpus (D-002) is not the only `n_queries` it takes — a
    recall of `0.00025` is one hit in four thousand queries, which is the shape
    of an externally produced sweep rather than of this repo's own fixture.
    """
    return RetrievalRun.from_json(
        {
            "strategy_name": name,
            "embedder_model": "hash",
            "dataset_version": "v0",
            "n_queries": 4000,
            "n_chunks_total": 29,
            "recall_at_k": {str(k): recall},
            "snippet_hit_at_k": {str(k): snippet},
            "per_query": [],
            "wall_clock_ms": wall_clock_ms,
        }
    )


def _row_cells(rendered: str, strategy: str = "fixed-size") -> list[str]:
    """Pull the GFM data row for `strategy` apart into cells.

    Reads the *table*, not a return value. `vsas#148` measured why: its
    formatter-level arms stayed green against a call-site revert, because the
    formatter was already correct and only the call site was not.
    """
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("| ") and line[2:].startswith(strategy + " ")
    ]
    assert len(rows) == 1, f"expected exactly one row for {strategy!r}, got {rows}"
    return [cell.strip() for cell in rows[0].strip().strip("|").split("|")]


def _render(run: RetrievalRun) -> str:
    return run_matrix._render_summary([run], run.embedder_model)


# Read off a printed run of the helper, not retyped from the issue. The boundary
# is NOT 0.0005: `f"{0.0005:.3f}"` is `'0.001'`, because the nearest double to
# 0.0005 sits just above it — which is the same reason D-016 put the guard on
# the rendered shape rather than on a magnitude threshold. A `if v < 0.0005`
# rule would be wrong here in the opposite direction from the `0.5` case that
# motivated it, and neither is a case anyone would have guessed right.
_BOUNDARY = (
    (0.0, "0.000"),
    (1e-9, "1e-09"),
    (0.00025, "0.00025"),
    (0.0004, "0.0004"),
    (0.00049, "0.00049"),
    (0.0005, "0.001"),
    (0.001, "0.001"),
    (0.083, "0.083"),
    (0.5, "0.500"),
    (1.0, "1.000"),
)


@pytest.mark.parametrize(("value", "expected"), _BOUNDARY, ids=[repr(v) for v, _ in _BOUNDARY])
def test_the_published_recall_cell_across_the_boundary(value: float, expected: str) -> None:
    """The recall@k cell as it appears in the table, either side of the collapse."""
    cells = _row_cells(_render(_run(recall=value, snippet=0.5)))
    assert cells[2] == expected, cells


@pytest.mark.parametrize(("value", "expected"), _BOUNDARY, ids=[repr(v) for v, _ in _BOUNDARY])
def test_the_published_snippet_hit_cell_across_the_boundary(value: float, expected: str) -> None:
    """Its own arm, not riding on the recall one (AC3).

    The two maps are independent: `ems#149` had the nDCG cell collapse at a
    value where recall survived, so a shared arm would have missed one of them.
    Here the same `_metric_cell` renders both, but that is a fact about today's
    implementation rather than a property of the columns — and it is exactly
    the kind of fact a later refactor changes silently.
    """
    cells = _row_cells(_render(_run(recall=0.5, snippet=value)))
    assert cells[3] == expected, cells


def test_a_sub_resolution_row_is_not_byte_identical_to_a_zero_row() -> None:
    """The harm, stated as the harm rather than as a formatting property.

    One gold chunk found in four thousand queries must not publish the same row
    as a run that found nothing. This is the assertion the issue's measurement
    made, and it is the one a reader of `results/summary.md` cares about.
    """
    found_nothing = _row_cells(_render(_run(recall=0.0, snippet=0.0)), "fixed-size")
    found_one = _row_cells(_render(_run(recall=0.00025, snippet=0.00025)), "fixed-size")
    assert found_nothing != found_one, found_nothing


def test_a_genuine_zero_recall_still_renders_narrow() -> None:
    """GREEN against the unfixed tree, deliberately.

    This is the arm that rejects a fix which widened the whole column, or one
    that folded the `_ABSENT_CELL` sentinel into the shared helper. A `recall@k`
    of `0.0` is a real measurement of a strategy that found nothing — unlike
    `wall_clock_ms`, whose `0.0` is D-009's backward-compat default and
    therefore *is* that field's "not measured" sentinel. The two callers
    disagree about what a genuine zero means and both are right, which is why
    `_render_no_fabricated_zero` widens a non-zero value and nothing else.

    `results/summary.md` contains six such cells today, so this is the ordinary
    case rather than a corner.
    """
    cells = _row_cells(_render(_run(recall=0.0, snippet=0.0)))
    assert cells[2] == "0.000", cells
    assert cells[3] == "0.000", cells


def test_an_absent_k_is_still_published_as_not_measured() -> None:
    """GREEN against the unfixed tree. #160's lock, restated at the new seam.

    The three-way split has to stay three: absent → `—`, genuine `0.0` →
    `0.000`, sub-resolution → widened. A fix that made "renders as zero" the
    whole rule would collapse the first two together.
    """
    assert run_matrix._metric_cell({}, 5) == run_matrix._ABSENT_CELL
    assert run_matrix._metric_cell({5: 0.0}, 5) == "0.000"
    assert run_matrix._metric_cell({5: 0.00025}, 5) == "0.00025"


def test_the_wall_clock_column_is_unchanged_by_the_extraction() -> None:
    """GREEN against the pre-#198 tree — the point is that it stays green.

    #198 moved `_wall_clock_cell`'s body into the shared helper. The risk in
    any such extraction is that the caller's own rules get absorbed into it:
    here, that a defaulted `0.0` stops rendering `—`. These are D-016's own
    boundary values, re-asserted through the refactored path.
    """
    assert run_matrix._wall_clock_cell(0.0) == run_matrix._ABSENT_CELL
    assert run_matrix._wall_clock_cell(0.4) == "0.4"
    assert run_matrix._wall_clock_cell(0.5) == "0.5"
    assert run_matrix._wall_clock_cell(0.500001) == "1"
    assert run_matrix._wall_clock_cell(19.987165927886963) == "20"


def test_stdout_recall_goes_through_the_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The third render site — the one neither the issue nor #196 covered.

    `--strategy` is patched to return a sub-resolution run because a real run
    of the corpus produces recalls in the 0.08–0.75 range, which render
    correctly under both the old and the new code. The defect is only
    observable below the collapse threshold, so the threshold has to be reached
    deliberately; timing or scoring the real corpus would make this arm green
    against the unfixed tree.
    """
    monkeypatch.setattr(
        run_matrix,
        "evaluate_strategy",
        lambda *a, **k: _run(recall=0.00025, snippet=0.5, k=5),
        raising=True,
    )
    rc = run_matrix.main(["--strategy", "fixed-size", "--results-dir", str(tmp_path), "--ks", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recall@5=0.00025" in out, f"stdout collapsed a sub-resolution recall. Got:\n{out}"
    # Anchored, not a substring check: `recall@5=0.000` is a *prefix* of the
    # correct `recall@5=0.00025`, so `not in` is red against correct output and
    # red against the unfixed tree alike — an arm that cannot distinguish them.
    assert not re.search(r"recall@5=0\.000(?!\d)", out), (
        f"stdout still publishes a fabricated zero. Got:\n{out}"
    )


def test_stdout_snippet_hit_goes_through_the_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fourth render site, and its own arm for the AC3 reason.

    Separated from the recall arm above and driven by a payload where recall is
    *fine* — so a fix that routed only the recall field through the helper, the
    exact shape of the omission #196 left behind, fails here.
    """
    monkeypatch.setattr(
        run_matrix,
        "evaluate_strategy",
        lambda *a, **k: _run(recall=0.5, snippet=0.00025, k=5),
        raising=True,
    )
    rc = run_matrix.main(["--strategy", "fixed-size", "--results-dir", str(tmp_path), "--ks", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "snippet-hit@5=0.00025" in out, f"stdout collapsed a sub-resolution value. Got:\n{out}"
    assert not re.search(r"snippet-hit@5=0\.000(?!\d)", out), (
        f"stdout still publishes a fabricated zero. Got:\n{out}"
    )


def test_the_committed_summary_regenerates_unchanged_from_the_committed_jsons() -> None:
    """Byte-identity, rendered from the committed JSONs rather than re-run (AC4).

    #196 learned this the hard way: `run_matrix.py --canonical-out` re-times the
    corpus on the host, so the wall-clock column moves for reasons that have
    nothing to do with the change under test. Reading the tracked result files
    and re-rendering isolates the renderer.

    The strategy order is the one the committed table uses, not `sorted()` —
    `_render_summary` preserves input order, so a sorted glob produces a table
    with identical cells in a different order and this arm would report a
    failure that is an artefact of the test.
    """
    order = ["fixed-size", "recursive", "semantic", "late-chunking", "structure-aware"]
    by_name = {
        path.stem.split("__", 1)[1]: path for path in (_ROOT / "results").glob("canonical__*.json")
    }
    assert set(by_name) == set(order), sorted(by_name)
    runs = [
        RetrievalRun.from_json(json.loads(by_name[n].read_text(encoding="utf-8"))) for n in order
    ]
    rendered = run_matrix._render_summary(runs, runs[0].embedder_model)
    committed = (_ROOT / "results" / "summary.md").read_text(encoding="utf-8")
    assert rendered == committed, "the committed summary no longer regenerates unchanged"


def test_the_committed_summary_actually_contains_a_genuine_zero_cell() -> None:
    """Anti-vacuity for the byte-identity arm above.

    That arm proves nothing about the genuine-`0.0` path unless the committed
    table exercises it. It does: `structure-aware`'s recall@1 and three
    snippet-hit columns are real measured zeros, and a fix that widened
    everything would have moved them.
    """
    committed = (_ROOT / "results" / "summary.md").read_text(encoding="utf-8")
    zero_cells = sum(line.count(" 0.000 ") for line in committed.splitlines())
    assert zero_cells >= 1, "no genuine-zero cell in the committed summary to protect"


# ----------------------------------------------------------------------
# Population arm — discovered, not listed
# ----------------------------------------------------------------------


def _float_format_specs(tree: ast.AST) -> list[tuple[int, str]]:
    """Every fixed-point/general float format spec in an f-string, with its line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FormattedValue) or node.format_spec is None:
            continue
        # CPython always parses a present `format_spec` as a `JoinedStr`. Assert
        # rather than `continue`: skipping an unrecognised shape would turn this
        # guard's blind spot into silence, which is the failure mode the arm
        # below exists to prevent in the first place.
        spec_node = node.format_spec
        assert isinstance(spec_node, ast.JoinedStr), (
            f"unexpected format_spec node at line {node.lineno}: {type(spec_node).__name__}"
        )
        spec = "".join(
            part.value
            for part in spec_node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if spec.endswith(("f", "g", "e")):
            found.append((node.lineno, spec))
    return found


def test_every_float_render_in_run_matrix_goes_through_the_one_helper() -> None:
    """The rule over the module, rather than over the sites a fix happened to touch.

    Four sites in two spellings is how the fourth member of this class got
    missed, so the enumeration is of *every float rendering in the file* and the
    only permitted one is inside `_render_no_fabricated_zero` itself. A fifth
    column added later — in a spelling nobody anticipated, which is how the
    stdout pair survived #196 — fails here instead of shipping a fabricated
    zero.

    Docstrings and comments are invisible to this arm by construction: `ast`
    only sees `JoinedStr` nodes, so the several ``f"{0.5:.0f}"`` examples quoted
    in this module's prose do not trip it. That is deliberate — a text-level
    grep for the same rule would flag its own explanation.
    """
    source = _RUN_MATRIX_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    helper = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_render_no_fabricated_zero"
        ),
        None,
    )
    assert helper is not None, "the shared helper is gone; this arm is walking the wrong module"

    inside_helper = {lineno for lineno, _ in _float_format_specs(helper)}
    assert inside_helper, "the helper renders no float — the anti-vacuity premise is broken"

    offenders = [
        (lineno, spec) for lineno, spec in _float_format_specs(tree) if lineno not in inside_helper
    ]
    assert offenders == [], (
        f"scripts/run_matrix.py renders a float outside `_render_no_fabricated_zero` "
        f"at {offenders}. Every measured value published by this module must go "
        f"through it, or a sub-resolution measurement renders as a fabricated "
        f"zero (#196, #198)."
    )
