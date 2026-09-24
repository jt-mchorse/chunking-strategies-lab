"""A strictly positive elapsed time is never published as ``0`` (#196).

``scripts/run_matrix.py`` rendered the wall-clock column as a bare
``{r.wall_clock_ms:.0f}``, so any elapsed time that rounds to zero published
``0`` — an *impossible* elapsed time, in the tracked ``results/summary.md``
that the README's comparison narrative reads.

``_validate_wall_clock``'s own docstring already framed the harm class: a
``wall_clock_ms`` is a *measured* number (D-009), and "a negative value renders
impossible elapsed time". Zero is impossible elapsed time too — nothing takes
zero milliseconds — so the cell was never a measurement, always a rendering
artefact. And here it is the *flattering* one: a strategy that takes zero
milliseconds wins any "which is fastest" read, and wall-clock is one of three
columns the README compares on. Third member of a class that also paid in
``embedding-model-shootout#149`` and ``vector-search-at-scale#148``.

Two things these tests pin that a formatter-only check would not:

* **The rendered line, not the formatter** (AC4). In ``vsas#148`` the
  formatter-level arms stayed green against a call-site revert, because the
  formatter was already correct. So the summary arms below go through
  ``_render_summary`` and read the actual GFM row, and the stdout arm runs
  ``main`` and reads captured stdout.
* **Every render site, discovered rather than listed.** There were *two*:
  the summary cell and the per-strategy stdout line, whose own comment argues
  that "stdout is a publication surface like the summary table". The AST arm
  below finds the population instead of trusting this docstring's count.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from chunking_lab.metrics import RetrievalRun, evaluate_strategy

_ROOT = Path(__file__).resolve().parents[1]
_RUN_MATRIX_PATH = _ROOT / "scripts" / "run_matrix.py"


def _load_run_matrix():
    spec = importlib.util.spec_from_file_location(
        "run_matrix_for_wall_clock_tests", _RUN_MATRIX_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_matrix = _load_run_matrix()


def _run(wall_clock_ms: float, name: str = "fixed-size") -> RetrievalRun:
    """A minimal `RetrievalRun` built through the validating read path.

    `from_json` rather than the constructor, so every value asserted on below is
    one `_validate_wall_clock` actually accepts — which is the reachability the
    issue rests on: `scripts/run_matrix.py --results-dir <dir>` walks
    hand-edited or externally generated result files.
    """
    return RetrievalRun.from_json(
        {
            "strategy_name": name,
            "embedder_model": "hash",
            "dataset_version": "v0",
            "n_queries": 12,
            "n_chunks_total": 29,
            "recall_at_k": {"5": 0.5},
            "snippet_hit_at_k": {"5": 0.25},
            "per_query": [],
            "wall_clock_ms": wall_clock_ms,
        }
    )


def _wall_clock_column(rendered: str, strategy: str = "fixed-size") -> str:
    """Pull the last cell out of the GFM row for `strategy`.

    Reads the *table*, not a return value, so a fix that corrects the formatter
    and leaves the call site alone cannot pass.
    """
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("| ") and line[2:].startswith(strategy + " ")
    ]
    assert len(rows) == 1, f"expected exactly one row for {strategy!r}, got {rows}"
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    return cells[-1]


# Read off a printed run of `_wall_clock_cell`, not retyped from the issue: the
# boundary is NOT 0.5. `f"{0.5:.0f}"` is `'0'`, because Python rounds half to
# even — so `if ms < 0.5` misses exactly 0.5, and 0.500001 is the first value
# the old renderer got right.
_BOUNDARY = (
    (0.0, "—"),
    (1e-9, "1e-09"),
    (0.001, "0.001"),
    (0.04, "0.04"),
    (0.4, "0.4"),
    (0.49999, "0.5"),
    (0.5, "0.5"),
    (0.500001, "1"),
    (1.0, "1"),
    (19.987165927886963, "20"),
    (85.1255829911679, "85"),
)


@pytest.mark.parametrize(("ms", "expected"), _BOUNDARY, ids=[repr(v) for v, _ in _BOUNDARY])
def test_the_rendered_summary_cell_across_the_boundary(ms: float, expected: str) -> None:
    """The published cell, for every value either side of the collapse.

    Asserted through `_render_summary` rather than against `_wall_clock_cell`,
    so this arm also pins that the summary's call site is wired to the helper.
    """
    cell = _wall_clock_column(run_matrix._render_summary([_run(ms)], "HashEmbedder"))
    assert cell == expected, (
        f"wall_clock_ms={ms!r} publishes {cell!r} in the summary table, expected {expected!r}."
    )


@pytest.mark.parametrize("ms", [1e-9, 0.001, 0.04, 0.4, 0.49999, 0.5])
def test_no_strictly_positive_elapsed_time_publishes_zero(ms: float) -> None:
    """AC1, stated as the property rather than as a table of values.

    The table above pins exact strings and would need editing if the precision
    changed; this arm survives that and is the claim the issue actually makes.
    Zero is impossible elapsed time, so a positive measurement rendering as zero
    is a fabricated benchmark number in the sense handoff §10 cares about.
    """
    cell = _wall_clock_column(run_matrix._render_summary([_run(ms)], "HashEmbedder"))
    assert float(cell) != 0.0, (
        f"a strictly positive wall_clock_ms={ms!r} published {cell!r}, which "
        f"reads as zero elapsed time — and zero is the *fastest* value in this "
        f"column, so the artefact flatters the strategy."
    )


def test_a_defaulted_zero_is_published_as_not_measured() -> None:
    """AC3, decided rather than left to fall out.

    D-009 made ``wall_clock_ms: float = 0.0`` the backward-compat default so
    pre-D-009 JSONs still load, which makes ``0.0`` this field's "not measured"
    sentinel and not a measurement. `_metric_cell` in the same module exists to
    draw exactly that distinction and #160 locked that an unmeasured cell is
    never published as a number, so ``0.0`` gets the same `_ABSENT_CELL`.

    The alternative — keep printing ``0`` because "zero is not misleading for a
    genuine zero" — fails on the fact that there *is* no genuine zero: the only
    way this field holds 0.0 is the default.
    """
    cell = _wall_clock_column(run_matrix._render_summary([_run(0.0)], "HashEmbedder"))
    assert cell == run_matrix._ABSENT_CELL, (
        f"an unmeasured wall_clock_ms=0.0 published {cell!r}; it should render "
        f"{run_matrix._ABSENT_CELL!r}, the same not-measured spelling "
        f"`_metric_cell` uses (#160)."
    )
    assert cell != "0", "0.0 is D-009's default, not a measurement."


def test_ordinary_values_are_byte_identical_to_the_old_renderer() -> None:
    """The control on the fix: nothing that already rendered correctly moved.

    Green against the pre-#196 tree by construction — that is the point. It is
    what separates this fix from the neighbour that widens the whole column to
    ``.1f``, which is a smaller diff and churns every committed cell (20 -> 20.0)
    and therefore `results/summary.md` and `docs/benchmarks.md` with it.

    The five values are the committed canonical runs, pasted from a print of the
    JSONs rather than retyped.
    """
    committed = {
        "fixed-size": (19.987165927886963, "20"),
        "recursive": (22.148292046040297, "22"),
        "semantic": (85.1255829911679, "85"),
        "late-chunking": (23.562125163152814, "24"),
        "structure-aware": (20.719958934932947, "21"),
    }
    runs = [_run(ms, name) for name, (ms, _) in committed.items()]
    rendered = run_matrix._render_summary(runs, "HashEmbedder")
    got = {name: _wall_clock_column(rendered, name) for name in committed}
    expected = {name: cell for name, (_, cell) in committed.items()}
    assert got == expected, (
        f"the committed canonical cells moved: {got} != {expected}. "
        f"results/summary.md is tracked, so any change here churns it."
    )


def test_the_committed_summary_regenerates_unchanged_from_the_committed_jsons() -> None:
    """AC2, against the real artifact rather than a fixture.

    Renders from the *committed* JSONs, so the only thing that can differ is the
    renderer. Regenerating by running the script instead would re-time the
    corpus on the test host and move the numbers for a reason that has nothing
    to do with this change — which is exactly what it did when verified by hand
    (20/22/85/24/21 committed vs 15/15/57/16/14 re-timed).
    """
    results = _ROOT / "results"
    order = ["fixed-size", "recursive", "semantic", "late-chunking", "structure-aware"]
    runs = [
        RetrievalRun.from_json(
            json.loads((results / f"canonical__{name}.json").read_text(encoding="utf-8"))
        )
        for name in order
    ]
    rendered = run_matrix._render_summary(runs, "HashEmbedder")
    committed = (results / "summary.md").read_text(encoding="utf-8")
    assert rendered == committed, (
        "results/summary.md does not regenerate byte-identically from the "
        "committed JSONs. Either the renderer changed an ordinary cell, or the "
        "artifact is stale."
    )


def test_stdout_wall_clock_goes_through_the_same_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The per-strategy stdout line is the second render site (AC1, other half).

    Its own neighbouring comment makes the argument: "stdout is a publication
    surface like the summary table, and a silent `0.000` here would be the same
    fabricated measurement." That reasoning is about the *surface*, so it covers
    this line's `wall_clock=…` as much as the recall cells it was written for —
    and the line was still a bare `.0f`.

    `evaluate_strategy` is patched to return a 0.4 ms run because a real run of
    the corpus takes ~15 ms on any host, which renders correctly under *both*
    the old and the new code. Timing the real corpus would make this arm green
    against the unfixed tree — the defect is only observable below the collapse
    threshold, so the threshold has to be reached deliberately.
    """
    monkeypatch.setattr(run_matrix, "evaluate_strategy", lambda *a, **k: _run(0.4), raising=True)
    rc = run_matrix.main(["--strategy", "fixed-size", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wall_clock=0.4ms" in out, (
        f"stdout published a collapsed wall-clock for a 0.4 ms run. Got:\n{out}"
    )
    assert "wall_clock=0ms" not in out, (
        f"stdout still renders a positive elapsed time as 0 ms. Got:\n{out}"
    )


def test_every_wall_clock_render_site_goes_through_the_helper() -> None:
    """Discover the render sites; don't trust a docstring's count of them.

    Walks `run_matrix.py` for every read of `.wall_clock_ms` and requires each
    to be either the helper's own parameter or an argument to
    `_wall_clock_cell`. Two sites existed when #196 was written, and the second
    one — stdout — was not in the issue. A third added later, in a spelling
    nobody anticipated, fails here rather than shipping a collapsed cell.

    This is the population arm: the *protected operation* is "render this
    field", and the sites that perform it are what has to be enumerated, not the
    ones a previous fix happened to touch.
    """
    tree = ast.parse(_RUN_MATRIX_PATH.read_text(encoding="utf-8"))

    wrapped: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_wall_clock_cell"
        ):
            for arg in node.args:
                wrapped.add(id(arg))

    unwrapped: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "wall_clock_ms"
            and id(node) not in wrapped
        ):
            unwrapped.append(node.lineno)

    assert not unwrapped, (
        f"`.wall_clock_ms` is read at scripts/run_matrix.py lines {unwrapped} "
        f"without going through `_wall_clock_cell`. Every publication of this "
        f"field must, or a value below the rounding threshold renders as a "
        f"fabricated 0 (#196)."
    )


def test_the_helper_is_reachable_for_a_real_measured_run() -> None:
    """Anti-vacuity on the fixture: a genuinely measured run is never 0.0.

    Everything above feeds `_render_summary` hand-built runs. This arm measures
    one for real, so the tests are not entirely a closed loop over values that
    `from_json` accepts but `evaluate_strategy` would never produce — and it
    pins the premise the AC3 decision rests on: the measured path does not emit
    0.0, so 0.0 really is only ever the default.
    """
    from chunking_lab.corpus import load_corpus
    from chunking_lab.embedder import HashEmbedder
    from chunking_lab.queries import load_queries
    from chunking_lab.strategies.fixed import FixedSizeStrategy

    run = evaluate_strategy(
        FixedSizeStrategy(),
        load_corpus(),
        load_queries(),
        HashEmbedder(),
        ks=[5],
    )
    assert run.wall_clock_ms > 0.0, (
        "a measured run reported wall_clock_ms == 0.0, which would make the "
        "not-measured sentinel ambiguous and invalidate the AC3 decision."
    )
    assert (
        _wall_clock_column(run_matrix._render_summary([run], "fixed-size"))
        != run_matrix._ABSENT_CELL
    )
