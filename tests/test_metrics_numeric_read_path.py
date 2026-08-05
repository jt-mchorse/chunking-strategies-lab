"""Value-axis guards for the numeric fields `from_json` never validated (#147).

`RetrievalRun.from_json` documents a loud failure mode on both the key
and the value axis, and `_validate_metric_map` delivered that for
`recall_at_k` / `snippet_hit_at_k`. The three *other* numeric fields on
the same read path — `wall_clock_ms` (a measured latency, D-009),
`n_queries`, and `n_chunks_total` — carried no guard at all.

That gap mattered because these fields are not inert: every one of them
renders into the tracked `results/summary.md`, the canonical aggregate
the README links. `json.loads` accepts the bare `NaN` / `Infinity`
literals by default, so a non-finite latency is reachable from a plain
hand-edited or externally-generated result file with no Python in the
loop.

The tests below are deliberately anchored to **the corruption**, not to
the exception type: `test_*_would_render_*` constructs the dataclass
directly (bypassing the loader) and asserts what `_render_summary`
actually publishes, so the guard tests can never be widened into
vacuity by someone later relaxing the raise. If a future change catches
a broader exception or drops a check, the render assertions still
describe exactly what comes back.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.metrics import RetrievalRun  # noqa: E402
from scripts.run_matrix import _render_summary  # noqa: E402


def _payload(**overrides: object) -> dict:
    """A minimal well-formed `to_json()` payload, with fields overridden."""
    base = RetrievalRun(
        strategy_name="fixed-size",
        embedder_model="HashEmbedder",
        dataset_version="v0",
        n_queries=2,
        n_chunks_total=42,
        recall_at_k={1: 0.5, 3: 1.0},
        snippet_hit_at_k={1: 0.0, 3: 0.5},
        per_query=(),
        wall_clock_ms=19.9,
    ).to_json()
    base.update(overrides)
    return base


def _wall_clock_cell(run: RetrievalRun) -> str:
    """The wall-clock column of the rendered summary's single data row."""
    row = _render_summary([run], run.embedder_model).splitlines()[-1]
    return row.split("|")[-2].strip()


def _n_chunks_cell(run: RetrievalRun) -> str:
    row = _render_summary([run], run.embedder_model).splitlines()[-1]
    return row.split("|")[2].strip()


# ---------------------------------------------------------------------------
# What the unguarded read path used to publish. These lock the *harm*, so the
# guard tests below can't quietly become assertions about nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "cell"),
    [
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        # bool subclasses int, so an `isinstance(v, (int, float))` check alone
        # admits `True` — and it renders as a *fabricated* 1 ms measurement.
        (True, "1"),
        (-5.0, "-5"),
    ],
)
def test_corrupt_wall_clock_would_render_into_the_summary(bad: object, cell: str) -> None:
    """Direct construction bypasses the loader, so this documents exactly what
    `results/summary.md` showed before #147 closed the read path."""
    run = RetrievalRun.from_json(_payload())
    corrupt = RetrievalRun(**{**run.__dict__, "wall_clock_ms": bad})
    assert _wall_clock_cell(corrupt) == cell


def test_string_wall_clock_would_crash_the_renderer() -> None:
    """A string latency never reaches a field-named loader error — it reaches
    `f"{value:.0f}"` and raises a raw formatting error at an unrelated site."""
    run = RetrievalRun.from_json(_payload())
    corrupt = RetrievalRun(**{**run.__dict__, "wall_clock_ms": "19.9"})
    with pytest.raises(ValueError, match="Unknown format code"):
        _render_summary([corrupt], corrupt.embedder_model)


def test_corrupt_n_chunks_total_would_render_into_the_summary() -> None:
    run = RetrievalRun.from_json(_payload())
    corrupt = RetrievalRun(**{**run.__dict__, "n_chunks_total": "many"})
    assert _n_chunks_cell(corrupt) == "many"


def test_corrupt_n_queries_would_render_into_the_summary_header() -> None:
    run = RetrievalRun.from_json(_payload())
    corrupt = RetrievalRun(**{**run.__dict__, "n_queries": True})
    header = _render_summary([corrupt], corrupt.embedder_model).splitlines()[2]
    assert "_n_queries_: True" in header


# ---------------------------------------------------------------------------
# The guards: none of the above can be loaded any more.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_from_json_rejects_non_finite_wall_clock(bad: float) -> None:
    with pytest.raises(ValueError, match="wall_clock_ms must be finite"):
        RetrievalRun.from_json(_payload(wall_clock_ms=bad))


@pytest.mark.parametrize("bad", [True, False, "19.9", None, [], {}])
def test_from_json_rejects_non_numeric_wall_clock(bad: object) -> None:
    with pytest.raises(ValueError, match="wall_clock_ms must be a number"):
        RetrievalRun.from_json(_payload(wall_clock_ms=bad))


@pytest.mark.parametrize("bad", [-0.001, -5.0, -1])
def test_from_json_rejects_negative_wall_clock(bad: float) -> None:
    with pytest.raises(ValueError, match="wall_clock_ms must be >= 0"):
        RetrievalRun.from_json(_payload(wall_clock_ms=bad))


@pytest.mark.parametrize("field", ["n_queries", "n_chunks_total"])
@pytest.mark.parametrize("bad", [True, False, "many", 4.0, None, [], {}])
def test_from_json_rejects_non_int_count(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be an int"):
        RetrievalRun.from_json(_payload(**{field: bad}))


@pytest.mark.parametrize("field", ["n_queries", "n_chunks_total"])
def test_from_json_rejects_negative_count(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be >= 0"):
        RetrievalRun.from_json(_payload(**{field: -1}))


# A NaN reaches the loader from a plain JSON file: `json.loads` accepts the
# bare `NaN` / `Infinity` literals by default, so no Python is needed in the
# loop to produce one. This pins the reachability the issue claims.
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_bare_json_non_finite_literal_is_reachable_and_rejected(literal: str) -> None:
    text = json.dumps(_payload()).replace('"wall_clock_ms": 19.9', f'"wall_clock_ms": {literal}')
    parsed = json.loads(text)
    assert not math.isfinite(parsed["wall_clock_ms"])
    with pytest.raises(ValueError, match="wall_clock_ms must be finite"):
        RetrievalRun.from_json(parsed)


# ---------------------------------------------------------------------------
# Nothing legitimate regressed.
# ---------------------------------------------------------------------------


def test_absent_wall_clock_still_defaults_to_zero() -> None:
    """D-009's backward-compat default survives the guard: pre-D-009 result
    files have no `wall_clock_ms` key at all and must still load."""
    payload = _payload()
    del payload["wall_clock_ms"]
    assert RetrievalRun.from_json(payload).wall_clock_ms == 0.0


@pytest.mark.parametrize("good", [0, 0.0, 19.9, 1, 10**6])
def test_from_json_accepts_valid_wall_clock(good: object) -> None:
    assert RetrievalRun.from_json(_payload(wall_clock_ms=good)).wall_clock_ms == good


def test_from_json_accepts_zero_counts() -> None:
    run = RetrievalRun.from_json(_payload(n_queries=0, n_chunks_total=0))
    assert (run.n_queries, run.n_chunks_total) == (0, 0)


def test_committed_canonical_fixtures_still_round_trip() -> None:
    """The guard must not reject the artifacts actually in the repo."""
    canonical = sorted((_REPO_ROOT / "results").glob("canonical__*.json"))
    assert canonical, "expected committed canonical__*.json files under results/"
    for path in canonical:
        raw = json.loads(path.read_text(encoding="utf-8"))
        run = RetrievalRun.from_json(raw)
        assert json.dumps(run.to_json(), sort_keys=True) == json.dumps(raw, sort_keys=True)
        assert run.wall_clock_ms >= 0.0
        assert isinstance(run.n_chunks_total, int)
