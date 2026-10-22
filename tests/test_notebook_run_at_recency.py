"""The notebook keeps the most recent run per strategy, by recorded time (#303).

#78 ranked `canonical` below every timestamped stamp, so a fresh scratch run
beat the committed fixtures. That over-corrected. Measured on `main` a3de41e:
a default `run_matrix.py` run, then a newer `--canonical-out --ks 1,10
--dataset-version v1-refreshed` run into the same directory, and the load cell
printed all five rows from the older scratch run, with no MIXED warning. The
filename cannot carry the answer, because a `canonical__` file holds no time.
And the stamp was local time, so across a DST fall-back the newer run got the
smaller stamp.

These tests drive `run_matrix.main` end to end and then run the load cell's
own functions, compiled out of `_build_notebook._LOAD_CELL`, so they test what
ships in `comparison.ipynb` and not a mirror of it.
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from scripts import run_matrix

_REPO_ROOT = Path(__file__).resolve().parents[1]

# US fall-back, 2026-11-01: 02:00 EDT becomes 01:00 EST at 06:00Z.
_BEFORE_FALL_BACK = datetime(2026, 11, 1, 5, 50, tzinfo=UTC).timestamp()  # 01:50 EDT
_AFTER_FALL_BACK = datetime(2026, 11, 1, 6, 10, tzinfo=UTC).timestamp()  # 01:10 EST


def _loader() -> Callable[[Path], list[dict[str, Any]]]:
    """`_load_latest_per_strategy` as the load cell defines it, with its helpers.

    Read out of the build script's source rather than imported: the script
    imports `nbformat` and the cell imports `matplotlib` (the `[notebook]`
    extra), and CI installs only `[dev]`, so an import would skip every test
    here in CI. The loader itself needs only `json` and `pathlib`.
    """
    script = _REPO_ROOT / "notebooks" / "_build_notebook.py"
    (call,) = [
        n.value
        for n in ast.parse(script.read_text(encoding="utf-8")).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_LOAD_CELL" for t in n.targets)
    ]
    assert isinstance(call, ast.Call), "_LOAD_CELL is no longer a dedent(...) literal"
    assert isinstance(call.func, ast.Name), "_LOAD_CELL is no longer a dedent(...) literal"
    assert call.func.id == "dedent", "_LOAD_CELL is no longer a dedent(...) literal"
    tree = ast.parse(dedent(ast.literal_eval(call.args[0])))
    imports = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom)) and "matplotlib" not in ast.unparse(n)
    ]
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns: dict[str, Any] = {}
    module = ast.Module(body=[*imports, *fns], type_ignores=[])
    exec(compile(module, "<load-cell>", "exec"), ns)  # noqa: S102
    return ns["_load_latest_per_strategy"]


def _run(results_dir: Path, *extra: str) -> None:
    assert run_matrix.main(["--results-dir", str(results_dir), *extra]) == 0


def _write(path: Path, **payload: Any) -> None:
    path.write_text(json.dumps({"strategy_name": "fixed-size", **payload}), encoding="utf-8")


@pytest.fixture
def new_york(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_a_canonical_refresh_after_a_scratch_run_wins(tmp_path: Path) -> None:
    # The issue's repro: the README's "fills results/" step, then the
    # `--canonical-out` refresh `capture_demo.sh` tells the operator to run.
    _run(tmp_path)
    _run(tmp_path, "--canonical-out", "--dataset-version", "v1-refreshed", "--ks", "1,10")
    runs = _loader()(tmp_path)
    assert len(runs) == 5
    for r in runs:
        assert r["dataset_version"] == "v1-refreshed", (
            f"{r['strategy_name']}: the notebook loaded the older scratch run over "
            "the newer --canonical-out refresh (#303)"
        )
        assert set(r["recall_at_k"]) == {"1", "10"}


def test_a_scratch_run_after_canonical_still_wins(tmp_path: Path) -> None:
    # Control, the #78 direction: a fresh run beats the fixtures it sits beside.
    _run(tmp_path, "--canonical-out")
    _run(tmp_path, "--dataset-version", "v1-scratch", "--ks", "2")
    runs = _loader()(tmp_path)
    assert len(runs) == 5
    assert {r["dataset_version"] for r in runs} == {"v1-scratch"}


def test_the_later_run_wins_across_a_dst_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_york: None
) -> None:
    # In local time these two runs are stamped 015000 then 011000: the newer
    # run sorted first, and the notebook picked the earlier one.
    clock = iter([_BEFORE_FALL_BACK, _AFTER_FALL_BACK])
    monkeypatch.setattr(run_matrix, "_run_clock", lambda: next(clock), raising=False)
    _run(tmp_path, "--dataset-version", "before")
    _run(tmp_path, "--dataset-version", "after")
    stamps = {p.name.split("__", 1)[0] for p in tmp_path.glob("*__*.json")}
    assert stamps == {"20261101T055000", "20261101T061000"}, (
        f"run stamps {sorted(stamps)} are not the runs' UTC start times (#303)"
    )
    assert {r["dataset_version"] for r in _loader()(tmp_path)} == {"after"}


def test_run_matrix_records_a_utc_run_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_matrix, "_run_clock", lambda: _AFTER_FALL_BACK, raising=False)
    _run(tmp_path, "--strategy", "fixed-size")
    (path,) = tmp_path.glob("*__fixed-size.json")
    run_at = json.loads(path.read_text(encoding="utf-8")).get("run_at")
    assert run_at == "2026-11-01T06:10:00.000000+00:00", (
        f"run_matrix wrote run_at={run_at!r}; the loader needs the run's UTC start (#303)"
    )


def test_a_recorded_run_at_outranks_the_filename_stamp(tmp_path: Path) -> None:
    # Two files whose stamps disagree with their recorded times: the time wins.
    _write(
        tmp_path / "20261101T015000__fixed-size.json",
        run_at="2026-11-01T05:50:00.000000+00:00",
        dataset_version="older",
    )
    _write(
        tmp_path / "20261101T011000__fixed-size.json",
        run_at="2026-11-01T06:10:00.000000+00:00",
        dataset_version="newer",
    )
    (run,) = _loader()(tmp_path)
    assert run["dataset_version"] == "newer"


def test_files_without_run_at_keep_the_stamp_order(tmp_path: Path) -> None:
    # Control: the committed fixtures and pre-#303 scratch runs carry no
    # `run_at`, and among them a timestamped run still beats `canonical` (#78).
    _write(tmp_path / "canonical__fixed-size.json", dataset_version="canonical")
    _write(tmp_path / "20260627T120000__fixed-size.json", dataset_version="old")
    _write(tmp_path / "20260627T130000__fixed-size.json", dataset_version="newer")
    (run,) = _loader()(tmp_path)
    assert run["dataset_version"] == "newer"


def test_the_committed_fixtures_still_load() -> None:
    # Control: the shipped `results/` set, which has no `run_at`, renders as before.
    runs = _loader()(_REPO_ROOT / "results")
    assert [r["strategy_name"] for r in runs] == [
        "fixed-size",
        "recursive",
        "semantic",
        "late-chunking",
        "structure-aware",
    ]
