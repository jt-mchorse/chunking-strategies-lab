"""Two `run_matrix` runs in the same second keep separate scratch sets (#289).

The timestamp prefix has one-second resolution. Measured on `main` 6c40be3, a
full run followed by `--strategy semantic --ks 2` inside one second replaced
`<stamp>__semantic.json`. The full run's `<stamp>__summary.md` still showed
recall@1/3/5 for semantic, while the JSON beside it held
`recall_at_k = {'2': 0.5}`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.run_matrix as run_matrix  # noqa: E402

FROZEN = "20261020T002952"


@pytest.fixture
def one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_matrix.time, "strftime", lambda fmt, *a: FROZEN)


def _stamps(d: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in d.glob("*__*"):
        stamp, rest = p.name.split("__", 1)
        out.setdefault(stamp, set()).add(rest)
    return out


def test_a_second_run_in_the_same_second_overwrites_nothing(
    tmp_path: Path, one_second: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_matrix.main(["--results-dir", str(tmp_path)]) == 0
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert (
        run_matrix.main(["--results-dir", str(tmp_path), "--strategy", "semantic", "--ks", "2"])
        == 0
    )
    for name, data in first.items():
        assert (tmp_path / name).read_bytes() == data, f"{name} was overwritten"
    stamps = _stamps(tmp_path)
    assert set(stamps) == {FROZEN, f"{FROZEN}-02"}
    assert stamps[f"{FROZEN}-02"] == {"semantic.json"}
    # The full run's summary still describes the JSON beside it.
    semantic = json.loads((tmp_path / f"{FROZEN}__semantic.json").read_text())
    assert set(semantic["recall_at_k"]) == {"1", "3", "5"}


def test_a_third_run_takes_the_next_suffix(tmp_path: Path, one_second: None) -> None:
    for _ in range(3):
        assert run_matrix.main(["--results-dir", str(tmp_path), "--strategy", "semantic"]) == 0
    assert set(_stamps(tmp_path)) == {FROZEN, f"{FROZEN}-02", f"{FROZEN}-03"}


def test_the_suffixed_stamp_ranks_between_its_second_and_the_next() -> None:
    # The notebook keeps the latest run per strategy by plain string order of
    # the stamp (`_stamp_rank`), so the newer same-second run must sort after
    # the bare stamp and before the next second's.
    assert FROZEN < f"{FROZEN}-02" < f"{FROZEN}-03" < "20261020T002953"


def test_canonical_out_still_writes_its_fixed_names(tmp_path: Path, one_second: None) -> None:
    # Control: the canonical fixtures are meant to be overwritten in place.
    for _ in range(2):
        assert run_matrix.main(["--results-dir", str(tmp_path), "--canonical-out"]) == 0
    assert set(_stamps(tmp_path)) == {"canonical"}
    assert (tmp_path / "summary.md").exists()


def test_two_runs_started_together_get_distinct_stamps(tmp_path: Path, one_second: None) -> None:
    # #294: #290 checked for existing files, but a run writes its files only
    # after evaluating, so two runs started together both saw an empty
    # directory and shared one stamp (measured: the `--ks 2` run's JSON was
    # replaced by the full run's `{'1','3','5'}`). The stamp is now reserved
    # atomically when it is picked.
    import threading

    barrier = threading.Barrier(2)
    real_free = run_matrix._free_stamp

    def synced_free(results_dir: Path, base: str) -> str:
        barrier.wait()  # both runs reach the pick together
        return real_free(results_dir, base)

    rcs: list[int] = []
    argvs = [
        ["--results-dir", str(tmp_path), "--strategy", "semantic"],
        ["--results-dir", str(tmp_path), "--strategy", "semantic", "--ks", "2"],
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_matrix, "_free_stamp", synced_free)
        threads = [
            threading.Thread(target=lambda a=a: rcs.append(run_matrix.main(a))) for a in argvs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert rcs == [0, 0]
    stamps = _stamps(tmp_path)
    assert set(stamps) == {FROZEN, f"{FROZEN}-02"}
    ks = sorted(
        tuple(sorted(json.loads((tmp_path / f"{s}__semantic.json").read_text())["recall_at_k"]))
        for s in stamps
    )
    assert ks == [("1", "3", "5"), ("2",)]
    assert not list(tmp_path.glob("*__reserved"))  # released at the end
