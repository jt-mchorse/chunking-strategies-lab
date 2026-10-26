"""The round-trip locks must pass on the files the documented refresh writes (#311).

`README.md`, `REGEN_HINT` in `test_canonical_fixture_freshness.py` and
`REGEN_COMMAND` in `test_summary_snapshot.py` all tell the operator to refresh
the committed fixtures with `python scripts/run_matrix.py --canonical-out`.
Since #303 that command writes a `run_at` key next to the `to_json()` fields.
Four exact round-trip locks compared `from_json(payload).to_json()` with the
whole payload, so the next legitimate refresh turned all four red. They now
share `tests._canonical_round_trip.assert_round_trips`.

The four locks read the committed `results/` files, which have not been
regenerated since #303, so they cannot catch this. This module runs the
documented command into a temp dir and applies the same helper to its output.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chunking_lab.metrics import RetrievalRun  # noqa: E402
from tests._canonical_round_trip import (  # noqa: E402
    FILE_PROVENANCE_KEYS,
    assert_round_trips,
)

_STRATEGIES = ("fixed-size", "late-chunking", "recursive", "semantic", "structure-aware")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """`{strategy: payload}` from the README's refresh command, aimed at a temp dir."""
    out_dir = tmp_path_factory.mktemp("canonical_regen")
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_matrix.py"),
            "--canonical-out",
            "--results-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"run_matrix exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    produced = {
        p.stem.split("__", 1)[1]: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out_dir.glob("canonical__*.json"))
    }
    assert sorted(produced) == list(_STRATEGIES), sorted(produced)
    return produced


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_a_freshly_regenerated_fixture_round_trips(
    regenerated: dict[str, dict[str, Any]], strategy: str
) -> None:
    """The regression: what the four locks assert, on what the refresh writes."""
    payload = regenerated[strategy]
    # The case #311 is about. If `run_at` stops being written this arm stops
    # exercising it, so say so rather than passing on a file without it.
    assert "run_at" in payload, sorted(payload)
    assert_round_trips(payload, f"regenerated canonical__{strategy}.json")


def test_the_dropped_keys_are_exactly_what_the_writer_adds(
    regenerated: dict[str, dict[str, Any]],
) -> None:
    """Measured from the written file, not assumed.

    Whatever `run_matrix.py` writes beyond `to_json()` must be exactly
    `FILE_PROVENANCE_KEYS`. A missing name turns the four locks red on the next
    refresh again. An extra name (say `wall_clock_ms`) would stop them noticing
    a `from_json` that lost that field.
    """
    for strategy, payload in regenerated.items():
        emitted = set(RetrievalRun.from_json(payload).to_json())
        assert set(payload) - emitted == FILE_PROVENANCE_KEYS, strategy
        assert not FILE_PROVENANCE_KEYS & emitted, strategy


# ---------------------------------------------------------------------------
# Controls: dropping `run_at` must not make the lock accept anything else.
# ---------------------------------------------------------------------------


def test_a_real_metric_change_still_fails(regenerated: dict[str, dict[str, Any]]) -> None:
    """A rate that does not match the file's own evidence is still refused."""
    payload = copy.deepcopy(regenerated["fixed-size"])
    k = sorted(payload["recall_at_k"], key=int)[0]
    original = payload["recall_at_k"][k]
    payload["recall_at_k"][k] = 0.0 if original != 0.0 else 1.0
    with pytest.raises((AssertionError, ValueError)):
        assert_round_trips(payload, "mutated")


def test_a_key_from_json_drops_that_is_not_provenance_still_fails(
    regenerated: dict[str, dict[str, Any]],
) -> None:
    """The comparison is not narrowed to the keys `to_json()` happens to emit.

    A field `from_json` silently ignores is what this lock exists to catch.
    Only the named provenance keys are excused.
    """
    payload = copy.deepcopy(regenerated["fixed-size"])
    payload["mrr_at_k"] = {"1": 0.5}
    with pytest.raises(AssertionError, match="does not reproduce the file"):
        assert_round_trips(payload, "extra key")
