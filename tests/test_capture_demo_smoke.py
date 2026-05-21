"""Smoke test for `scripts/capture_demo.sh` (issue #17).

The capture script is the deterministic driver for the 60-second README
demo. JT records the GIF/video while it runs; CI runs it with
`CAPTURE_PACE_SECONDS=0` to make sure the demo can't bitrot the same
way `tests/test_summary_snapshot.py` already protects the per-strategy
summary rendering in isolation.

Contract this test pins:

1. The script exits 0 on a fresh clone with no API key.
2. Each of the two surfaces actually runs (the surface header + the
   surface's distinctive output both appear).
3. The matrix step prints one line per strategy (all five — fixed-size,
   recursive, semantic, late-chunking, structure-aware) — proving the
   shared `ChunkingStrategy` Protocol still wires every implementation.
4. The summary step emits the same markdown header signature that
   `results/summary.md` carries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capture_demo.sh"

STRATEGIES = (
    "fixed-size",
    "recursive",
    "semantic",
    "late-chunking",
    "structure-aware",
)


@pytest.fixture(scope="module")
def capture_output() -> str:
    """Run the capture script once and reuse its stdout across assertions.

    `CAPTURE_PACE_SECONDS=0` removes the recording pauses so the test
    isn't gated on sleep durations.
    """
    if not SCRIPT.exists():
        pytest.fail(f"missing {SCRIPT}")
    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    env = dict(os.environ)
    env["CAPTURE_PACE_SECONDS"] = "0"
    # Ensure `python` resolves to the venv pytest is running under, since
    # capture_demo.sh shells out to `python scripts/run_matrix.py`.
    venv_bin = Path(sys.executable).parent
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"capture_demo.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


def test_surface_1_matrix_run_prints_every_strategy(capture_output: str) -> None:
    assert "1/2 · matrix run" in capture_output
    # run_matrix prints "<strategy>  n_chunks=<N>  recall@5=<N>  snippet-hit@5=<N> wall_clock=<N>ms"
    # per strategy. Lock the strategy name + the three stat keys that
    # form the per-line contract.
    for strat in STRATEGIES:
        assert strat in capture_output, f"missing {strat!r} per-strategy line"
    assert capture_output.count("recall@5=") >= len(STRATEGIES)
    assert capture_output.count("snippet-hit@5=") >= len(STRATEGIES)
    assert capture_output.count("wall_clock=") >= len(STRATEGIES)


def test_surface_2_summary_renders_committed_table_header(capture_output: str) -> None:
    """The rendered summary is the load-bearing artifact for the viewer.

    Lock the exact header row that `results/summary.md` ships (and that
    `tests/test_summary_snapshot.py` already locks elsewhere) — if the
    summary format ever drifts, this test catches it via the capture
    path too, not just via the on-disk file.
    """
    assert "2/2 · rendered summary" in capture_output
    expected_header = (
        "| strategy | n_chunks | recall@1 | recall@3 | recall@5 "
        "| snippet-hit@1 | snippet-hit@3 | snippet-hit@5 | wall-clock (ms) |"
    )
    assert expected_header in capture_output, (
        "summary markdown header drifted; test_summary_snapshot.py and this test must agree"
    )
    # Both fences of the rendered table appear — proves it isn't a
    # truncated header alone.
    for strat in STRATEGIES:
        assert f"| {strat} |" in capture_output, f"missing {strat!r} data row in summary"


def test_capture_demo_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"
