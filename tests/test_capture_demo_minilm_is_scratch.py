"""No suggested MiniLM run writes over the committed canonical fixtures (#313).

`results/canonical__*.json` and `results/summary.md` are the HashEmbedder runs
`test_canonical_fixture_freshness` and `test_summary_snapshot` re-derive, so a
real-embedder `--canonical-out` refresh turns 11 of those locks red by
construction. The README moved its MiniLM command to scratch mode, but
`capture_demo.sh` still printed `--embedder minilm --canonical-out   # refreshes
results/`. Every `run_matrix.py` command either file suggests is checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_COMMAND = re.compile(r"python scripts/run_matrix\.py[^\n'\"]*")


def _commands(path: Path) -> list[str]:
    return _COMMAND.findall(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel", ["scripts/capture_demo.sh", "README.md"])
def test_no_suggested_command_refreshes_canonical_fixtures_with_minilm(rel: str) -> None:
    commands = _commands(ROOT / rel)
    assert commands, f"{rel} suggests no run_matrix.py command; the pattern is stale"
    bad = [c for c in commands if "--embedder minilm" in c and "--canonical-out" in c]
    assert bad == [], f"{rel} suggests a MiniLM canonical refresh: {bad}"


def test_the_demo_still_points_at_a_minilm_run() -> None:
    # Control: the fix moves the MiniLM suggestion to scratch, it doesn't drop it.
    assert any("--embedder minilm" in c for c in _commands(ROOT / "scripts" / "capture_demo.sh"))


def test_the_pattern_sees_the_shape_it_guards_against() -> None:
    line = "printf '  python scripts/run_matrix.py --embedder minilm --canonical-out   # refreshes results/\\n'"
    (found,) = _COMMAND.findall(line)
    assert "--embedder minilm" in found
    assert "--canonical-out" in found
