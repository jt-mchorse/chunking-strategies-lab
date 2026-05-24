"""Tests for `scripts/run_matrix.py --strategy <name>`.

The filter is a dev-iteration knob: pinpoint one strategy without
regenerating the other four (slow on the minilm embedder) or
clobbering them under `--canonical-out`. The "no summary when
filtered" rule keeps the partial-row failure mode from poisoning
the snapshot lock — these tests pin that contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing the script as a module exercises `main` without spawning a subprocess.
from scripts.run_matrix import main  # noqa: E402

_ALL_STRATEGIES = ("fixed-size", "recursive", "semantic", "late-chunking", "structure-aware")


def _stamped_jsons(results_dir: Path) -> list[Path]:
    """Per-strategy JSONs the script wrote, regardless of timestamp prefix."""
    return sorted(p for p in results_dir.iterdir() if p.suffix == ".json")


def test_strategy_filter_writes_only_chosen_strategy(tmp_path: Path) -> None:
    rc = main(["--strategy", "semantic", "--results-dir", str(tmp_path)])
    assert rc == 0
    jsons = _stamped_jsons(tmp_path)
    assert len(jsons) == 1, f"expected one JSON; got {[p.name for p in jsons]}"
    assert jsons[0].name.endswith("__semantic.json")


def test_strategy_filter_does_not_touch_existing_other_strategy_outputs(tmp_path: Path) -> None:
    """A filtered run mustn't overwrite the other four strategies' files
    that may have landed in an earlier full run."""
    # Seed canonical files for the other four strategies; the filter must leave them.
    pre_existing: dict[str, bytes] = {}
    for name in _ALL_STRATEGIES:
        if name == "semantic":
            continue
        path = tmp_path / f"canonical__{name}.json"
        path.write_bytes(b'{"sentinel": true}')
        pre_existing[name] = path.read_bytes()

    rc = main(["--strategy", "semantic", "--canonical-out", "--results-dir", str(tmp_path)])
    assert rc == 0
    for name, blob in pre_existing.items():
        path = tmp_path / f"canonical__{name}.json"
        assert path.read_bytes() == blob, f"{name} canonical file was clobbered by filtered run"
    # And the chosen strategy's canonical was written:
    semantic_path = tmp_path / "canonical__semantic.json"
    assert semantic_path.exists()
    assert semantic_path.read_bytes() != b'{"sentinel": true}'


def test_strategy_filter_does_not_write_any_summary_md(tmp_path: Path) -> None:
    """No summary when filtered — neither the canonical nor the stamped variant."""
    rc = main(["--strategy", "fixed-size", "--results-dir", str(tmp_path)])
    assert rc == 0
    summary_files = sorted(p for p in tmp_path.iterdir() if p.suffix == ".md")
    assert summary_files == [], (
        f"--strategy must not emit any markdown summary; got {[p.name for p in summary_files]}"
    )


def test_strategy_filter_with_canonical_out_also_skips_summary(tmp_path: Path) -> None:
    rc = main(["--strategy", "structure-aware", "--canonical-out", "--results-dir", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "summary.md").exists()


def test_unfiltered_run_still_writes_all_five_and_summary(tmp_path: Path) -> None:
    """Regression guard: the default (no --strategy) path keeps writing
    all five strategy JSONs and a markdown summary, exactly as before."""
    rc = main(["--results-dir", str(tmp_path)])
    assert rc == 0
    json_names = {p.name.split("__", 1)[1] for p in _stamped_jsons(tmp_path)}
    assert json_names == {f"{name}.json" for name in _ALL_STRATEGIES}
    summary_md = sorted(p for p in tmp_path.iterdir() if p.name.endswith("summary.md"))
    assert len(summary_md) == 1


def test_strategy_filter_unknown_value_argparse_rejects(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--strategy", "made-up", "--results-dir", "/tmp/should-not-touch"])
    # argparse exits 2 on choices= violation.
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
