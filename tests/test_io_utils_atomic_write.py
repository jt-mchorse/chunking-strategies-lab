"""Atomicity contract for `chunking_lab.io_utils.atomic_write_text` (issue #33).

Until this PR, two production write sites in `scripts/run_matrix.py`
used `Path.write_text` without atomicity. A signal between the implicit
`open(..., "w")` truncate and `close()` flush leaves the destination
zero-length or partial. Particularly nasty for canonical-fixture writes
(`results/canonical__<strategy>.json`) because a half-written file
either fails the snapshot test loudly (obscuring the underlying crash)
or — worse — gets committed and silently changes the published numbers.

This PR routes both sites through a new public helper at
`chunking_lab.io_utils.atomic_write_text`, matching the 2026-05-26
portfolio atomic-write arc.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from chunking_lab import io_utils as io_utils_mod
from chunking_lab.io_utils import atomic_write_text

# ---------------------------------------------------------------------------
# Unit tests on the helper.
# ---------------------------------------------------------------------------


def test_atomic_write_text_happy_path(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    atomic_write_text(out, "hello\nworld\n")
    assert out.read_text(encoding="utf-8") == "hello\nworld\n"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "x.json"
    assert not out.parent.exists()
    atomic_write_text(out, "{}")
    assert out.read_text(encoding="utf-8") == "{}"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    out.write_text("STALE-CONTENT-MUST-NOT-SURVIVE", encoding="utf-8")
    atomic_write_text(out, "fresh")
    body = out.read_text(encoding="utf-8")
    assert body == "fresh"
    assert "STALE" not in body


def test_atomic_write_text_long_basename_within_name_max(tmp_path: Path) -> None:
    """A destination basename near NAME_MAX that `write_text` accepts must also
    succeed via `atomic_write_text` — the temp name `.<base>.<rand>.tmp` must not
    overflow the 255-byte filename limit (sibling of rag-production-kit#128,
    mcp-server-cookbook#96)."""
    base = "a" * 250  # 250 < 255 NAME_MAX; a plain write_text of this succeeds
    (tmp_path / base).write_text("ok", encoding="utf-8")  # proves the name is legal
    out = tmp_path / (base[:-1] + "b")  # distinct 250-byte basename
    atomic_write_text(out, "payload")
    assert out.read_text(encoding="utf-8") == "payload"


def test_atomic_write_text_replace_failure_leaves_destination_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "result.json"

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')

    assert not out.exists()


def test_atomic_write_text_replace_failure_cleans_up_tmp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "artifacts" / "delta.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')

    siblings = list(out.parent.iterdir())
    assert siblings == [], f"expected no temp leftovers in {out.parent}, got {siblings}"


def test_atomic_write_text_destination_unchanged_when_overwriting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Particularly load-bearing here: a half-written
    `canonical__<strategy>.json` overwrite that overwrote the
    committed fixture would silently change the published numbers.
    The atomic helper preserves the committed file on overwrite
    failure.
    """
    out = tmp_path / "canonical__fixed-size.json"
    out.write_text('{"committed": true}', encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(out, '{"would_have_overwritten": true}')

    assert out.read_text(encoding="utf-8") == '{"committed": true}'


# ---------------------------------------------------------------------------
# Integration: scripts/run_matrix.py routes through the helper.
# ---------------------------------------------------------------------------


def _load_script(name: str):
    """Load `scripts/<name>` as a module. Pre-registers in `sys.modules`
    before exec so dataclasses inside the script can resolve their
    `__module__` during class creation."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / name
    module_name = f"_under_test_{name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def test_run_matrix_routes_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scripts/run_matrix.py:main()` must route its per-strategy JSON
    write through atomic_write_text. Restrict to a single strategy
    so the script finishes in well under a second under the hash
    embedder.
    """
    run_matrix = _load_script("run_matrix.py")

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)

    results_dir = tmp_path / "results"
    with pytest.raises(OSError, match="simulated rename failure"):
        run_matrix.main(
            [
                "--embedder",
                "hash",
                "--results-dir",
                str(results_dir),
                "--strategy",
                "fixed-size",
            ]
        )

    # No JSON should be written under results_dir on the failed rename.
    if results_dir.exists():
        jsons = [p for p in results_dir.iterdir() if p.suffix == ".json"]
        assert jsons == [], f"unexpected files written: {jsons}"
