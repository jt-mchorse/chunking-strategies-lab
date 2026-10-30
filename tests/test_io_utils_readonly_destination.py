"""`atomic_write_text` refuses a read-only existing destination (#328).

`os.replace` swaps the directory entry, so the target's own write bit was
never consulted and a ``chmod 444`` file was silently rewritten, while
`Path.write_text`, the call the helper replaced, raises ``PermissionError``.
The write seams (`validate --out`, `scripts/run_matrix.py`'s results and
summary writes) turn an unwritable output into exit 2, the documented I/O
error code, and a read-only file was the one unwritable case that did not
reach them. Helper cases run both writers on the same layout, so the lock is
parity with `Path.write_text`. Port of prompt-regression-suite#318.

Root bypasses permission bits (`Path.write_text` succeeds too), so the
read-only arms skip as root. GitHub's ubuntu runners run as the non-root
``runner`` user, so CI exercises them.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from chunking_lab.io_utils import atomic_write_text
from chunking_lab.validate import main as validate_main

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits"),
    pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses file permission bits, so a 0444 file stays writable",
    ),
]

_QUERIES = Path(__file__).resolve().parent.parent / "data" / "queries.jsonl"


def _write(writer: str, path: Path, text: str) -> None:
    if writer == "atomic":
        atomic_write_text(path, text)
    else:
        path.write_text(text)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def frozen(tmp_path: Path) -> Iterator[Path]:
    """A 0444 file holding ``old``; made writable again for tmp_path cleanup."""
    path = tmp_path / "report.txt"
    path.write_text("old\n")
    path.chmod(0o444)
    yield path
    path.chmod(0o644)


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_readonly_file_is_refused(frozen: Path, writer: str) -> None:
    with pytest.raises(PermissionError) as excinfo:
        _write(writer, frozen, "new\n")
    assert excinfo.value.errno == errno.EACCES
    assert frozen.read_text() == "old\n"
    assert _mode(frozen) == 0o444


def test_refusal_leaves_no_temp_file(frozen: Path) -> None:
    with pytest.raises(PermissionError):
        atomic_write_text(frozen, "new\n")
    assert sorted(p.name for p in frozen.parent.iterdir()) == ["report.txt"]


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_readonly_file_behind_a_symlink_is_refused(tmp_path: Path, writer: str) -> None:
    # The check runs on the RESOLVED file (#248): a writable link name over a
    # read-only file must not get through.
    real = tmp_path / "real.md"
    real.write_text("old\n")
    real.chmod(0o444)
    link = tmp_path / "link.md"
    link.symlink_to(real)
    try:
        with pytest.raises(PermissionError):
            _write(writer, link, "new\n")
        assert real.read_text() == "old\n"
        assert link.is_symlink()
    finally:
        real.chmod(0o644)


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_writable_and_missing_targets_still_write(tmp_path: Path, writer: str) -> None:
    # Controls: the refusal is scoped to an existing unwritable file.
    owner_only = tmp_path / "owner.md"
    owner_only.write_text("old\n")
    owner_only.chmod(0o600)
    _write(writer, owner_only, "new\n")
    assert owner_only.read_text() == "new\n"
    assert _mode(owner_only) == 0o600

    missing = tmp_path / "new.md"
    _write(writer, missing, "fresh\n")
    assert missing.read_text() == "fresh\n"


def test_validate_out_on_a_readonly_file_exits_2(
    frozen: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = validate_main([str(_QUERIES), "--out", str(frozen)])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith(f"failed to write {frozen}")
    assert frozen.read_text() == "old\n"


def test_run_matrix_onto_a_readonly_canonical_fixture_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.run_matrix import main as run_matrix_main

    fixture = tmp_path / "canonical__fixed-size.json"
    fixture.write_text("old\n")
    fixture.chmod(0o444)
    try:
        rc = run_matrix_main(
            [
                "--results-dir",
                str(tmp_path),
                "--embedder",
                "hash",
                "--canonical-out",
                "--strategy",
                "fixed-size",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 2
        assert f"failed to write {fixture}" in err
        assert fixture.read_text() == "old\n"
    finally:
        fixture.chmod(0o644)
