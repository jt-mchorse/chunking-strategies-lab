"""`atomic_write_text` writes THROUGH a symlinked destination (#248).

`os.replace` renames onto the link itself. A symlinked `--out` used to become a
regular file, and the file it pointed at kept its old contents.
`Path.write_text`, which this helper replaced and whose file-mode behaviour
#212 restored, writes through the link. Each write-through case runs
`Path.write_text` on an identical layout as well, so the lock checks parity
with it instead of a hand-written expectation. Sibling of
python-async-llm-pipelines#157.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from chunking_lab.io_utils import atomic_write_text
from chunking_lab.validate import main

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs privileges on Windows"
)


def _layout(root: Path, *, absolute: bool, name: str = "report.txt") -> tuple[Path, Path]:
    real_dir = root / "real"
    real_dir.mkdir(parents=True)
    real = real_dir / name
    real.write_text("old\n")
    real.chmod(0o640)
    link = root / "link.txt"
    link.symlink_to(real if absolute else Path("real") / name)
    return link, real


@pytest.mark.parametrize("absolute", [False, True], ids=["relative-link", "absolute-link"])
@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_write_goes_through_the_link(tmp_path: Path, absolute: bool, writer: str) -> None:
    link, real = _layout(tmp_path, absolute=absolute)
    if writer == "atomic":
        atomic_write_text(link, "new\n")
    else:
        link.write_text("new\n")
    assert link.is_symlink(), "the link was replaced by a regular file"
    assert real.read_text() == "new\n", "the linked file kept its old contents"
    assert link.read_text() == "new\n"
    # The linked file's mode is kept (#212), on the file that was written.
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o640
    # No temp file left behind beside the link or beside the linked file.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.txt", "real"]
    assert sorted(p.name for p in real.parent.iterdir()) == ["report.txt"]


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_dangling_link_creates_its_target(tmp_path: Path, writer: str) -> None:
    (tmp_path / "real").mkdir()
    real = tmp_path / "real" / "new.txt"
    link = tmp_path / "link.txt"
    link.symlink_to(Path("real") / "new.txt")
    if writer == "atomic":
        atomic_write_text(link, "fresh\n")
    else:
        link.write_text("fresh\n")
    assert link.is_symlink()
    assert real.read_text() == "fresh\n"


def test_link_loop_raises_eloop_and_leaves_no_temp(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.symlink_to("b.txt")
    b.symlink_to("a.txt")
    with pytest.raises(OSError, match="symbolic links") as exc:
        atomic_write_text(a, "x\n")
    assert exc.value.errno == errno.ELOOP
    assert a.is_symlink()
    assert b.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]


def test_long_linked_name_still_gets_a_capped_temp_name(tmp_path: Path) -> None:
    """The temp name is built from the RESOLVED basename, so the NAME_MAX cap applies to it."""
    link, real = _layout(tmp_path, absolute=False, name="r" * 251 + ".txt")
    atomic_write_text(link, "new\n")
    assert link.is_symlink()
    assert real.read_text() == "new\n"


def test_plain_destination_is_unchanged_behaviour(tmp_path: Path) -> None:
    dest = tmp_path / "plain.txt"
    dest.write_text("old\n")
    dest.chmod(0o640)
    atomic_write_text(dest, "new\n")
    assert not dest.is_symlink()
    assert dest.read_text() == "new\n"
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o640


_QUERIES = Path(__file__).resolve().parents[1] / "data" / "queries.jsonl"


def test_validate_out_through_a_link_updates_the_linked_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: `chunking_lab.validate --json --out link.txt` updates the file the link names."""
    link, real = _layout(tmp_path, absolute=False)
    assert main([str(_QUERIES), "--json", "--out", str(link)]) == 0
    capsys.readouterr()
    assert link.is_symlink()
    assert json.loads(real.read_text())
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o640
