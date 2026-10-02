"""File-mode contract for `chunking_lab.io_utils.atomic_write_text` (#212).

The helper used to create its temp file with `tempfile.NamedTemporaryFile`,
which always creates **0600** whatever the umask, and `os.replace` carried
that mode onto the target. So every new file was owner-only, and overwriting
an existing 0644 file silently demoted it to 0600 -- neither of which the
`Path.write_text` it replaced ever did (portfolio-ops#81).

The contract pinned here:

- a new file is `0o666 & ~umask` -- checked under several umasks, so a
  hard-coded 0644 cannot pass;
- an overwrite keeps the existing file's mode, whatever it was.

Both are checked on the helper and through a real caller, the
`python -m chunking_lab.validate --out` report writer (run in-process so the
test's umask applies).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from chunking_lab.io_utils import atomic_write_text
from chunking_lab.validate import main as validate_main

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission bits; Windows has no umask"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def set_umask() -> Iterator[Callable[[int], None]]:
    """Set the process umask for one test and always restore it."""
    original = os.umask(0o022)
    os.umask(original)

    def _set(mask: int) -> None:
        os.umask(mask)

    try:
        yield _set
    finally:
        os.umask(original)


def _write_helper(path: Path) -> None:
    atomic_write_text(path, "report\n")


def _write_validate_report(path: Path) -> None:
    queries = path.parent / "queries.jsonl"
    row = {
        "id": "q01",
        "question": "What parameter controls HNSW build candidate list size?",
        "expected_doc": "01_hnsw.md",
        "expected_snippet": "ef_construction",
    }
    queries.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert validate_main([str(queries), "--out", str(path)]) == 0


WRITERS = pytest.mark.parametrize(
    "write",
    [_write_helper, _write_validate_report],
    ids=["atomic_write_text", "validate_out"],
)


@WRITERS
@pytest.mark.parametrize(("umask", "expected"), [(0o022, 0o644), (0o077, 0o600), (0o002, 0o664)])
def test_new_file_mode_honours_the_umask(
    write: Callable[[Path], None],
    umask: int,
    expected: int,
    set_umask: Callable[[int], None],
    tmp_path: Path,
) -> None:
    set_umask(umask)
    out = tmp_path / "out.txt"
    write(out)
    assert _mode(out) == expected == 0o666 & ~umask, (
        f"new file under umask {umask:#o} is {_mode(out):#o}, want {expected:#o}"
    )


@WRITERS
@pytest.mark.parametrize("existing", [0o644, 0o600, 0o640, 0o664])
def test_overwrite_keeps_the_existing_mode(
    write: Callable[[Path], None],
    existing: int,
    set_umask: Callable[[int], None],
    tmp_path: Path,
) -> None:
    # A umask that would produce a *different* mode for a new file, so the
    # assertion can only pass if the target's own mode was carried over.
    set_umask(0o027)
    out = tmp_path / "out.txt"
    out.write_text("old\n", encoding="utf-8")
    out.chmod(existing)
    write(out)
    assert out.read_text(encoding="utf-8") != "old\n"
    assert _mode(out) == existing, f"overwrite of a {existing:#o} file left {_mode(out):#o}"


def test_overwrite_leaves_no_temp_file(set_umask: Callable[[int], None], tmp_path: Path) -> None:
    set_umask(0o022)
    out = tmp_path / "out.txt"
    out.write_text("old\n", encoding="utf-8")
    out.chmod(0o640)
    atomic_write_text(out, "new\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]
