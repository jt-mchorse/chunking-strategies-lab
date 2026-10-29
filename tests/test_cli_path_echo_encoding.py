"""Echoing an operator path never crashes a strict-UTF-8 writer (#322).

A path byte that is not valid UTF-8 reaches Python through `sys.argv` as a lone
surrogate in `U+DC80..U+DCFF` (`surrogateescape`, the #178 mechanism). Three
writers encode strictly: `sys.stdout` under a UTF-8 locale without UTF-8 mode
(`LANG=en_US.UTF-8`, a typical dev machine), and `atomic_write_text` always. So
a **clean** queries file whose path held such a byte exited 1 -- the "findings"
code -- with a `UnicodeEncodeError` from the `validate` summary line, on stdout
and in the `--out` report alike, and `run_matrix` did the same on its
`-> {path}` and `summary wrote {path}` lines for such a `--results-dir`. Under
C.UTF-8 or UTF-8 mode stdout uses `surrogateescape` and the bug hides, which is
why every child here pins `PYTHONIOENCODING=utf-8:strict` and `PYTHONUTF8=0`.

Built from `b"\\xff"`, which `surrogateescape` turns into `U+DCFF`. A surrogate
outside `U+DC80..U+DCFF` (say `U+D800`) is not what argv can produce, so a test
built on one would exercise a path no operator can type.

Every child's output is captured as BYTES. Where the filesystem accepts the
name (ext4, i.e. CI), a stdout that is not strict writes the raw `\\xff`, and a
parent `text=True` would raise `UnicodeDecodeError` inside `subprocess` -- a
harness failure, not a result. APFS refuses the name, so the arms that need a
real file skip there and a second arm simulates the same argv by pointing the
read at a clean file.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_lab.io_utils import display_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.run_matrix as rm  # noqa: E402

BAD_BYTE = b"\xff"
SURROGATE = BAD_BYTE.decode("utf-8", "surrogateescape")
assert SURROGATE == "\udcff"
# What `backslashreplace` writes for it -- the same text `sys.stderr` writes.
ESCAPED = b"\\udcff"

_CLEAN_QUERIES = _REPO_ROOT / "data" / "queries.jsonl"
_N_CLEAN_ROWS = sum(1 for line in _CLEAN_QUERIES.read_text(encoding="utf-8").splitlines() if line)

_VALID_ROW = {
    "id": "q01",
    "question": "What parameter controls HNSW build candidate list size?",
    "expected_doc": "01_hnsw.md",
    "expected_snippet": "ef_construction",
}


def _strict_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env.update(
        PYTHONIOENCODING="utf-8:strict",
        PYTHONUTF8="0",
        LANG="en_US.UTF-8",
        LC_ALL="en_US.UTF-8",
        PYTHONPATH=os.pathsep.join([str(_REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    return env


def _run(args: list[str | bytes]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        check=False,
        env=_strict_env(),
        cwd=_REPO_ROOT,
    )


def _make_or_skip(path: Path, *, directory: bool = False) -> None:
    """Create *path*, or skip where the filesystem refuses the byte (APFS)."""
    try:
        if directory:
            path.mkdir()
        else:
            path.write_text(json.dumps(_VALID_ROW) + "\n", encoding="utf-8")
    except OSError as e:
        pytest.skip(f"filesystem refuses a non-UTF-8 name: {e}")


def test_children_really_get_a_strict_stdout() -> None:
    """The environment pins what it claims to: without this, every arm below
    could pass because stdout quietly used `surrogateescape`."""
    proc = _run(["-c", "import sys; print(sys.stdout.errors, sys.stderr.errors)"])
    assert proc.stdout.split() == [b"strict", b"backslashreplace"], proc


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("data/queries.jsonl", "data/queries.jsonl"),
        ("donn\u00e9es/requ\u00eates.jsonl", "donn\u00e9es/requ\u00eates.jsonl"),
        ("q" + SURROGATE + ".jsonl", "q\\udcff.jsonl"),
        ("q\udc80.jsonl", "q\\udc80.jsonl"),
        # Not reachable from argv, but a `Path` built in code can hold one; the
        # renderer must not care where the surrogate came from.
        ("q\ud800.jsonl", "q\\ud800.jsonl"),
    ],
)
def test_display_path_always_encodes_and_leaves_encodable_text_alone(
    raw: str, expected: str
) -> None:
    for value in (raw, Path(raw)):
        shown = display_path(value)
        assert shown == (str(Path(expected)) if isinstance(value, Path) else expected)
        shown.encode("utf-8")  # strict: must not raise


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

# Runs `validate.main` on a surrogate-bearing argv with the read pointed at the
# clean pinned file: the ext4 case, on any filesystem. The rendered line uses
# the argv string, not the path that was read, so this is the operator's view.
_VALIDATE_DRIVER = """
import sys
import chunking_lab.validate as v
real = v.validate_queries
v.validate_queries = lambda _p, corpus_dir=None: real(sys.argv[1], corpus_dir=corpus_dir)
sys.exit(v.main(["q" + "\\udcff" + ".jsonl", *sys.argv[2:]]))
"""


def test_validate_summary_line_escapes_the_byte_on_a_strict_stdout() -> None:
    proc = _run(["-c", _VALIDATE_DRIVER, str(_CLEAN_QUERIES)])
    assert b"Traceback" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, "a clean file must exit 0, not the findings code"
    assert proc.stdout == (
        b"ok: q"
        + ESCAPED
        + b".jsonl rows=%d valid=%d findings=0\n" % (_N_CLEAN_ROWS, _N_CLEAN_ROWS)
    )


def test_validate_out_report_escapes_the_byte(tmp_path: Path) -> None:
    out = tmp_path / "report.txt"
    proc = _run(["-c", _VALIDATE_DRIVER, str(_CLEAN_QUERIES), "--out", str(out)])
    assert b"Traceback" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0
    assert proc.stdout == b""
    assert out.read_bytes().startswith(b"ok: q" + ESCAPED + b".jsonl rows=")
    assert [p.name for p in tmp_path.iterdir()] == ["report.txt"], "no temp file left behind"


def test_validate_json_mode_was_already_safe() -> None:
    """Control: `json.dumps` escapes the surrogate itself. Green before #322."""
    proc = _run(["-c", _VALIDATE_DRIVER, str(_CLEAN_QUERIES), "--json"])
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert json.loads(proc.stdout)["ok"] is True


def test_validate_real_surrogate_named_file(tmp_path: Path) -> None:
    """The issue's own shape, end to end, where the filesystem can name it."""
    queries = Path(os.fsdecode(os.fsencode(tmp_path) + b"/q" + BAD_BYTE + b".jsonl"))
    _make_or_skip(queries)
    out = tmp_path / "report.txt"
    for extra in ([], ["--out", str(out)]):
        proc = _run(["-m", "chunking_lab.validate", os.fsencode(queries), *extra])
        assert b"Traceback" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
        assert proc.returncode == 0
    assert ESCAPED + b".jsonl rows=1 valid=1 findings=0\n" in out.read_bytes()


def test_validate_missing_surrogate_path_reports_on_stderr() -> None:
    """Control for the deferred half: the `failed to read` / `file not found`
    lines go to stderr, whose handler is `backslashreplace` even when
    `PYTHONIOENCODING` asks for strict, so they cannot raise. Green before #322."""
    proc = _run(["-m", "chunking_lab.validate", b"/nonexistent/q" + BAD_BYTE + b".jsonl"])
    assert proc.returncode == 2
    assert b"Traceback" not in proc.stderr
    assert ESCAPED in proc.stderr


# ---------------------------------------------------------------------------
# run_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", [["--strategy", "fixed-size"], []], ids=["filtered", "full"])
def test_run_matrix_real_surrogate_results_dir(tmp_path: Path, strategy: list[str]) -> None:
    results = Path(os.fsdecode(os.fsencode(tmp_path) + b"/r" + BAD_BYTE))
    _make_or_skip(results, directory=True)
    proc = _run(["scripts/run_matrix.py", "--results-dir", os.fsencode(results), *strategy])
    assert b"Traceback" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0
    assert b"/r" + ESCAPED + b"/" in proc.stdout
    if not strategy:
        assert b"summary wrote " in proc.stdout
        assert proc.stdout.rstrip().endswith(b"/r" + ESCAPED + b"/" + _summary_name(results))


def _summary_name(results: Path) -> bytes:
    (summary,) = [p for p in results.iterdir() if p.name.endswith("__summary.md")]
    return summary.name.encode("ascii")


def test_run_matrix_both_path_lines_escape_on_a_strict_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same two lines on any filesystem: the four calls that touch the
    surrogate-named directory are stubbed, and stdout is a strict UTF-8 writer
    over bytes -- what `sys.stdout` is under `LANG=en_US.UTF-8`."""
    results = tmp_path / ("r" + SURROGATE)
    written: list[str] = []
    real_mkdir = Path.mkdir

    def _mkdir(self: Path, *a: object, **k: object) -> None:
        if self != results:
            real_mkdir(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", _mkdir)
    monkeypatch.setattr(rm, "atomic_write_text", lambda p, _t: written.append(str(p)))
    monkeypatch.setattr(rm, "_free_stamp", lambda _d, base: base)
    monkeypatch.setattr(rm, "_release_stamp", lambda _d, _s: None)
    raw = io.BytesIO()
    strict = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", strict)

    rc = rm.main(["--results-dir", str(results)])
    strict.flush()
    out = raw.getvalue()

    assert rc == 0
    assert len(written) == 6, written
    lines = [line for line in out.splitlines() if b"r" + ESCAPED in line]
    assert len(lines) == 6, out
    assert lines[-1].startswith(b"summary wrote ")
    assert SURROGATE.encode("utf-8", "surrogateescape") not in out
