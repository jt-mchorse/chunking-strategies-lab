"""A bad byte in a queries file is reported at its file offset and line (#273).

Both queries readers iterated a handle opened with ``encoding="utf-8-sig"``,
which decodes in 8 KiB pieces. A bad byte past the first piece raised a
`UnicodeDecodeError` whose ``start`` counted from that piece: measured on main,
a Latin-1 byte at file offset 43364 on line 301 was reported as "position
2404", a different and valid byte, with no line. The corpus reads decode whole
but counted from past a BOM, with no line; see
``test_corpus_decode_error_position.py`` (#298).
"""

from __future__ import annotations

import codecs
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_lab.queries import load_queries
from chunking_lab.validate import validate_queries

ROWS = 401
BAD_LINE = 301


def _row(i: int, *, bad: bool = False) -> bytes:
    rec = {
        "id": f"q{i}",
        "question": f"What does the corpus say about topic number {i} in detail?",
        "expected_doc": "01_intro.md",
        "expected_snippet": "x",
    }
    line = json.dumps(rec).encode("utf-8")
    return line.replace(b"detail", b"d\xe9tail") if bad else line


def _write(
    tmp_path: Path, *, bad_line: int, eol: bytes = b"\n", bom: bool = False
) -> tuple[Path, int]:
    body = eol.join(_row(i, bad=(i == bad_line)) for i in range(1, ROWS + 1)) + eol
    data = (codecs.BOM_UTF8 if bom else b"") + body
    path = tmp_path / "q.jsonl"
    path.write_bytes(data)
    return path, data.index(b"\xe9")


CASES = {
    # (bad line, line ending, BOM) -- line 1 is inside the first 8 KiB piece,
    # line 301 is well past it; the others move the line ending and the BOM.
    "first-piece": (1, b"\n", False),
    "past-first-piece": (BAD_LINE, b"\n", False),
    "past-first-piece-bom": (BAD_LINE, b"\n", True),
    "crlf": (BAD_LINE, b"\r\n", False),
    "bare-cr": (BAD_LINE, b"\r", False),
    "last-line": (ROWS, b"\n", False),
}


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_load_queries_names_the_file_offset_and_line(tmp_path: Path, case: tuple) -> None:
    bad_line, eol, bom = case
    path, offset = _write(tmp_path, bad_line=bad_line, eol=eol, bom=bom)
    with pytest.raises(UnicodeDecodeError) as info:
        load_queries(path)
    err = info.value
    assert err.start == offset
    assert err.object == path.read_bytes()
    assert err.object[err.start] == 0xE9
    assert str(err).endswith(f"(line {bad_line})")


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_validate_queries_reports_the_same_offset_and_line(tmp_path: Path, case: tuple) -> None:
    bad_line, eol, bom = case
    path, offset = _write(tmp_path, bad_line=bad_line, eol=eol, bom=bom)
    with pytest.raises(UnicodeDecodeError) as via_validate:
        validate_queries(path)
    with pytest.raises(UnicodeDecodeError) as via_load:
        load_queries(path)
    assert str(via_validate.value) == str(via_load.value)
    assert via_validate.value.start == offset


def test_the_cli_prints_the_offset_and_line_at_exit_2(tmp_path: Path) -> None:
    path, offset = _write(tmp_path, bad_line=BAD_LINE)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "chunking_lab.validate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert f"in position {offset}:" in proc.stderr
    assert f"(line {BAD_LINE})" in proc.stderr


def test_a_file_that_decodes_is_read_line_for_line_as_before(tmp_path: Path) -> None:
    # Parity with the text handle the readers used: only \r\n, \r and \n end a
    # line. U+2028 and U+0085 are legal raw inside a JSON string and must not
    # split a row (str.splitlines would), and a BOM is still dropped.
    rows = [
        {"id": "a", "question": "line sep?", "expected_doc": "d.md", "expected_snippet": "s"},
        {"id": "b", "question": "next\x85line?", "expected_doc": "d.md", "expected_snippet": "s"},
        {"id": "c", "question": "plain?", "expected_doc": "d.md", "expected_snippet": "s"},
    ]
    raw = "\r".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\r\n"
    path = tmp_path / "q.jsonl"
    path.write_bytes(codecs.BOM_UTF8 + raw.encode("utf-8"))
    queries = load_queries(path)
    assert [q.id for q in queries] == ["a", "b", "c"]
    report = validate_queries(path)
    assert report.n_rows == 3


def test_the_fixture_puts_the_bad_byte_past_the_first_decode_piece(tmp_path: Path) -> None:
    # Anti-vacuity for the size: the defect needs a file past one decode piece.
    # The fixture's bad byte on line 301 must sit well past 8 KiB.
    _, offset = _write(tmp_path, bad_line=BAD_LINE)
    assert offset > 4 * 8192
