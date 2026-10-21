"""A bad byte in a corpus document is reported at its file offset and line (#298).

#273 gave the queries readers this. The corpus readers (`load_corpus`, and
`validate --corpus-dir`) used ``read_text(encoding="utf-8-sig")``, which decodes
whole but counts from the end of the BOM: measured on main, a BOM-prefixed
document with its bad byte at file offset 15007 on line 3001 was reported as
"position 15004", a different and valid byte, with no line.
"""

from __future__ import annotations

import codecs
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_lab.corpus import load_corpus
from chunking_lab.validate import validate_queries

BAD_LINE = 3001


def _corpus(tmp_path: Path, *, bom: bool, eol: bytes = b"\n") -> tuple[Path, Path, int]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    data = (codecs.BOM_UTF8 if bom else b"") + b"line" + eol
    data += (b"line" + eol) * (BAD_LINE - 2) + b"bad \xff byte" + eol
    doc = corpus / "a.md"
    doc.write_bytes(data)
    return corpus, doc, data.index(b"\xff")


def _queries(tmp_path: Path) -> Path:
    path = tmp_path / "q.jsonl"
    rec = {"id": "q1", "question": "x?", "expected_doc": "a.md", "expected_snippet": "line"}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return path


CASES = {
    # (BOM, line ending). The BOM cases are the defect; the no-BOM case is the
    # control whose offset was already right and must stay so.
    "bom": (True, b"\n"),
    "bom-crlf": (True, b"\r\n"),
    "no-bom": (False, b"\n"),
}


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_load_corpus_names_the_file_offset_and_line(tmp_path: Path, case: tuple) -> None:
    bom, eol = case
    corpus, doc, offset = _corpus(tmp_path, bom=bom, eol=eol)
    with pytest.raises(UnicodeDecodeError) as info:
        load_corpus(corpus)
    err = info.value
    assert err.start == offset
    assert err.object[err.start] == 0xFF
    assert f"(line {BAD_LINE})" in str(err)
    assert str(err).endswith(f"(corpus document {doc})")


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_validate_corpus_dir_reports_the_same_offset_and_line(tmp_path: Path, case: tuple) -> None:
    bom, eol = case
    corpus, doc, offset = _corpus(tmp_path, bom=bom, eol=eol)
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "chunking_lab.validate",
            str(_queries(tmp_path)),
            "--corpus-dir",
            str(corpus),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert f"failed to read corpus document {doc}: not valid UTF-8:" in proc.stderr
    assert f"in position {offset}:" in proc.stderr
    assert f"(line {BAD_LINE})" in proc.stderr


def test_the_two_corpus_readers_raise_the_same_codec_detail(tmp_path: Path) -> None:
    corpus, _, _ = _corpus(tmp_path, bom=True)
    with pytest.raises(UnicodeDecodeError) as via_load:
        load_corpus(corpus)
    with pytest.raises(OSError, match="not valid UTF-8") as via_validate:
        validate_queries(_queries(tmp_path), corpus_dir=corpus)
    detail = str(via_load.value).removesuffix(f" (corpus document {corpus / 'a.md'})")
    assert via_validate.value.strerror == f"not valid UTF-8: {detail}"


def test_a_document_that_decodes_reads_as_the_text_mode_read_did(tmp_path: Path) -> None:
    # The neighbour the fix must leave alone: the bytes are now decoded whole,
    # so \r\n and \r must still become \n and the BOM must still go (#95), or
    # every chunk offset into the document moves.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_bytes(codecs.BOM_UTF8 + "café\r\nnaïve\rend\n".encode())
    (corpus / "b.md").write_bytes(b"plain\nlines\n")
    docs = load_corpus(corpus)
    assert [(d.filename, d.text) for d in docs] == [
        ("a.md", "café\nnaïve\nend\n"),
        ("b.md", "plain\nlines\n"),
    ]
    for d in docs:
        assert d.text == (corpus / d.filename).read_text(encoding="utf-8-sig")
