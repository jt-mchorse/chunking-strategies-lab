"""`validate --corpus-dir` names the corpus document it could not read (#258).

#256 (closing #255) made `validate_queries` open each row's `expected_doc` to
check that the snippet occurs in it. `main()`'s I/O-error handling dates from
when the queries file was the only file the validator read, so it reported any
read failure as `failed to read <queries>`. A latin-1 byte in `corpus/a.md` read
`failed to read q.jsonl: not valid UTF-8: ...` at exit 2. A `UnicodeDecodeError`
carries no filename, so the document was never named. The same run without
`--corpus-dir` exited 0 `ok`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from chunking_lab.validate import _build_parser, main


def _setup(tmp_path: Path, doc_bytes: bytes) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "a.md"
    doc.write_bytes(doc_bytes)
    queries = tmp_path / "q.jsonl"
    row = {"id": "q1", "question": "?", "expected_doc": "a.md", "expected_snippet": "hello"}
    queries.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return queries, corpus, doc


def _assert_names_the_document(
    capsys: pytest.CaptureFixture[str], rc: int, queries: Path, doc: Path
) -> str:
    err = capsys.readouterr().err
    assert rc == 2, err
    assert f"failed to read corpus document {doc}" in err, err
    assert f"failed to read {queries}" not in err, err
    return err


def test_a_non_utf8_corpus_document_is_named_not_the_queries_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 0xE9 is latin-1 `é`, a lone continuation-less lead byte in UTF-8.
    queries, corpus, doc = _setup(tmp_path, b"hello caf" + bytes([0xE9]) + b"\n")
    assert main([str(queries)]) == 0  # the queries file itself is fine
    capsys.readouterr()
    rc = main([str(queries), "--corpus-dir", str(corpus)])
    err = _assert_names_the_document(capsys, rc, queries, doc)
    assert "not valid UTF-8" in err


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny reads on Windows or to root",
)
def test_an_unreadable_corpus_document_is_named_not_the_queries_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries, corpus, doc = _setup(tmp_path, b"hello\n")
    doc.chmod(0)
    try:
        rc = main([str(queries), "--corpus-dir", str(corpus)])
    finally:
        doc.chmod(0o644)
    _assert_names_the_document(capsys, rc, queries, doc)


def test_a_non_utf8_queries_file_still_blames_the_queries_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The new catch must not swallow the queries-file decode error (#124).
    queries, corpus, _doc = _setup(tmp_path, b"hello\n")
    queries.write_bytes(b"\xff\xfe\x00not utf-8")
    rc = main([str(queries), "--corpus-dir", str(corpus)])
    err = capsys.readouterr().err
    assert rc == 2, err
    assert f"failed to read {queries}: not valid UTF-8" in err, err


def test_corpus_dir_help_names_both_cross_file_codes() -> None:
    help_text = _build_parser().format_help()
    assert "expected_doc_not_found" in help_text
    assert "expected_snippet_not_in_doc" in help_text
